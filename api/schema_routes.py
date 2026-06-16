"""
GET /schema — combined metadata feed for the metadata-ui frontend.

Returns:
  {
    "labels": [
      {
        "name": "Patient",
        "description": "An RP patient with ...",
        "count": 12345,
        "properties": [
          {"name": "payor_cohort", "description": "...", "values": {"bai": "BCBS..."}},
          ...
        ]
      },
      ...
    ],
    "relationships": [
      {"type": "REGISTERED_AT", "description": "...", "count": 12345,
       "from": ["Patient"], "to": ["Practice"]},
      ...
    ],
    "graph": {
      "nodes": [{"id": "Patient", "count": 12345}, ...],
      "edges": [{"source": "Patient", "target": "Practice", "type": "REGISTERED_AT", "count": 12345}, ...]
    }
  }
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from graph.connection import Neo4jConnection
from metadata.catalog import get_catalog
from guardrails import redact_rows

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schema", tags=["schema"])


def _safe_count(cypher: str) -> int:
    try:
        rows = Neo4jConnection.run_query(cypher)
        if rows and rows[0]:
            v = next(iter(rows[0].values()))
            return int(v or 0)
    except Exception as e:
        logger.warning(f"Count failed for {cypher!r}: {e}")
    return 0


@router.get("")
def get_schema() -> dict:
    catalog = get_catalog()

    # ── Per-label counts ─────────────────────────────────────────────
    label_counts: dict[str, int] = {}
    for label in catalog.labels:
        label_counts[label] = _safe_count(f"MATCH (n:`{label}`) RETURN count(n) AS c")

    # ── Per-relationship counts + endpoints ──────────────────────────
    rel_meta: dict[str, dict] = {}
    try:
        rows = Neo4jConnection.run_query("""
            MATCH (a)-[r]->(b)
            WITH type(r) AS rtype,
                 labels(a) AS from_labels,
                 labels(b) AS to_labels,
                 count(r) AS c
            RETURN rtype,
                   collect(DISTINCT from_labels) AS from_groups,
                   collect(DISTINCT to_labels)   AS to_groups,
                   sum(c) AS total
        """)
        for row in rows:
            rtype = row["rtype"]
            from_set = {lab for grp in (row.get("from_groups") or []) for lab in (grp or [])}
            to_set = {lab for grp in (row.get("to_groups") or []) for lab in (grp or [])}
            rel_meta[rtype] = {
                "from": sorted(from_set),
                "to": sorted(to_set),
                "count": int(row.get("total") or 0),
            }
    except Exception as e:
        logger.warning(f"Relationship discovery failed: {e}")

    # ── Build labels payload ─────────────────────────────────────────
    labels_out = []
    for name, meta in catalog.labels.items():
        props_out = []
        for prop_name, prop_meta in (meta.get("properties") or {}).items():
            props_out.append({
                "name": prop_name,
                "description": (prop_meta or {}).get("description", ""),
                "values": (prop_meta or {}).get("values") or {},
            })
        labels_out.append({
            "name": name,
            "description": meta.get("description", ""),
            "count": label_counts.get(name, 0),
            "pii": meta.get("pii", []),
            "properties": props_out,
        })

    # ── Build relationships payload ──────────────────────────────────
    rels_out = []
    for rtype, meta in catalog.relationships.items():
        live = rel_meta.get(rtype, {})
        rels_out.append({
            "type": rtype,
            "description": meta.get("description", ""),
            "count": live.get("count", 0),
            "from": live.get("from", []),
            "to": live.get("to", []),
        })
    for rtype, live in rel_meta.items():
        if rtype not in catalog.relationships:
            rels_out.append({
                "type": rtype,
                "description": "(undocumented — present in graph but not in data dictionary)",
                "count": live.get("count", 0),
                "from": live.get("from", []),
                "to": live.get("to", []),
            })

    # ── Graph payload (nodes + edges for force-directed view) ────────
    graph_nodes = [{"id": l["name"], "count": l["count"]} for l in labels_out]
    graph_edges = []
    for r in rels_out:
        for src in (r["from"] or [""]):
            for tgt in (r["to"] or [""]):
                if src and tgt:
                    graph_edges.append({
                        "source": src,
                        "target": tgt,
                        "type": r["type"],
                        "count": r["count"],
                    })

    return {
        "labels": labels_out,
        "relationships": rels_out,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "totals": {
            "labels": len(labels_out),
            "relationships": len(rels_out),
            "nodes": sum(label_counts.values()),
            "edges": sum((r.get("count") or 0) for r in rels_out),
        },
    }


# ── Patient journey ──────────────────────────────────────────────────────

_PATIENT_SAMPLE_CYPHER = """
MATCH (p:Patient)
WHERE p.source_db IS NOT NULL AND p.patient_id IS NOT NULL
RETURN (p.source_db + ':' + toString(p.patient_id)) AS id,
       p.state                   AS state,
       p.payor_cohort            AS payor_cohort,
       p.call_tier               AS call_tier,
       coalesce(p.outstanding_balance, 0.0) AS balance,
       coalesce(p.adj_bad_debt, 0.0)        AS bad_debt,
       coalesce(p.visit_count, 0)           AS visit_count,
       coalesce(p.charge_count, 0)          AS charge_count
ORDER BY balance DESC
LIMIT $limit
"""


@router.get("/patients/sample")
def patient_sample(limit: int = 50) -> dict:
    """Top N patients by outstanding balance — useful seeds for the journey picker."""
    rows = Neo4jConnection.run_query(_PATIENT_SAMPLE_CYPHER, {"limit": int(limit)})
    return {"patients": rows or []}


_JOURNEY_HEADER = """
MATCH (p:Patient {source_db: $src, patient_id: $pid})
OPTIONAL MATCH (p)-[:REGISTERED_AT]->(pr:Practice)
RETURN ($src + ':' + toString($pid))         AS patient_id,
       p.state                               AS state,
       p.city                                AS city,
       p.payor_cohort                        AS payor_cohort,
       p.call_tier                           AS call_tier,
       p.carrier_name                        AS carrier_name,
       coalesce(p.outstanding_balance, 0.0)  AS outstanding_balance,
       coalesce(p.total_charged, 0.0)        AS total_charged,
       coalesce(p.total_paid, 0.0)           AS total_paid,
       coalesce(p.adj_bad_debt, 0.0)         AS adj_bad_debt,
       coalesce(p.visit_count, 0)            AS visit_count,
       coalesce(p.charge_count, 0)           AS charge_count,
       coalesce(p.statement_count, 0)        AS statement_count,
       collect(DISTINCT pr.code)             AS practices
"""

_JOURNEY_EVENTS = """
MATCH (p:Patient {source_db: $src, patient_id: $pid})

OPTIONAL MATCH (p)-[:HAD_VISIT]->(v:Visit)
WITH p, collect({
  kind: 'visit',
  ts:   v.admit_date,
  id:   v.visit_id,
  label: 'Visit',
  detail: coalesce(v.visit_id, ''),
  amount: null
}) AS visits

OPTIONAL MATCH (p)-[:HAS_CHARGE]->(c:Charge)
WITH p, visits, collect({
  kind: 'charge',
  ts:   c.service_date,
  id:   c.charge_id,
  label: coalesce(c.procedure_modality, 'Charge'),
  detail: coalesce(c.procedure_description, c.procedure_code, ''),
  amount: c.charge_amount
}) AS charges

OPTIONAL MATCH (p)-[:HAS_TRANSACTION]->(t:Transaction)
WITH p, visits, charges, collect({
  kind: 'transaction',
  ts:   t.post_date,
  id:   t.payment_id,
  label: coalesce(t.adjustment_bucket, t.transaction_type, 'Transaction'),
  detail: coalesce(t.payment_method, t.paysource, ''),
  amount: coalesce(t.payment_amount, t.adjustment_amount, 0.0)
}) AS transactions

OPTIONAL MATCH (p)-[:RECEIVED_STATEMENT]->(s:Statement)
WITH p, visits, charges, transactions, collect({
  kind: 'statement',
  ts:   s.created_date,
  id:   s.statement_id,
  label: 'Statement',
  detail: coalesce(s.statement_level, ''),
  amount: s.patient_balance
}) AS statements

OPTIONAL MATCH (p)-[:CALLED_IVR]->(ivr:IVRInbound)
WITH p, visits, charges, transactions, statements, collect({
  kind: 'ivr',
  ts:   ivr.call_datetime,
  id:   ivr.response_id,
  label: 'IVR call',
  detail: coalesce(ivr.result_desc, ''),
  amount: ivr.amount_paid
}) AS ivrs

OPTIONAL MATCH (p)-[:CONTACTED_BY_DIALLER]->(d:DiallerCall)
WITH p, visits, charges, transactions, statements, ivrs, collect({
  kind: 'dialler',
  ts:   d.call_datetime,
  id:   d.account,
  label: 'Dialler call',
  detail: coalesce(d.result_desc, ''),
  amount: null
}) AS diallers

WITH visits + charges + transactions + statements + ivrs + diallers AS all_events
UNWIND all_events AS e
WITH e WHERE e.ts IS NOT NULL AND e.id IS NOT NULL
  AND NOT toString(e.ts) STARTS WITH '1970'
RETURN e.kind AS kind, toString(e.ts) AS ts, toString(e.id) AS id,
       e.label AS label, e.detail AS detail, e.amount AS amount
ORDER BY ts ASC
LIMIT 500
"""


def _split_composite(patient_id: str) -> tuple[str, str]:
    """Composite ID format is '{source_db}:{patient_id}', e.g. 'SAPA:1000102'."""
    if ":" not in patient_id:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid patient id {patient_id!r}; expected format 'source_db:patient_id' (e.g. 'SAPA:1000102')",
        )
    src, pid = patient_id.split(":", 1)
    return src.strip(), pid.strip()


@router.get("/patient-journey/{patient_id}")
def patient_journey(patient_id: str) -> dict:
    src, pid = _split_composite(patient_id)
    params = {"src": src, "pid": pid}

    head = Neo4jConnection.run_query(_JOURNEY_HEADER, params)
    if not head:
        raise HTTPException(status_code=404, detail=f"Patient not found: {patient_id}")
    patient = head[0]

    events = Neo4jConnection.run_query(_JOURNEY_EVENTS, params) or []

    # PII redaction for the patient header (last name etc. aren't selected
    # here, but any future addition stays safe).
    patient = redact_rows([patient])[0]

    # Bucket counts for a quick "summary strip" in the UI.
    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1

    return {
        "patient": patient,
        "events": events,
        "counts": counts,
    }
