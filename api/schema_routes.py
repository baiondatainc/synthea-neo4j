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


# ── Policy journey ───────────────────────────────────────────────────────

_POLICY_SAMPLE_CYPHER = """
MATCH (p:Policy)
WHERE p.policy_id IS NOT NULL
RETURN toString(p.policy_id)                          AS id,
       coalesce(p.policy_number, '')                  AS policy_number,
       coalesce(p.line_of_business, '')               AS line_of_business,
       coalesce(p.policy_status, '')                  AS policy_status,
       coalesce(p.is_active, false)                   AS is_active,
       coalesce(p.gross_written_premium, 0.0)         AS gwp,
       coalesce(p.total_paid, 0.0)                    AS total_paid,
       coalesce(p.total_outstanding, 0.0)             AS outstanding,
       coalesce(p.claim_count, 0)                     AS claim_count,
       coalesce(p.calculated_loss_ratio_pct, 0.0)     AS loss_ratio_pct
ORDER BY gwp DESC
LIMIT $limit
"""


@router.get("/policies/sample")
def policy_sample(limit: int = 50) -> dict:
    """Top N policies by gross written premium — seed list for the journey picker."""
    rows = Neo4jConnection.run_query(_POLICY_SAMPLE_CYPHER, {"limit": int(limit)})
    return {"policies": rows or []}


_POLICY_HEADER = """
MATCH (p:Policy {policy_id: $pid})
OPTIONAL MATCH (p)-[:ISSUED_BY]->(i:Insurer)
OPTIONAL MATCH (p)-[:SOLD_VIA]->(d:DistributionChannel)
OPTIONAL MATCH (p)-[:HAS_PRODUCT]->(prd:Product)
OPTIONAL MATCH (p)-[:REINSURED_BY]->(r:Reinsurer)
RETURN toString(p.policy_id)                          AS policy_id,
       coalesce(p.policy_number, '')                  AS policy_number,
       coalesce(p.policy_type, '')                    AS policy_type,
       coalesce(p.line_of_business, '')               AS line_of_business,
       coalesce(p.policy_status, '')                  AS policy_status,
       coalesce(p.is_active, false)                   AS is_active,
       coalesce(p.is_cancelled, false)                AS is_cancelled,
       coalesce(p.is_reinsured, false)                AS is_reinsured,
       toString(p.issue_date)                         AS issue_date,
       toString(p.effect_date)                        AS effect_date,
       toString(p.expiry_date)                        AS expiry_date,
       toString(p.renewal_date)                       AS renewal_date,
       toString(p.cancellation_date)                  AS cancellation_date,
       coalesce(p.cancellation_reason, '')            AS cancellation_reason,
       coalesce(p.gross_written_premium, 0.0)         AS gross_written_premium,
       coalesce(p.net_written_premium, 0.0)           AS net_written_premium,
       coalesce(p.commission, 0.0)                    AS commission,
       coalesce(p.total_claimed, 0.0)                 AS total_claimed,
       coalesce(p.total_approved, 0.0)                AS total_approved,
       coalesce(p.total_paid, 0.0)                    AS total_paid,
       coalesce(p.total_outstanding, 0.0)             AS total_outstanding,
       coalesce(p.calculated_loss_ratio_pct, 0.0)     AS loss_ratio_pct,
       coalesce(p.claim_count, 0)                     AS claim_count,
       coalesce(p.open_claim_count, 0)                AS open_claim_count,
       coalesce(p.renewal_count, 0)                   AS renewal_count,
       coalesce(p.event_count, 0)                     AS event_count,
       collect(DISTINCT i.name)[0..1]                 AS insurer,
       collect(DISTINCT d.name)[0..3]                 AS channels,
       collect(DISTINCT prd.name)[0..3]               AS products,
       collect(DISTINCT r.name)[0..3]                 AS reinsurers
"""

# All events flow to a single UNION'd stream keyed by (kind, ts, id, label, detail, amount).
# Ordering + a hard LIMIT are applied after the union.
_POLICY_EVENTS = """
MATCH (p:Policy {policy_id: $pid})

// Issue / effect (single anchor event)
WITH p, [
  {kind: 'issue',
   ts:    toString(coalesce(p.effect_date, p.issue_date)),
   id:    'policy-' + toString(p.policy_id),
   label: 'Policy issued',
   detail: coalesce(p.policy_number, ''),
   amount: coalesce(p.gross_written_premium, 0.0)}
] AS issue_events

// Lifecycle events (endorsements, renewals, cancellations tracked as PolicyEvent)
OPTIONAL MATCH (p)-[:HAS_EVENT]->(pe:PolicyEvent)
WITH p, issue_events, collect({
  kind: 'event',
  ts:    toString(pe.event_date),
  id:    toString(pe.policy_event_id),
  label: coalesce(pe.event_type, 'Policy event'),
  detail: coalesce(pe.reason, coalesce(pe.event_status, '')),
  amount: null
}) AS lifecycle_events

// Claims
OPTIONAL MATCH (c:Claim)-[:UNDER_POLICY]->(p)
WITH p, issue_events, lifecycle_events, collect({
  kind: 'claim',
  ts:    toString(c.claim_creation_date),
  id:    toString(c.claim_id),
  label: coalesce(c.claim_status, 'Claim'),
  detail: coalesce(c.claim_lob, coalesce(c.adjudication_outcome, '')),
  amount: coalesce(c.claimed_amount, 0.0)
}) AS claim_events

// Pre-authorizations (traverse through claims)
OPTIONAL MATCH (c2:Claim)-[:UNDER_POLICY]->(p), (c2)-[:HAS_PRE_AUTH]->(pa:PreAuthorization)
WITH p, issue_events, lifecycle_events, claim_events, collect({
  kind: 'pre_auth',
  ts:    toString(pa.pre_authorization_creation_date),
  id:    toString(pa.pre_authorization_id),
  label: coalesce(pa.pre_authorization_status, 'Pre-auth'),
  detail: coalesce(pa.adjudication_outcome, ''),
  amount: coalesce(pa.pre_authorization_amount, 0.0)
}) AS preauth_events

// Payments (claim resolutions with an amount paid)
OPTIONAL MATCH (c3:Claim)-[:UNDER_POLICY]->(p)
WHERE c3.amount_paid IS NOT NULL AND c3.amount_paid > 0
  AND coalesce(c3.last_payment_date, c3.claim_resolution_date) IS NOT NULL
WITH p, issue_events, lifecycle_events, claim_events, preauth_events, collect({
  kind: 'payment',
  ts:    toString(coalesce(c3.last_payment_date, c3.claim_resolution_date)),
  id:    'pay-' + toString(c3.claim_id),
  label: 'Claim paid',
  detail: coalesce(c3.claim_number, toString(c3.claim_id)),
  amount: coalesce(c3.amount_paid, 0.0)
}) AS payment_events

// Cancellation / expiry (only if set)
WITH issue_events + lifecycle_events + claim_events + preauth_events + payment_events + [
  CASE WHEN p.cancellation_date IS NOT NULL THEN
    {kind: 'cancellation',
     ts:    toString(p.cancellation_date),
     id:    'cancel-' + toString(p.policy_id),
     label: 'Cancelled',
     detail: coalesce(p.cancellation_reason, ''),
     amount: null}
  END,
  CASE WHEN p.expiry_date IS NOT NULL THEN
    {kind: 'expiry',
     ts:    toString(p.expiry_date),
     id:    'expiry-' + toString(p.policy_id),
     label: 'Policy expires',
     detail: '',
     amount: null}
  END
] AS all_events

UNWIND all_events AS e
WITH e WHERE e IS NOT NULL AND e.ts IS NOT NULL AND e.id IS NOT NULL
  AND NOT toString(e.ts) STARTS WITH '1970'
RETURN e.kind AS kind, e.ts AS ts, e.id AS id,
       e.label AS label, e.detail AS detail, e.amount AS amount
ORDER BY ts ASC
LIMIT 500
"""


@router.get("/policy-journey/{policy_id}")
def policy_journey(policy_id: str) -> dict:
    params = {"pid": policy_id.strip()}

    head = Neo4jConnection.run_query(_POLICY_HEADER, params)
    if not head:
        raise HTTPException(status_code=404, detail=f"Policy not found: {policy_id}")
    policy = head[0]

    events = Neo4jConnection.run_query(_POLICY_EVENTS, params) or []

    policy = redact_rows([policy])[0]

    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1

    return {
        "policy": policy,
        "events": events,
        "counts": counts,
    }
