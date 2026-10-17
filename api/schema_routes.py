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
import re
from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

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


# ── Ontology (n10s) ──────────────────────────────────────────────────────
#
# The RP ontology (ia_ontology.ttl) is loaded into Neo4j via
#   CALL n10s.onto.import.fetch("file:///ontology/ia_ontology.ttl", "Turtle")
# which materialises three node types plus DOMAIN/RANGE edges:
#   (:Class)                    ← owl:Class          (one per node label)
#   (:Relationship)             ← owl:ObjectProperty (one per rel type)
#   (:Property)                 ← owl:DatatypeProperty (one per property key)
#   (:Class)<-[:DOMAIN]-(:Relationship)-[:RANGE]->(:Class)
#   (:Property)-[:DOMAIN]->(:Class)
#
# Names/labels/comments/URIs come straight from the TTL, so this endpoint
# exposes the ontology view that Neo4j Browser shows in the "Database
# information" panel — just in a shape the metadata-ui can render.

_ONT_CLASSES = """
MATCH (c:Class)
WHERE c.name IS NOT NULL
OPTIONAL MATCH (p:Property)-[:DOMAIN]->(c)
WITH c, collect(DISTINCT {
  name:    coalesce(p.name, p.label, ''),
  label:   coalesce(p.label, p.name, ''),
  comment: coalesce(p.comment, ''),
  uri:     coalesce(p.uri, '')
}) AS props
OPTIONAL MATCH (c)<-[:DOMAIN]-(rout:Relationship)-[:RANGE]->(tgt:Class)
WITH c, props, collect(DISTINCT {
  type:    coalesce(rout.name, rout.label, ''),
  label:   coalesce(rout.label, rout.name, ''),
  comment: coalesce(rout.comment, ''),
  target:  coalesce(tgt.name, tgt.label, '')
}) AS outgoing
OPTIONAL MATCH (c)<-[:RANGE]-(rin:Relationship)-[:DOMAIN]->(src:Class)
WITH c, props, outgoing, collect(DISTINCT {
  type:    coalesce(rin.name, rin.label, ''),
  label:   coalesce(rin.label, rin.name, ''),
  comment: coalesce(rin.comment, ''),
  source:  coalesce(src.name, src.label, '')
}) AS incoming
RETURN coalesce(c.name, c.label, '') AS name,
       coalesce(c.label, c.name, '') AS label,
       coalesce(c.comment, '')       AS comment,
       coalesce(c.uri, '')           AS uri,
       [p IN props WHERE p.name <> ''] AS properties,
       [r IN outgoing WHERE r.type <> ''] AS outgoing,
       [r IN incoming WHERE r.type <> ''] AS incoming
ORDER BY name
"""

_ONT_RELATIONSHIPS = """
MATCH (r:Relationship)
WHERE r.name IS NOT NULL
OPTIONAL MATCH (r)-[:DOMAIN]->(src:Class)
WITH r, collect(DISTINCT coalesce(src.name, src.label)) AS from_labels
OPTIONAL MATCH (r)-[:RANGE]->(tgt:Class)
WITH r, from_labels, collect(DISTINCT coalesce(tgt.name, tgt.label)) AS to_labels
RETURN coalesce(r.name, r.label, '')  AS type,
       coalesce(r.label, r.name, '')  AS label,
       coalesce(r.comment, '')        AS comment,
       coalesce(r.uri, '')            AS uri,
       [l IN from_labels WHERE l IS NOT NULL] AS from,
       [l IN to_labels   WHERE l IS NOT NULL] AS to
ORDER BY type
"""

_ONT_PROPERTIES = """
MATCH (p:Property)
WHERE p.name IS NOT NULL
OPTIONAL MATCH (p)-[:DOMAIN]->(c:Class)
WITH p, collect(DISTINCT coalesce(c.name, c.label)) AS domains
OPTIONAL MATCH (p)-[:RANGE]->(rng)
WITH p, domains, collect(DISTINCT coalesce(rng.name, rng.label, rng.uri)) AS ranges
RETURN coalesce(p.name, p.label, '')  AS name,
       coalesce(p.label, p.name, '')  AS label,
       coalesce(p.comment, '')        AS comment,
       coalesce(p.uri, '')            AS uri,
       [d IN domains WHERE d IS NOT NULL] AS domains,
       [r IN ranges  WHERE r IS NOT NULL] AS ranges
ORDER BY name
"""

_ONT_METADATA = """
OPTIONAL MATCH (o:Resource) WHERE 'http://www.w3.org/2002/07/owl#Ontology' IN labels(o)
   OR any(k IN keys(o) WHERE k = 'uri' AND o.uri CONTAINS 'ontology')
WITH collect(o)[0] AS onto
RETURN
  coalesce(onto.label, '')   AS label,
  coalesce(onto.comment, '') AS comment,
  coalesce(onto.uri, '')     AS uri
"""


@router.get("/ontology")
def get_ontology() -> dict:
    """
    Return the n10s-loaded ontology as a structured payload the UI can render.

    Shape:
      {
        "meta":   {"label": "...", "comment": "...", "uri": "..."},
        "classes": [
          {"name": "Policy", "label": "...", "comment": "...", "uri": "...",
           "properties": [{"name": "...", "label": "...", "comment": "...", "uri": "..."}],
           "outgoing":   [{"type": "ISSUED_BY", "label": "...", "comment": "...", "target": "Insurer"}],
           "incoming":   [{"type": "UNDER_POLICY", "label": "...", "comment": "...", "source": "Claim"}]}
        ],
        "relationships": [
          {"type": "ISSUED_BY", "label": "...", "comment": "...", "uri": "...",
           "from": ["Policy"], "to": ["Insurer"]}
        ],
        "properties": [
          {"name": "policy_id", "label": "...", "comment": "...", "uri": "...",
           "domains": ["Policy"], "ranges": ["xsd:string"]}
        ],
        "totals": {"classes": N, "relationships": M, "properties": K}
      }
    """
    try:
        classes = Neo4jConnection.run_query(_ONT_CLASSES) or []
    except Exception as e:
        logger.warning(f"Ontology classes query failed: {e}")
        classes = []
    try:
        rels = Neo4jConnection.run_query(_ONT_RELATIONSHIPS) or []
    except Exception as e:
        logger.warning(f"Ontology relationships query failed: {e}")
        rels = []
    try:
        props = Neo4jConnection.run_query(_ONT_PROPERTIES) or []
    except Exception as e:
        logger.warning(f"Ontology properties query failed: {e}")
        props = []
    try:
        meta_rows = Neo4jConnection.run_query(_ONT_METADATA) or []
        meta = meta_rows[0] if meta_rows else {"label": "", "comment": "", "uri": ""}
    except Exception as e:
        logger.warning(f"Ontology metadata query failed: {e}")
        meta = {"label": "", "comment": "", "uri": ""}

    if not classes and not rels and not props:
        raise HTTPException(
            status_code=404,
            detail=(
                "No ontology found in Neo4j. Load one with "
                "CALL n10s.onto.import.fetch(\"file:///ontology/ia_ontology.ttl\", \"Turtle\")."
            ),
        )

    return {
        "meta": meta,
        "classes": classes,
        "relationships": rels,
        "properties": props,
        "totals": {
            "classes": len(classes),
            "relationships": len(rels),
            "properties": len(props),
        },
    }


# ── Per-class ontology export (Turtle / RDF-XML) ─────────────────────────
#
# Serialises the subset of the n10s-loaded ontology that pertains to one
# class: the class declaration, its incoming + outgoing owl:ObjectProperty
# declarations (both endpoints included so the file is self-contained), and
# the owl:DatatypeProperty declarations for its properties.
#
# Output keeps the source ia: vocabulary (http://ia.gov.sa/ontology#) so a
# round-trip through n10s import produces the same triples we started with.

_ONT_EXPORT_CLASS = """
MATCH (c:Class)
WHERE c.name = $name OR c.label = $name
WITH c LIMIT 1
OPTIONAL MATCH (p:Property)-[:DOMAIN]->(c)
WITH c, collect(DISTINCT p) AS class_props
OPTIONAL MATCH (c)<-[:DOMAIN]-(rout:Relationship)-[:RANGE]->(tgt:Class)
WITH c, class_props,
     collect(DISTINCT {rel: rout, from: c, to: tgt}) AS outgoing
OPTIONAL MATCH (c)<-[:RANGE]-(rin:Relationship)-[:DOMAIN]->(src:Class)
WITH c, class_props, outgoing,
     collect(DISTINCT {rel: rin, from: src, to: c}) AS incoming
RETURN
  {
    name:    coalesce(c.name, c.label, ''),
    label:   coalesce(c.label, c.name, ''),
    comment: coalesce(c.comment, ''),
    uri:     coalesce(c.uri, '')
  } AS cls,
  [p IN class_props | {
    name:    coalesce(p.name, p.label, ''),
    label:   coalesce(p.label, p.name, ''),
    comment: coalesce(p.comment, ''),
    uri:     coalesce(p.uri, '')
  }] AS properties,
  [t IN outgoing WHERE t.rel IS NOT NULL AND t.to IS NOT NULL | {
    type:    coalesce(t.rel.name, t.rel.label, ''),
    label:   coalesce(t.rel.label, t.rel.name, ''),
    comment: coalesce(t.rel.comment, ''),
    uri:     coalesce(t.rel.uri, ''),
    from:    coalesce(t.from.name, t.from.label, ''),
    to:      coalesce(t.to.name,   t.to.label, '')
  }] AS outgoing,
  [t IN incoming WHERE t.rel IS NOT NULL AND t.from IS NOT NULL | {
    type:    coalesce(t.rel.name, t.rel.label, ''),
    label:   coalesce(t.rel.label, t.rel.name, ''),
    comment: coalesce(t.rel.comment, ''),
    uri:     coalesce(t.rel.uri, ''),
    from:    coalesce(t.from.name, t.from.label, ''),
    to:      coalesce(t.to.name,   t.to.label, '')
  }] AS incoming
"""

_IA_NS = "http://ia.gov.sa/ontology#"
_OWL_NS = "http://www.w3.org/2002/07/owl#"
_RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XSD_NS = "http://www.w3.org/2001/XMLSchema#"

_LOCAL_NAME_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _local_name(uri: str, fallback: str) -> str:
    """Extract the local part of a URI (after # or last /) for a QName."""
    if uri:
        for sep in ("#", "/"):
            if sep in uri:
                tail = uri.rsplit(sep, 1)[-1]
                if tail and _LOCAL_NAME_OK.match(tail):
                    return tail
    return fallback


def _ttl_str(s: str) -> str:
    return '"' + (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _render_turtle(payload: dict) -> str:
    cls = payload["cls"]
    props = payload["properties"] or []
    outgoing = payload["outgoing"] or []
    incoming = payload["incoming"] or []

    cls_name = _local_name(cls.get("uri", ""), cls.get("name") or "UnknownClass")
    lines: list[str] = [
        "# Per-class ontology export — RP IA Knowledge Graph",
        f"# Class: {cls_name}",
        f"@prefix ia:   <{_IA_NS}> .",
        f"@prefix owl:  <{_OWL_NS}> .",
        f"@prefix rdfs: <{_RDFS_NS}> .",
        f"@prefix xsd:  <{_XSD_NS}> .",
        "",
        "# ---- Class ----",
        f"ia:{cls_name} a owl:Class ;",
        f"    rdfs:label {_ttl_str(cls.get('label') or cls_name)} ;",
    ]
    if cls.get("comment"):
        lines.append(f"    rdfs:comment {_ttl_str(cls['comment'])} ;")
    # replace trailing ';' on last stmt with '.'
    lines[-1] = lines[-1].rstrip(" ;") + " ."
    lines.append("")

    # Referenced classes (so the file is self-contained) — declare each once.
    referenced: dict[str, str] = {}
    for rel in outgoing:
        n = _local_name("", rel.get("to") or "")
        if n and n != cls_name:
            referenced.setdefault(n, rel.get("to") or n)
    for rel in incoming:
        n = _local_name("", rel.get("from") or "")
        if n and n != cls_name:
            referenced.setdefault(n, rel.get("from") or n)
    if referenced:
        lines.append("# ---- Referenced classes ----")
        for local, label in referenced.items():
            lines += [
                f"ia:{local} a owl:Class ;",
                f"    rdfs:label {_ttl_str(label)} .",
                "",
            ]

    # Object properties
    all_rels = outgoing + incoming
    if all_rels:
        lines.append("# ---- Object properties ----")
        seen: set[str] = set()
        for rel in all_rels:
            rtype = _local_name(rel.get("uri", ""), rel.get("type") or "")
            if not rtype or rtype in seen:
                continue
            seen.add(rtype)
            frm = _local_name("", rel.get("from") or "")
            to = _local_name("", rel.get("to") or "")
            lines += [
                f"ia:{rtype} a owl:ObjectProperty ;",
                f"    rdfs:label {_ttl_str(rel.get('label') or rtype)} ;",
            ]
            if frm:
                lines.append(f"    rdfs:domain ia:{frm} ;")
            if to:
                lines.append(f"    rdfs:range ia:{to} ;")
            if rel.get("comment"):
                lines.append(f"    rdfs:comment {_ttl_str(rel['comment'])} ;")
            lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")

    # Datatype properties
    if props:
        lines.append("# ---- Datatype properties ----")
        for p in props:
            pname = _local_name(p.get("uri", ""), p.get("name") or "")
            if not pname:
                continue
            lines += [
                f"ia:{pname} a owl:DatatypeProperty ;",
                f"    rdfs:label {_ttl_str(p.get('label') or pname)} ;",
                f"    rdfs:domain ia:{cls_name} ;",
            ]
            if p.get("comment"):
                lines.append(f"    rdfs:comment {_ttl_str(p['comment'])} ;")
            lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_rdfxml(payload: dict) -> str:
    cls = payload["cls"]
    props = payload["properties"] or []
    outgoing = payload["outgoing"] or []
    incoming = payload["incoming"] or []

    cls_name = _local_name(cls.get("uri", ""), cls.get("name") or "UnknownClass")

    def about(local: str) -> str:
        return xml_quoteattr(_IA_NS + local)

    def res(local: str) -> str:
        return xml_quoteattr(_IA_NS + local)

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- Per-class ontology export — RP IA Knowledge Graph -->",
        f"<!-- Class: {xml_escape(cls_name)} -->",
        "<rdf:RDF",
        f'    xmlns:rdf="{_RDF_NS}"',
        f'    xmlns:rdfs="{_RDFS_NS}"',
        f'    xmlns:owl="{_OWL_NS}"',
        f'    xmlns:xsd="{_XSD_NS}"',
        f'    xmlns:ia="{_IA_NS}">',
        "",
        f"  <owl:Class rdf:about={about(cls_name)}>",
        f"    <rdfs:label>{xml_escape(cls.get('label') or cls_name)}</rdfs:label>",
    ]
    if cls.get("comment"):
        out.append(f"    <rdfs:comment>{xml_escape(cls['comment'])}</rdfs:comment>")
    out.append("  </owl:Class>")
    out.append("")

    referenced: dict[str, str] = {}
    for rel in outgoing:
        n = _local_name("", rel.get("to") or "")
        if n and n != cls_name:
            referenced.setdefault(n, rel.get("to") or n)
    for rel in incoming:
        n = _local_name("", rel.get("from") or "")
        if n and n != cls_name:
            referenced.setdefault(n, rel.get("from") or n)
    for local, label in referenced.items():
        out += [
            f"  <owl:Class rdf:about={about(local)}>",
            f"    <rdfs:label>{xml_escape(label)}</rdfs:label>",
            "  </owl:Class>",
            "",
        ]

    seen: set[str] = set()
    for rel in outgoing + incoming:
        rtype = _local_name(rel.get("uri", ""), rel.get("type") or "")
        if not rtype or rtype in seen:
            continue
        seen.add(rtype)
        frm = _local_name("", rel.get("from") or "")
        to = _local_name("", rel.get("to") or "")
        out.append(f"  <owl:ObjectProperty rdf:about={about(rtype)}>")
        out.append(f"    <rdfs:label>{xml_escape(rel.get('label') or rtype)}</rdfs:label>")
        if frm:
            out.append(f"    <rdfs:domain rdf:resource={res(frm)}/>")
        if to:
            out.append(f"    <rdfs:range rdf:resource={res(to)}/>")
        if rel.get("comment"):
            out.append(f"    <rdfs:comment>{xml_escape(rel['comment'])}</rdfs:comment>")
        out.append("  </owl:ObjectProperty>")
        out.append("")

    for p in props:
        pname = _local_name(p.get("uri", ""), p.get("name") or "")
        if not pname:
            continue
        out.append(f"  <owl:DatatypeProperty rdf:about={about(pname)}>")
        out.append(f"    <rdfs:label>{xml_escape(p.get('label') or pname)}</rdfs:label>")
        out.append(f"    <rdfs:domain rdf:resource={res(cls_name)}/>")
        if p.get("comment"):
            out.append(f"    <rdfs:comment>{xml_escape(p['comment'])}</rdfs:comment>")
        out.append("  </owl:DatatypeProperty>")
        out.append("")

    out.append("</rdf:RDF>")
    return "\n".join(out) + "\n"


_EXPORT_FORMATS = {
    "turtle": ("text/turtle; charset=utf-8", "ttl", _render_turtle),
    "ttl":    ("text/turtle; charset=utf-8", "ttl", _render_turtle),
    "rdfxml": ("application/rdf+xml; charset=utf-8", "rdf", _render_rdfxml),
    "rdf":    ("application/rdf+xml; charset=utf-8", "rdf", _render_rdfxml),
    "xml":    ("application/rdf+xml; charset=utf-8", "rdf", _render_rdfxml),
}


@router.get("/ontology/{class_name}/export")
def export_class_ontology(class_name: str, format: str = "turtle") -> Response:
    """
    Export the ontology subset for one class as an RDF file.

    Query params:
      format = turtle | rdfxml

    The response is a downloadable file (Content-Disposition: attachment)
    in the ia: vocabulary, matching the shape of ia_ontology.ttl so it can
    be reloaded through n10s.onto.import.fetch.
    """
    fmt = (format or "turtle").lower().strip()
    if fmt not in _EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format {format!r}. Use one of: turtle, rdfxml.",
        )

    name = (class_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="class_name is required")

    try:
        rows = Neo4jConnection.run_query(_ONT_EXPORT_CLASS, {"name": name})
    except Exception as e:
        logger.exception("Ontology export query failed")
        raise HTTPException(status_code=500, detail=f"Ontology query failed: {e}")

    if not rows or not rows[0].get("cls"):
        raise HTTPException(
            status_code=404,
            detail=f"No ontology class found with name: {name}",
        )

    payload = rows[0]
    mime, ext, render = _EXPORT_FORMATS[fmt]
    body = render(payload)

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "class"
    filename = f"ia_ontology_{safe}.{ext}"
    return Response(
        content=body,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
