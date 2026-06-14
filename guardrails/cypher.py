"""
Cypher guardrail — three checks before the query reaches Neo4j:
  1. Read-only — reject mutating clauses (CREATE/DELETE/MERGE/SET/REMOVE/DROP, apoc writes).
  2. Schema — every label/relationship type referenced must exist in the catalog.
  3. Row cap — inject `LIMIT N` if absent.

Returns a `GuardrailResult[str]` whose `payload` is the (possibly mutated) Cypher.
The caller passes that payload to Neo4j with a session-level timeout.
"""
from __future__ import annotations

import re

from config import get_settings
from metadata.catalog import get_catalog
from guardrails import GuardrailResult

WRITE_CLAUSES = re.compile(
    r"\b(CREATE|DELETE|MERGE|SET|REMOVE|DROP|DETACH\s+DELETE)\b",
    re.IGNORECASE,
)
APOC_WRITE = re.compile(
    r"\b(apoc\.(create|merge|refactor|periodic|nodes\.delete|export|cypher\.runWrite))",
    re.IGNORECASE,
)
LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)

# A :Foo token inside (node:Foo) is a label; inside [edge:FOO] is a rel type.
# We use bracket context to disambiguate so an unknown rel doesn't get flagged
# as an unknown label and vice versa.
NODE_LABEL_RE = re.compile(r"\(\s*\w*\s*:([A-Z]\w*)")
REL_TYPE_RE = re.compile(r"\[\s*\w*\s*:([A-Z_][A-Z0-9_]*)")


def _check_readonly(cypher: str) -> str | None:
    if WRITE_CLAUSES.search(cypher):
        return "Cypher contains write clauses (CREATE/DELETE/MERGE/SET/REMOVE/DROP)."
    if APOC_WRITE.search(cypher):
        return "Cypher calls a mutating APOC procedure."
    return None


def _check_schema(cypher: str) -> str | None:
    catalog = get_catalog()
    allowed_labels = set(catalog.labels.keys())
    allowed_rels = set(catalog.relationships.keys())

    used_labels = set(NODE_LABEL_RE.findall(cypher))
    used_rels = set(REL_TYPE_RE.findall(cypher))

    unknown_labels = used_labels - allowed_labels
    unknown_rels = used_rels - allowed_rels

    if unknown_labels:
        return f"Cypher references unknown label(s): {sorted(unknown_labels)}"
    if unknown_rels:
        return f"Cypher references unknown relationship type(s): {sorted(unknown_rels)}"
    return None


def _inject_limit(cypher: str, row_cap: int) -> str:
    if LIMIT_RE.search(cypher):
        return cypher
    return cypher.rstrip().rstrip(";") + f"\nLIMIT {row_cap}"


def check_cypher(cypher: str) -> GuardrailResult[str]:
    settings = get_settings()
    if not cypher or not cypher.strip():
        return GuardrailResult(ok=False, payload="", reason="Empty Cypher.")

    err = _check_readonly(cypher)
    if err:
        return GuardrailResult(ok=False, payload=cypher, reason=err)

    err = _check_schema(cypher)
    if err:
        return GuardrailResult(ok=False, payload=cypher, reason=err)

    cypher = _inject_limit(cypher, settings.cypher_row_limit)
    return GuardrailResult(ok=True, payload=cypher)
