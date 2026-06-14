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

# SQL-shaped output from a confused LLM. Cypher never uses these as openers
# or in the canonical positions SQL puts them in.
SQL_FROM = re.compile(r"(?:^|\n)\s*FROM\s+[A-Za-z]", re.IGNORECASE)
SQL_SELECT = re.compile(r"(?:^|\n)\s*SELECT\s+", re.IGNORECASE)
SQL_JOIN = re.compile(r"\b(INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b", re.IGNORECASE)
SQL_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)


def _check_not_sql(cypher: str) -> str | None:
    if SQL_FROM.search(cypher) or SQL_SELECT.search(cypher):
        return "Cypher generator produced SQL syntax (FROM/SELECT). Try rephrasing the question."
    if SQL_JOIN.search(cypher):
        return "Cypher generator produced SQL JOIN. Cypher uses pattern matching, not JOIN."
    if SQL_GROUP_BY.search(cypher):
        return "Cypher generator used GROUP BY. Cypher groups implicitly via aggregation in RETURN."
    return None

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

    err = _check_not_sql(cypher)
    if err:
        return GuardrailResult(ok=False, payload=cypher, reason=err)

    err = _check_readonly(cypher)
    if err:
        return GuardrailResult(ok=False, payload=cypher, reason=err)

    err = _check_schema(cypher)
    if err:
        return GuardrailResult(ok=False, payload=cypher, reason=err)

    cypher = _inject_limit(cypher, settings.cypher_row_limit)
    return GuardrailResult(ok=True, payload=cypher)
