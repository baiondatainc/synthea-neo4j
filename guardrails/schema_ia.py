"""
guardrails/schema_ia.py
────────────────────────
Cypher-guardrail schema vocabulary for the Insurance Authority (IA)
knowledge graph.

Loads the set of allowed node labels and relationship types from
data_catalog.yaml so that a schema change (nodes added / removed in Neo4j
and re-exported) doesn't require editing Python.

Public API:
  allowed_labels()         -> frozenset[str]
  allowed_relationships()  -> frozenset[str]
  reload()                 -> None
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data_catalog.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    if not CATALOG_PATH.exists():
        return {}
    with open(CATALOG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def allowed_labels() -> frozenset[str]:
    return frozenset((_load().get("nodes") or {}).keys())


@lru_cache(maxsize=1)
def allowed_relationships() -> frozenset[str]:
    return frozenset((_load().get("relationships") or {}).keys())


def reload() -> None:
    """Clear caches; call after data_catalog.yaml is regenerated."""
    _load.cache_clear()
    allowed_labels.cache_clear()
    allowed_relationships.cache_clear()
