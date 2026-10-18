"""
guardrails/schema_dof.py
────────────────────────
Cypher-guardrail schema vocabulary for the Future Advisor (DoF) knowledge
graph: labels like Kpi, Series, KpiObservation, Source, Country, DeMeasure,
DeScenarioDriver, BspCycle, …; relationships FOR_KPI, FOR_COUNTRY, FOR_SOURCE,
FOR_CODE, FOR_DRIVER, FOR_CYCLE.

Reads from the merged catalog (metadata/data_dictionary.yaml overlaid with
metadata/dictionary_overrides.yaml), so a schema change doesn't require
editing Python — regenerate the dictionary (or edit the overrides file)
and call reload().

Public API:
  allowed_labels()         -> frozenset[str]
  allowed_relationships()  -> frozenset[str]
  reload()                 -> None
"""
from __future__ import annotations

from functools import lru_cache

from metadata.catalog import get_catalog


@lru_cache(maxsize=1)
def allowed_labels() -> frozenset[str]:
    return frozenset(get_catalog().labels.keys())


@lru_cache(maxsize=1)
def allowed_relationships() -> frozenset[str]:
    return frozenset(get_catalog().relationships.keys())


def reload() -> None:
    """Clear caches; call after the dictionary or overrides change."""
    get_catalog.cache_clear()
    allowed_labels.cache_clear()
    allowed_relationships.cache_clear()
