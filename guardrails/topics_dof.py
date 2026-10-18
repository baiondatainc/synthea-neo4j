"""
guardrails/topics_dof.py
────────────────────────
Domain-specific input-guardrail vocabulary for the Future Advisor (DoF)
knowledge graph (macro indicators, Dubai fiscal, scenarios, BSP).

Sources of allowed topics:
  1. Node labels from the merged catalog (auto: lower-cased, camel-split, plural)
  2. Relationship types from the catalog (auto: lower-cased, "_" → " ")
  3. `allowed_topics` list from the catalog (en + ar, hand-curated)
  4. `label_keywords` values from the catalog (en + ar triggers per label)
  5. Country names from metadata/country_names.yaml ("United Arab Emirates", "Saudi Arabia", …)
  6. Curated DoF vocab here for terms that aren't in the catalog

The catalog is read lazily and cached. Call `reload()` after regenerating
the dictionary or editing overrides.

Public API:
  matches_allowed_topic(question) -> bool
  is_aggregate_bypass(question)   -> bool
  BLOCK_MESSAGE                    -> str
  reload()                          -> None
"""
from __future__ import annotations

import re
from functools import lru_cache

from metadata.catalog import get_catalog


# ── Curated DoF-domain vocabulary not already in the catalog ────────────────
# Macro / fiscal / scenario terms an analyst would use. The catalog already
# contains most of these via `allowed_topics` and `label_keywords`; this list
# is the safety net for phrasings that aren't.

_CURATED_TOPICS: list[str] = [
    # Macro indicators
    "gdp", "real gdp", "nominal gdp", "gdp per capita", "gdp growth",
    "inflation", "cpi", "core inflation", "deflation",
    "unemployment", "jobless", "employment", "population",
    "policy rate", "interest rate", "fed funds", "fed", "central bank",
    "bond yield", "10-year", "10 year", "treasury", "treasuries",
    "dollar", "dollar index", "dxy", "exchange rate", "fx", "euro dollar",
    "current account", "balance of payments", "fiscal balance",
    "government debt", "public debt", "debt to gdp", "deficit", "surplus",
    "exports", "imports", "trade balance", "fdi",
    # Oil & commodities
    "oil", "brent", "wti", "crude", "oil price", "oil forecast",
    # Dubai fiscal
    "revenue", "expenditure", "spending", "budget", "actual", "recurrent",
    "capital spending", "fees", "vat", "customs", "entity", "entities",
    "cofog", "function", "sector", "isic", "health spending", "education spending",
    "aed", "aed millions", "dubai budget", "budget book",
    # Scenarios / drivers / engine
    "scenario", "baseline", "downside", "upside", "driver", "lever",
    "elasticity", "sensitivity", "beta", "stress test", "what if",
    "measure", "policy measure", "fee change", "stimulus",
    "forecast", "projection", "horizon", "focus year",
    # Backtest / BSP
    "backtest", "mape", "accuracy", "variant", "vintage",
    "bsp", "budget strategy paper", "cycle", "stage", "milestone",
    "owner", "due", "overdue",
    # Sources / pipeline
    "source", "sources", "publisher", "imf", "world bank", "wb", "fred",
    "eia", "oecd", "bis", "ecb", "dsc", "statistics centre", "dof",
    "dataset", "datasets", "document", "documents", "publication",
    "ingest", "ingestion", "load", "fetched", "last run",
    "observation", "observations", "data point", "value",
    "revision", "revisions", "revised", "vintage",
    "confidence", "validated", "release", "calendar",
    # Analytics intents
    "trend", "monthly", "quarterly", "yearly", "annual",
    "by year", "by month", "by country", "by entity", "by sector",
    "by function", "by source", "top", "average", "total", "mean",
    "compare", "comparison", "versus", "vs",
    "how many", "show me", "list", "find", "breakdown",
    "latest", "current", "since", "range", "min", "max", "history",
    # Country aliases (full names come from country_names.yaml at load time)
    "uae", "emirates", "dubai", "saudi", "ksa", "us", "usa",
    "united states", "america", "fed", "uk", "britain",
    "china", "india", "germany", "japan", "euro area", "eurozone",
    "advanced economies", "emerging markets", "developing", "world",
    # Arabic
    "الناتج", "النمو", "التضخم", "البطالة", "النفط", "الفائدة",
    "الإيرادات", "النفقات", "الميزانية", "السيناريو", "التوقعات",
    "المصدر", "المؤشر", "الإمارات", "دبي", "السعودية", "أمريكا",
]


# ── Aggregate / summary intents (bypass the topic check entirely) ───────────

_AGGREGATE_BYPASS = re.compile(
    r'\b('
    # Summary verbs
    r'summarize|summary|overview|profile|describe|breakdown|report|'
    # DoF-flavoured aggregates
    r'executive summary|macro summary|fiscal summary|portfolio|'
    r'macro overview|fiscal overview|dashboard|'
    # "give me everything"
    r'all countries|all indicators|all kpis|all sources|all series|'
    r'all observations|all drivers|all measures|all scenarios|all stages|'
    r'all datasets|all documents|'
    # Meta
    r'high.?level|how are we doing|give me a summary|show me a summary'
    r')\b',
    re.IGNORECASE,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pluralize(word: str) -> str:
    if word.endswith("y") and (len(word) < 2 or word[-2] not in "aeiou"):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _camel_to_words(label: str) -> str:
    """KpiObservation -> 'kpi observation', BspStage -> 'bsp stage'."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', label).lower()


# ── Catalog-derived topics (cached) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _catalog_topics() -> frozenset[str]:
    cat = get_catalog()
    topics: set[str] = set()

    # Node labels: singular + plural + camel-split (both forms)
    for label in cat.labels.keys():
        single = label.lower()
        topics.add(single)
        topics.add(_pluralize(single))

        multi = _camel_to_words(label)
        if multi != single:
            topics.add(multi)
            head, _, tail = multi.rpartition(" ")
            topics.add(f"{head} {_pluralize(tail)}".strip())

    # Relationship types: "FOR_COUNTRY" -> "for country"
    for rel in cat.relationships.keys():
        topics.add(rel.lower().replace("_", " "))

    # allowed_topics (en + ar, from dictionary_overrides.yaml)
    topics.update(t.lower() for t in cat.allowed_topics)

    # label_keywords values (per-label en + ar trigger lists)
    for kws in (cat._doc.get("label_keywords") or {}).values():
        topics.update(kw.lower() for kw in kws)

    # Country full names from country_names.yaml → "united arab emirates", etc.
    for name in cat.country_names.values():
        topics.add(name.lower())

    return frozenset(topics)


@lru_cache(maxsize=1)
def _all_topics() -> tuple[str, ...]:
    combined = set(_catalog_topics()) | {t.lower() for t in _CURATED_TOPICS}
    # Longer topics first — more specific matches would win if we ever log them.
    return tuple(sorted(combined, key=len, reverse=True))


# ── Public API ───────────────────────────────────────────────────────────────

BLOCK_MESSAGE = (
    "Question does not appear to relate to the Future Advisor (DoF) "
    "knowledge graph. Try asking about macro indicators (GDP, inflation, "
    "policy rate, oil), Dubai fiscal data (revenue, expenditure, entities, "
    "COFOG), scenarios and drivers, backtests, BSP stages, data sources, "
    "or ingestion status."
)


def is_aggregate_bypass(question: str) -> bool:
    return bool(_AGGREGATE_BYPASS.search(question))


def matches_allowed_topic(question: str) -> bool:
    q = question.lower()
    return any(topic in q for topic in _all_topics())


def reload() -> None:
    """Clear caches; call after the dictionary or overrides change."""
    get_catalog.cache_clear()
    _catalog_topics.cache_clear()
    _all_topics.cache_clear()
