"""
guardrails/topics_ia.py
────────────────────────
Domain-specific input-guardrail vocabulary for the Insurance Authority (IA)
knowledge graph.

The IA graph has nodes like Insurer, Policy, Claim, Reinsurer, Vehicle,
Policyholder, HealthMember, Agent, PreAuthorization, DistributionChannel, etc.
(see data_catalog.yaml). Analysts also talk in domain terms that never appear
as node/property names — "loss ratio", "GWP", "combined ratio", "underwriting".
This module owns both.

Sources of allowed topics:
  1. Node labels from data_catalog.yaml  (auto: lower-cased + camel-split + plural)
  2. Relationship types from data_catalog.yaml (auto: lower-cased, "_" → " ")
  3. Curated insurance vocab that isn't in the schema (loss ratio, gwp, ...)
  4. Aggregate / summary intent phrases (portfolio, executive summary, ...)

The catalog is read lazily and cached; call `reload()` after regenerating
data_catalog.yaml to pick up new nodes/relationships without a process restart.

Public API:
  matches_allowed_topic(question) -> bool
  is_aggregate_bypass(question)   -> bool
  BLOCK_MESSAGE                    -> str
  reload()                          -> None
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data_catalog.yaml"


# ── Curated insurance-domain vocabulary ──────────────────────────────────────
# Terms an analyst would use that aren't node/property names in the schema.

_CURATED_TOPICS: list[str] = [
    # Financial / KPI
    "loss ratio", "reported loss ratio", "calculated loss ratio",
    "combined ratio", "expense ratio", "retention ratio", "claims ratio",
    "gwp", "gross written premium", "gross premium",
    "nwp", "net written premium", "net premium",
    "premium", "premiums", "written premium", "earned premium",
    "commission", "commissions", "underwriting", "underwriter",
    "portfolio", "book of business", "renewal", "renewals",
    "cancellation", "cancelled", "lapse", "in-force", "inforce",
    # Claim-side
    "settled", "settlement", "settlement days",
    "outstanding", "paid", "rejected", "approved", "adjudication",
    "fraud", "fraud probability", "fraud suspicion", "fraud flag",
    "pre-auth", "preauth", "pre authorization", "pre-authorization",
    "deductible", "copay", "co-pay", "patient share", "payer share",
    # Motor
    "damage", "damage assessment", "chassis", "license plate", "plate",
    "ev", "electric vehicle", "coverage type",
    # Health
    "member", "members", "beneficiary", "diagnosis", "diagnosis code",
    "icd", "service line", "health claim", "benefit class",
    # Insurer / regulator
    "reinsurance", "insurance authority", "ia product",
    "broker", "agency", "tpa", "channel",
    "line of business", "lob", "product", "products",
    # Geo / segmentation
    "region", "city", "country", "local", "saudi", "saudization",
    # Analytics intents
    "trend", "monthly", "quarterly", "yearly", "annual",
    "by year", "by month", "by insurer", "by policy", "by claim",
    "by product", "by channel", "top", "average", "total", "compare",
    "how many", "show me", "list", "find", "breakdown",
]


# ── Aggregate / summary intents (bypass topic check entirely) ────────────────

_AGGREGATE_BYPASS = re.compile(
    r'\b('
    # Summary verbs
    r'summarize|summary|overview|profile|describe|breakdown|report|'
    # Executive / financial
    r'executive summary|financial summary|financial overview|'
    r'complete financial|financial report|revenue summary|'
    # Portfolio / aggregate (very IA)
    r'portfolio|book of business|aggregate|'
    # "give me everything" — IA-flavoured
    r'all policies|all claims|all insurers|all reinsurers|'
    r'all policyholders|all vehicles|all products|all channels|'
    r'all agents|all accidents|'
    # Meta
    r'performance summary|how are we doing|high.?level|'
    r'give me a summary|show me a summary'
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
    """PolicyEvent -> 'policy event', HealthClaimLine -> 'health claim line'."""
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', label).lower()


# ── Catalog-derived topics (cached) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _catalog_topics() -> frozenset[str]:
    if not CATALOG_PATH.exists():
        return frozenset()
    with open(CATALOG_PATH, "r") as f:
        doc = yaml.safe_load(f) or {}

    topics: set[str] = set()

    # Node labels: singular + plural + camel-split (both forms)
    for label in (doc.get("nodes") or {}).keys():
        single_word = label.lower()
        topics.add(single_word)
        topics.add(_pluralize(single_word))

        multi = _camel_to_words(label)
        if multi != single_word:
            topics.add(multi)
            # Pluralize just the last token: "policy event" -> "policy events"
            head, _, tail = multi.rpartition(" ")
            topics.add(f"{head} {_pluralize(tail)}".strip())

    # Relationship types: "HAS_PRE_AUTH" -> "has pre auth"
    for rel in (doc.get("relationships") or {}).keys():
        topics.add(rel.lower().replace("_", " "))

    return frozenset(topics)


@lru_cache(maxsize=1)
def _all_topics() -> tuple[str, ...]:
    combined = set(_catalog_topics()) | {t.lower() for t in _CURATED_TOPICS}
    # Longer topics first so more specific matches log preferentially in future.
    return tuple(sorted(combined, key=len, reverse=True))


# ── Public API ───────────────────────────────────────────────────────────────

BLOCK_MESSAGE = (
    "Question does not appear to relate to the Insurance Authority (IA) "
    "knowledge graph. Try asking about policies, claims, insurers, reinsurers, "
    "policyholders, premiums, loss ratios, agents, distribution channels, "
    "vehicles, accidents, or pre-authorizations."
)


def is_aggregate_bypass(question: str) -> bool:
    return bool(_AGGREGATE_BYPASS.search(question))


def matches_allowed_topic(question: str) -> bool:
    q = question.lower()
    return any(topic in q for topic in _all_topics())


def reload() -> None:
    """Clear caches; call after data_catalog.yaml is regenerated."""
    _catalog_topics.cache_clear()
    _all_topics.cache_clear()
