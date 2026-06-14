"""
Route a question to either the text2cypher path (exact / aggregate) or the
hybrid retriever path (similarity / behaviour).

Heuristic-first, keep it cheap. Ambiguous questions go to cypher because
it produces an auditable Cypher query the user can sanity-check.
"""
from __future__ import annotations

import re
from enum import Enum


class Path(str, Enum):
    CYPHER = "cypher"
    HYBRID = "hybrid"


# Strong signals for hybrid: similarity, neighborhoods, "patients like X"
HYBRID_PATTERNS = [
    r"\bsimilar to\b",
    r"\blike (this|that|those|these|the (patient|cohort))\b",
    r"\b(find|show) (me )?(patients|people|cases) like\b",
    r"\bcomparable to\b",
    r"\bclosest to\b",
    r"\bneighbou?rs of\b",
    r"\bsame (kind|sort|profile|cohort|behaviou?r) as\b",
    r"\bin the same cohort\b",
    r"\brecommend\b",
]

# Strong signals for cypher: counts, aggregates, time series
CYPHER_PATTERNS = [
    r"\bhow many\b",
    r"\bcount of\b",
    r"\btotal( amount| sum| count)?\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bmedian\b",
    r"\bmax(imum)?\b",
    r"\bmin(imum)?\b",
    r"\bsum of\b",
    r"\bgroup by\b",
    r"\bbreakdown by\b",
    r"\bby (year|month|state|practice|cohort|payor|carrier|modality|category)\b",
    r"\b(trend|over time|yearly|monthly|by year|by month)\b",
    r"\btop \d+\b",
    r"\bdistribution of\b",
    r"\bpercentage of\b",
]

_HYBRID = [re.compile(p, re.IGNORECASE) for p in HYBRID_PATTERNS]
_CYPHER = [re.compile(p, re.IGNORECASE) for p in CYPHER_PATTERNS]


def route(question: str) -> Path:
    q = (question or "").strip()
    if not q:
        return Path.CYPHER

    if any(p.search(q) for p in _HYBRID):
        return Path.HYBRID
    if any(p.search(q) for p in _CYPHER):
        return Path.CYPHER

    # Default: cypher path — predictable + auditable.
    return Path.CYPHER
