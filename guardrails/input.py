"""
guardrails/input.py
───────────────────
Input guardrail — checks an incoming question before it reaches the LLM.

Checks (in order):
  1. Aggregate/summary shortcut — always allowed, bypass topic check entirely
  2. Jailbreak / injection patterns — always blocked
  3. Topic allowlist — must match at least one allowed_topic from catalog

Returns GuardrailResult(ok=True, payload=cleaned_question) on pass,
        GuardrailResult(ok=False, reason=...) on block.
"""
from __future__ import annotations

import re
import logging

from metadata.catalog import get_catalog
from guardrails import GuardrailResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate / summary shortcut — always allowed, skip topic check
# These phrases are clearly RP-domain intents and must never be blocked.
# ─────────────────────────────────────────────────────────────────────────────

_AGGREGATE_BYPASS = re.compile(
    r'\b('
    # Summary verbs
    r'summarize|summary|overview|profile|describe|breakdown|report|'
    # Executive / financial
    r'executive summary|financial summary|financial overview|'
    r'complete financial|financial report|revenue summary|'
    r'collections overview|how are we doing|'
    # Portfolio / aggregate
    r'portfolio|aggregate|all patients|all locations|all campaigns|'
    r'all practices|all charges|all transactions|all reviews|'
    r'performance summary|high.?level|give me a summary|show me a summary'
    r')\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Jailbreak / injection patterns — always blocked regardless of topic
# ─────────────────────────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS = [
    re.compile(r'\bignore\s+(previous|all|prior|above)\b', re.IGNORECASE),
    re.compile(r'\bact\s+as\b', re.IGNORECASE),
    re.compile(r'\bpretend\s+(you\s+are|to\s+be)\b', re.IGNORECASE),
    re.compile(r'\bsystem\s+prompt\b', re.IGNORECASE),
    re.compile(r'\bDAN\b'),
    re.compile(r'\bjailbreak\b', re.IGNORECASE),
    re.compile(r'\byou\s+are\s+now\b', re.IGNORECASE),
    re.compile(r'\bforget\s+(your|all|previous)\b', re.IGNORECASE),
    re.compile(r'\bdo\s+anything\s+now\b', re.IGNORECASE),
    re.compile(r'\bdisregard\b', re.IGNORECASE),
]


def _is_jailbreak(question: str) -> bool:
    return any(p.search(question) for p in _JAILBREAK_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Topic allowlist check
# ─────────────────────────────────────────────────────────────────────────────

def _matches_allowed_topic(question: str) -> bool:
    """
    Returns True if the question contains at least one allowed topic substring.
    Uses case-insensitive substring matching (not word-boundary).
    Multi-word topics like "bad debt" match anywhere in the question.
    """
    catalog = get_catalog()
    allowed = catalog.allowed_topics  # list[str] loaded from YAML

    q_lower = question.lower()
    for topic in allowed:
        if topic.lower() in q_lower:
            logger.debug(f"input guardrail: topic match '{topic}'")
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def check_input(question: str) -> GuardrailResult[str]:
    """
    Validate incoming question.

    Returns GuardrailResult with:
      ok=True,  payload=cleaned question   → proceed
      ok=False, reason=explanation         → block and surface to user
    """
    if not question or not question.strip():
        return GuardrailResult(ok=False, payload="", reason="Empty question.")

    # ── Clean up whitespace ───────────────────────────────────────────────
    question = question.strip()

    # ── 1. Aggregate/summary shortcut — always allow ──────────────────────
    # These are clearly RP-domain intents. Bypass topic check entirely.
    if _AGGREGATE_BYPASS.search(question):
        logger.info(f"input guardrail: aggregate/summary bypass → allowed: {question[:80]}")
        return GuardrailResult(ok=True, payload=question)

    # ── 2. Jailbreak check — always block ────────────────────────────────
    if _is_jailbreak(question):
        logger.warning(f"input guardrail: jailbreak pattern detected: {question[:120]}")
        return GuardrailResult(
            ok=False,
            payload=question,
            reason="This question appears to be a prompt injection attempt and has been blocked.",
        )

    # ── 3. Topic allowlist ────────────────────────────────────────────────
    if not _matches_allowed_topic(question):
        logger.info(f"input guardrail: no topic match — blocked: {question[:80]}")
        return GuardrailResult(
            ok=False,
            payload=question,
            reason=(
                "Question does not appear to relate to the RP knowledge graph. "
                "Try asking about patients, practices, charges, balances, claims, "
                "financial summaries, campaigns, or Birdeye reviews."
            ),
        )

    logger.info(f"input guardrail: passed — {question[:80]}")
    return GuardrailResult(ok=True, payload=question)