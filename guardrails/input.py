"""
guardrails/input.py
───────────────────
Input guardrail — checks an incoming question before it reaches the LLM.

Checks (in order):
  1. Aggregate/summary shortcut — always allowed (delegated to topics_ia)
  2. Jailbreak / injection patterns — always blocked (domain-agnostic)
  3. Topic allowlist — must match at least one IA topic (delegated to topics_ia)

Domain-specific vocabulary lives in guardrails/topics_ia.py so that a schema
change (add nodes/relationships to data_catalog.yaml) or a new insurance term
doesn't require editing this file.

Returns GuardrailResult(ok=True, payload=cleaned_question) on pass,
        GuardrailResult(ok=False, reason=...) on block.
"""
from __future__ import annotations

import re
import logging

from guardrails import GuardrailResult
from guardrails.topics_ia import (
    BLOCK_MESSAGE,
    is_aggregate_bypass,
    matches_allowed_topic,
)

logger = logging.getLogger(__name__)


# ── Jailbreak / injection patterns — always blocked, regardless of domain ───

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


# ── Main entry point ────────────────────────────────────────────────────────

def check_input(question: str) -> GuardrailResult[str]:
    """
    Validate incoming question.

    Returns GuardrailResult with:
      ok=True,  payload=cleaned question   → proceed
      ok=False, reason=explanation         → block and surface to user
    """
    if not question or not question.strip():
        return GuardrailResult(ok=False, payload="", reason="Empty question.")

    question = question.strip()

    # 1. Aggregate/summary shortcut — always allow.
    if is_aggregate_bypass(question):
        logger.info(f"input guardrail: aggregate/summary bypass → allowed: {question[:80]}")
        return GuardrailResult(ok=True, payload=question)

    # 2. Jailbreak check — always block.
    if _is_jailbreak(question):
        logger.warning(f"input guardrail: jailbreak pattern detected: {question[:120]}")
        return GuardrailResult(
            ok=False,
            payload=question,
            reason="This question appears to be a prompt injection attempt and has been blocked.",
        )

    # 3. Topic allowlist (IA insurance domain).
    if not matches_allowed_topic(question):
        logger.info(f"input guardrail: no topic match — blocked: {question[:80]}")
        return GuardrailResult(
            ok=False,
            payload=question,
            reason=BLOCK_MESSAGE,
        )

    logger.info(f"input guardrail: passed — {question[:80]}")
    return GuardrailResult(ok=True, payload=question)
