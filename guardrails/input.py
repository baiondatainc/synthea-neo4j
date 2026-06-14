"""
Input guardrail — rejects off-topic and prompt-injection attempts before they
reach the LLM.

Heuristic, intentionally cheap:
  1. Length cap — block walls of text (likely injection payloads).
  2. Topic keyword match — must mention at least one allowed-topic keyword.
  3. Injection phrase blacklist — common jailbreak patterns.

A more accurate classifier can replace this later; the protocol stays the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from metadata.catalog import get_catalog
from guardrails import GuardrailResult

MAX_QUESTION_CHARS = 1500

INJECTION_PATTERNS = [
    r"ignore (the )?(previous|above|prior) (instructions|prompts|rules)",
    r"disregard (the )?(previous|above) (instructions|prompts)",
    r"you are (now|actually) (a|an) ",
    r"system prompt",
    r"reveal (your )?(system )?prompt",
    r"jailbreak",
    r"DAN mode",
    r"developer mode",
    r"as an? (unfiltered|unrestricted|uncensored)",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def check_input(question: str) -> GuardrailResult[str]:
    q = (question or "").strip()
    if not q:
        return GuardrailResult(ok=False, payload="", reason="Empty question.")

    if len(q) > MAX_QUESTION_CHARS:
        return GuardrailResult(
            ok=False,
            payload=q,
            reason=f"Question exceeds {MAX_QUESTION_CHARS} characters.",
        )

    for pat in _compiled:
        if pat.search(q):
            return GuardrailResult(
                ok=False,
                payload=q,
                reason="Question matches a prompt-injection pattern and was blocked.",
            )

    catalog = get_catalog()
    q_lower = q.lower()
    if not any(kw in q_lower for kw in catalog.allowed_topics):
        return GuardrailResult(
            ok=False,
            payload=q,
            reason=(
                "Question does not appear to relate to the RP knowledge graph. "
                "Try asking about patients, practices, charges, balances, or claims."
            ),
        )

    return GuardrailResult(ok=True, payload=q)
