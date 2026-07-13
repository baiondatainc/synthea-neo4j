"""
Follow-up rewriter.

Given a transcript, a list of focus entity IDs, and the new user question,
return a *standalone* version of the question that an LLM can answer without
any of the prior context.

Decision flow:

  1. No transcript  → nothing to rewrite against (first turn). Return unchanged.
  2. Not a follow-up → question is self-contained. Return unchanged (no LLM call).
  3. Otherwise      → rewrite with the LLM, using the transcript as the primary
                      anchor and focus entity IDs as *optional* augmentation.

Why the transcript is the anchor (not focus IDs):
  Many follow-ups don't reference a returned entity at all — they tweak a filter
  from the previous QUESTION: "how about 2024?" (change the year), "in TN?"
  (change the state), "by modality?" (change the grouping). The antecedent lives
  in the transcript text, and the prior turn may have returned a bare aggregate
  (e.g. a single count) with zero extractable entity IDs. Gating the rewrite on
  focus IDs silently drops this entire class of follow-up. Focus IDs still help
  for entity drill-downs ("those patients", "that cohort"), so we pass them in
  when present — but their absence never blocks a rewrite.

We never call the LLM in the heuristic-skip cases so first turns and clearly
self-contained questions stay free.
"""
from __future__ import annotations

import logging
import re

from langchain_core.prompts import PromptTemplate

from qa.llm import get_llm

logger = logging.getLogger(__name__)

DEMONSTRATIVES = re.compile(
    r"\b(those|these|them|they|that|it|theirs?|same)\b",
    re.IGNORECASE,
)
FOLLOWUP_VERBS = re.compile(
    r"\b(narrow|drill|filter|exclude|include|instead|only|just|but|also|then|now)\b",
    re.IGNORECASE,
)
# Bare tweaks that carry an implicit "…of the previous question": a new year,
# state, month, or a "how about …" / "what about …" / "and …" opener.
FOLLOWUP_OPENERS = re.compile(
    r"^\s*(how about|what about|and|or|also|now|then)\b",
    re.IGNORECASE,
)

REWRITE_PROMPT = PromptTemplate(
    input_variables=["transcript", "focus", "question"],
    template="""You rewrite elliptical follow-up questions into standalone
questions for a Cypher-generating system over a healthcare knowledge graph.

RULES:
- Use the TRANSCRIPT to recover anything the follow-up leaves implicit —
  filters, entities, locations, time ranges, metrics — and carry it into the
  rewrite so the result stands on its own.
- Change ONLY what the follow-up explicitly changes. If it says "how about
  2024?", keep the previous question's subject and filters and swap just the
  year. If it says "in TN?", keep the previous metric and swap the state.
- When focus entity IDs are provided, use them to resolve pronouns and
  demonstratives ("those", "them", "that cohort"). Focus IDs may be "(none)",
  in which case rely entirely on the transcript.
- Keep the rewritten question short and concrete.
- If the new question is already standalone, return it UNCHANGED.
- Do not invent entities, filters, or values not present in the transcript or
  focus list.
- Output ONLY the rewritten question — no explanation, no quotes, no labels.

Transcript (most recent turns):
{transcript}

Focus entity IDs (the conversation is currently about these; may be "(none)"):
{focus}

New user question:
{question}

Rewritten standalone question:""",
)


def _is_likely_followup(question: str) -> bool:
    q = question.strip()
    # Very long questions are usually self-contained.
    if len(q) > 220:
        return False
    if DEMONSTRATIVES.search(q):
        return True
    if FOLLOWUP_VERBS.search(q):
        return True
    if FOLLOWUP_OPENERS.search(q):
        return True
    # Short imperative tweaks ("by state?", "in TN?", "how about 2024?").
    if len(q) < 35 and q.endswith("?"):
        return True
    return False


def rewrite_question(question: str, transcript: str, focus_ids: list[str]) -> str:
    """Return a standalone version of `question`.

    Falls back to the original on any error so a memory glitch never breaks the
    request. focus_ids are optional — an empty list does NOT prevent a rewrite;
    the transcript alone is sufficient to resolve most follow-ups.
    """
    # 1. First turn / nothing to anchor against.
    if not transcript:
        return question

    # 2. Self-contained question — no rewrite, no LLM call.
    if not _is_likely_followup(question):
        return question

    # 3. Follow-up with a transcript → rewrite. focus_ids augment but don't gate.
    if not focus_ids:
        logger.info("rewriter: follow-up with transcript, no focus IDs — rewriting from transcript")

    try:
        llm = get_llm(streaming=False)
        focus_str = ", ".join(focus_ids[:30]) if focus_ids else "(none)"
        prompt = REWRITE_PROMPT.format(
            transcript=transcript,
            focus=focus_str,
            question=question,
        )
        # Single short synchronous call.
        result = llm.invoke(prompt)
        rewritten = result.content if hasattr(result, "content") else str(result)
        rewritten = rewritten.strip().strip('"').strip()
        if not rewritten:
            logger.info("rewriter: empty rewrite — using original question")
            return question
        if rewritten == question:
            logger.info("rewriter: model returned question unchanged (already standalone)")
            return question
        logger.info(f"rewriter: {question!r} -> {rewritten!r}")
        return rewritten
    except Exception as e:
        logger.warning(f"rewriter failed ({e}); using original question")
        return question