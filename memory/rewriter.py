"""
Follow-up rewriter.

Given a transcript, a list of focus entity IDs, and the new user question,
return a *standalone* version of the question that an LLM can answer without
any of the prior context. Three-stage decision:

  1. Heuristic skip — if the question is long and self-contained
     (mentions a concrete noun from the schema), return unchanged.
  2. Heuristic detect — if the question contains demonstratives ("those",
     "that", "them", "they") or comparatives ("narrow", "drill", "filter",
     "instead"), it's almost certainly elliptical → rewrite.
  3. Skip when no context — empty transcript or no focus IDs means there's
     nothing to rewrite *into*; return unchanged.

The actual rewrite uses the QA LLM with a tight prompt. We never call the
LLM in the heuristic-skip case so first turns and clear questions stay free.
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

REWRITE_PROMPT = PromptTemplate(
    input_variables=["transcript", "focus", "question"],
    template="""You rewrite elliptical follow-up questions into standalone
questions for a Cypher-generating system.

RULES:
- Resolve pronouns and demonstratives ("those", "them", "that cohort") using
  the focus entity IDs listed below.
- Keep the rewritten question short and concrete.
- If the new question is already standalone, return it UNCHANGED.
- Do not invent entities that aren't in the focus list or the transcript.
- Output ONLY the rewritten question — no explanation, no quotes, no labels.

Transcript (most recent turns):
{transcript}

Focus entity IDs (the conversation is currently about these):
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
    # Short imperative tweaks ("by state?", "in TN?") are also follow-ups.
    if len(q) < 35 and q.endswith("?"):
        return True
    return False


def rewrite_question(question: str, transcript: str, focus_ids: list[str]) -> str:
    """Returns a standalone version of `question`. Falls back to the original
    on any error so a memory glitch never breaks the request."""
    if not transcript:
        return question
    if not focus_ids:
        # Nothing concrete to anchor a rewrite to.
        if _is_likely_followup(question):
            logger.info("rewriter: looks like a follow-up but no focus IDs — passing through")
        return question
    if not _is_likely_followup(question):
        return question

    try:
        llm = get_llm(streaming=False)
        focus_str = ", ".join(focus_ids[:30])  # cap to keep prompt small
        prompt = REWRITE_PROMPT.format(
            transcript=transcript,
            focus=focus_str,
            question=question,
        )
        # Use the synchronous invoke; this is a single short call.
        result = llm.invoke(prompt)
        rewritten = result.content if hasattr(result, "content") else str(result)
        rewritten = rewritten.strip().strip('"').strip()
        if not rewritten:
            return question
        logger.info(f"rewriter: {question!r} -> {rewritten!r}")
        return rewritten
    except Exception as e:
        logger.warning(f"rewriter failed ({e}); using original question")
        return question
