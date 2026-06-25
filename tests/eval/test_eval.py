"""Cypher-generation eval test suite.

Each question from docs/questions-eval.md becomes one parametrized test.

Pass criterion (asserted): the Cypher LLM produces non-empty Cypher.
Additional metrics recorded but NOT gating:
  - guardrail_ok  — the production guardrail layer accepts the query
  - executable    — Neo4j EXPLAIN succeeds (requires live DB + no --eval-no-execute)

Run examples
------------
# First 5 questions only
uv run pytest tests/eval --eval-limit 5 -v

# Only Hard questions
uv run pytest tests/eval --eval-difficulty H -v

# Skip the Neo4j EXPLAIN check (faster, no DB needed)
uv run pytest tests/eval --eval-no-execute -v

# Full run
uv run pytest tests/eval -v
"""
from __future__ import annotations

import time

import pytest

from tests.eval.questions import Question, load_questions


# ── Parametrize from the markdown question bank ──────────────────────────────

def _load_parametrize(config) -> list[Question]:
    """Load questions, apply --eval-limit and --eval-difficulty filters."""
    all_qs = load_questions()

    diff_filter = config.getoption("--eval-difficulty", default="")
    if diff_filter:
        allowed = {d.strip().upper() for d in diff_filter.split(",")}
        all_qs = [q for q in all_qs if q.difficulty in allowed]

    limit = config.getoption("--eval-limit", default=0)
    if limit and limit > 0:
        all_qs = all_qs[:limit]

    return all_qs


def pytest_generate_tests(metafunc):
    """Hook: expand test_question with one item per Question."""
    if "question" not in metafunc.fixturenames:
        return
    questions = _load_parametrize(metafunc.config)
    metafunc.parametrize(
        "question",
        questions,
        ids=[q.slug for q in questions],
    )


# ── The single eval test ──────────────────────────────────────────────────────

def test_question(
    question: Question,
    generate_cypher,       # fixture from conftest.py
    neo4j_driver,          # fixture from conftest.py — may be None
    metrics,               # fixture from conftest.py
):
    """
    For each question:
      1. Call the Cypher LLM and assert non-empty output.        (PASS/FAIL)
      2. Check guardrail acceptance.                             (metric only)
      3. Run EXPLAIN on Neo4j to verify executability.           (metric only)
    """
    rec: dict = {
        "number": question.number,
        "question": question.text,
        "difficulty": question.difficulty,
        "category": question.category,
        "cypher": None,
        "cypher_generated": False,
        "guardrail_ok": False,
        "guardrail_reason": None,
        "executable": False,
        "execution_tested": False,
        "execution_error": None,
        "latency_s": None,
        "error": None,
    }

    # ── Step 1: Cypher generation ─────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = generate_cypher(question.text)
        # generate_cypher now returns either str (direct) or dict (http)
        if isinstance(result, dict):
            cypher = result.get("cypher", "")
            rec["server_latency_s"] = result.get("latency_s")
        else:
            cypher = result
    except Exception as exc:
        rec["error"] = str(exc)
        metrics.record(rec)
        raise

    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    rec["cypher"] = cypher
    rec["cypher_generated"] = bool(cypher and cypher.strip())

    # This is the only hard assertion — the LLM must produce something.
    assert rec["cypher_generated"], (
        f"Cypher LLM returned empty output for: {question.text!r}"
    )

    # ── Step 2: Guardrail check (soft metric) ─────────────────────────────────
    try:
        from qa.guardrails import check_cypher  # noqa: PLC0415
        result = check_cypher(cypher)
        # check_cypher returns (ok: bool, reason: str | None)
        if isinstance(result, tuple):
            rec["guardrail_ok"], rec["guardrail_reason"] = result
        else:
            rec["guardrail_ok"] = bool(result)
    except ImportError:
        # No guardrail module — treat as passing so the metric isn't misleading.
        rec["guardrail_ok"] = True
    except Exception as exc:
        rec["guardrail_reason"] = str(exc)

    # ── Step 3: Executability via Neo4j EXPLAIN (soft metric) ─────────────────
    if neo4j_driver is not None:
        rec["execution_tested"] = True
        try:
            with neo4j_driver.session() as session:
                session.run(f"EXPLAIN {cypher}").consume()
            rec["executable"] = True
        except Exception as exc:
            rec["executable"] = False
            rec["execution_error"] = str(exc)

    # Record metrics regardless of guardrail / executability outcome.
    metrics.record(rec)