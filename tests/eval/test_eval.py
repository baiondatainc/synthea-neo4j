"""Cypher-generation eval test suite.

Two modes in one file:

  STRUCTURAL EVAL  (existing)
  ───────────────
  Each question from docs/questions-eval.md becomes one parametrized test.
  Questions are independent — no conversation context needed.

  Pass criterion (asserted): the Cypher LLM produces non-empty Cypher.
  Additional metrics recorded but NOT gating:
    - guardrail_ok  — the production guardrail layer accepts the query
    - executable    — Neo4j EXPLAIN succeeds (requires live DB)

  CONTEXT CHAIN EVAL  (new)
  ──────────────────
  Thread questions from docs/context-chain-eval.md must run IN SEQUENCE.
  Each thread shares a stable conversation_id so the rewriter can carry
  forward focus entities across turns.

  Extra metrics recorded:
    - rewritten        — the rewriter fired and changed the question
    - rewritten_text   — the standalone question the rewriter produced
    - thread_id        — which thread this turn belongs to
    - turn_index       — position within the thread (0-based)

Run examples
────────────
# Structural eval only (existing behaviour)
uv run pytest tests/eval -v

# First 5 structural questions only
uv run pytest tests/eval --eval-limit 5 -v

# Only Hard structural questions
uv run pytest tests/eval --eval-difficulty H -v

# Context chain eval only
uv run pytest tests/eval -v -k "context_chain"

# Both evals together
uv run pytest tests/eval -v

# Skip Neo4j EXPLAIN check (faster, no DB needed)
uv run pytest tests/eval --eval-no-execute -v

# Run against live HTTP server (both evals)
uv run pytest tests/eval --eval-url http://localhost:8001 -v
"""
from __future__ import annotations

import time
import uuid

import pytest

from tests.eval.questions import Question, load_questions


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _guardrail_check(cypher: str) -> tuple[bool, str | None]:
    """Run the production Cypher guardrail. Returns (ok, reason)."""
    try:
        from guardrails import check_cypher  # noqa: PLC0415
        result = check_cypher(cypher)
        if isinstance(result, tuple):
            return result
        return bool(result), None
    except ImportError:
        return True, None
    except Exception as exc:
        return False, str(exc)


def _explain(neo4j_driver, cypher: str) -> tuple[bool, str | None]:
    """Run EXPLAIN on Neo4j. Returns (ok, error_message)."""
    if neo4j_driver is None:
        return False, None
    try:
        with neo4j_driver.session() as session:
            session.run(f"EXPLAIN {cypher}").consume()
        return True, None
    except Exception as exc:
        return False, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL EVAL — parametrized, independent questions
# ─────────────────────────────────────────────────────────────────────────────

def _load_parametrize(config) -> list[Question]:
    """Load structural questions, apply --eval-limit / --eval-difficulty."""
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
    """Expand test_question with one item per structural Question."""
    if "question" not in metafunc.fixturenames:
        return
    questions = _load_parametrize(metafunc.config)
    metafunc.parametrize(
        "question",
        questions,
        ids=[q.slug for q in questions],
    )


def test_question(
    question: Question,
    generate_cypher,
    neo4j_driver,
    metrics,
):
    """
    Structural eval: one independent question per test.

      1. Call the Cypher LLM — assert non-empty output.    (PASS/FAIL)
      2. Check guardrail acceptance.                        (metric only)
      3. Run EXPLAIN on Neo4j.                              (metric only)
    """
    rec: dict = {
        "number":           question.number,
        "question":         question.text,
        "difficulty":       question.difficulty,
        "category":         question.category,
        "cypher":           None,
        "cypher_generated": False,
        "guardrail_ok":     False,
        "guardrail_reason": None,
        "executable":       False,
        "execution_tested": False,
        "execution_error":  None,
        "latency_s":        None,
        "error":            None,
    }

    # ── Step 1: Cypher generation ─────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = generate_cypher(question.text)
        if isinstance(result, dict):
            cypher = result.get("cypher", "")
            rec["server_latency_s"] = result.get("latency_s")
        else:
            cypher = result
    except Exception as exc:
        rec["error"] = str(exc)
        metrics.record(rec)
        pytest.fail(
            f"Q{question.number} [{question.difficulty}] exception: {exc}",
            pytrace=False,
        )
        return

    rec["latency_s"]        = round(time.perf_counter() - t0, 3)
    rec["cypher"]           = cypher
    rec["cypher_generated"] = bool(cypher and cypher.strip())

    if not rec["cypher_generated"]:
        metrics.record(rec)
        pytest.fail(
            f"Q{question.number} [{question.difficulty}] empty output: {question.text!r}"
            + (f" — {rec['error']}" if rec.get("error") else ""),
            pytrace=False,
        )
        return

    # ── Step 2: Guardrail check ───────────────────────────────────────────
    rec["guardrail_ok"], rec["guardrail_reason"] = _guardrail_check(cypher)

    # ── Step 3: EXPLAIN ───────────────────────────────────────────────────
    if neo4j_driver is not None:
        rec["execution_tested"] = True
        rec["executable"], rec["execution_error"] = _explain(neo4j_driver, cypher)

    metrics.record(rec)


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT CHAIN EVAL — sequential threads, conversation memory required
# ─────────────────────────────────────────────────────────────────────────────

# Each thread is a list of (turn_text, difficulty, expected_rewrite_hint).
# expected_rewrite_hint is a short string that should appear in the rewritten
# question (case-insensitive substring check). Set to "" to skip that check.
CONTEXT_THREADS: list[dict] = [

    # ── Thread 1: Location & volume (all standalone) ──────────────────────
    {
        "id": "T01_location_volume",
        "label": "Location & volume — TX / Brooks City",
        "turns": [
            ("How many unique patients in the state of TX in May 2026?",                              "E", ""),
            ("How many unique patients for Complete Care Brooks City in May 2026?",                   "E", ""),
            ("How many unique scans in Complete Care Brooks City in May 2026?",                       "M", ""),
            ("How many unique invoices were sent for Complete Care Brooks City in May 2026?",         "M", ""),
            ("Show me all overdue invoices for Complete Care Brooks City in May 2026.",               "M", ""),
            ("How many patient visits for Complete Care Brooks City in May 2026?",                    "E", ""),
            ("How many no shows for Complete Care Brooks City in May 2026?",                          "M", ""),
        ],
    },

    # ── Thread 2: Invoice drill-down by patient name ──────────────────────
    {
        "id": "T02_invoice_by_name",
        "label": "Invoice drill-down by patient name",
        "turns": [
            ("Bring up all invoices for Sarah Mitchell for May 2026.",                                "E", ""),
            ("Bring up all invoices related to Chest X-Rays for Sarah Mitchell for 2026.",           "M", ""),
            ("Bring up invoices for Sarah Mitchell from Complete Care Brooks City in 2026.",          "M", ""),
            ("Show me all invoices for above but only for San Antonio location.",                     "M", "san antonio"),
            ("…. Only unpaid invoices.",                                                             "M", "unpaid"),
        ],
    },

    # ── Thread 3: Unpaid by name with location narrowing ─────────────────
    {
        "id": "T03_unpaid_by_name",
        "label": "Unpaid invoices — patient name chain",
        "turns": [
            ("Show me all unpaid invoices for Sarah Mitchell.",                                       "E", ""),
            ("…. from Complete Care Brooks City in 2026.",                                           "M", "complete care"),
            ("…. but only for San Antonio location.",                                                "M", "san antonio"),
        ],
    },

    # ── Thread 4: Unpaid by patient number with location narrowing ────────
    {
        "id": "T04_unpaid_by_number",
        "label": "Unpaid invoices — patient number chain",
        "turns": [
            ("Show me all unpaid invoices for patient RADM:1000042.",                                "E", ""),
            ("…. from Complete Care Brooks City in 2026.",                                           "M", "complete care"),
            ("…. but only for San Antonio location.",                                                "M", "san antonio"),
        ],
    },

    # ── Thread 5: Unpaid → full May picture (two-turn) ───────────────────
    {
        "id": "T05_unpaid_to_may",
        "label": "Unpaid → all May invoices + status",
        "turns": [
            ("Show me all unpaid invoices for patient RADM:1000042 from Complete Care Brooks City in 2026.", "M", ""),
            ("…. Show me all other May invoices, locations and status (paid, due, overdue).",        "H", "may"),
        ],
    },

    # ── Thread 6: Full patient profile chain (4 turns) ───────────────────
    {
        "id": "T06_full_profile",
        "label": "Unpaid → May picture → star rating → call count",
        "turns": [
            ("Show me all unpaid invoices for patient RADM:1000042 from Complete Care Brooks City.", "M", ""),
            ("…. Show me all other May invoices, locations and status (paid, due, overdue).",        "H", "may"),
            ("….. show me their star rating.",                                                       "H", "birdeye"),
            ("… how many times did they call.",                                                      "H", "call"),
        ],
    },

    # ── Thread 7: Latest invoices → star rating → call count ─────────────
    {
        "id": "T07_latest_to_calls",
        "label": "Latest invoices → star rating → call count",
        "turns": [
            ("Show me latest invoices for patient RADM:1000042 with locations and status (paid, due, overdue).", "M", ""),
            ("….. show me their star rating.",                                                       "H", "birdeye"),
            ("… how many times did they call.",                                                      "H", "call"),
        ],
    },

    # ── Thread 8: Responsible party chain ────────────────────────────────
    {
        "id": "T08_responsible_party",
        "label": "Responsible party — invoices → star rating → call count",
        "turns": [
            ("Show me latest invoices for Robert Chen with locations and status (paid, due, overdue).", "M", ""),
            ("….. show me their star rating.",                                                       "H", ""),
            ("… how many times did they call.",                                                      "H", ""),
        ],
    },

    # ── Thread 9: Bills sent today (standalone, 1 turn) ──────────────────
    {
        "id": "T09_bills_today",
        "label": "Bills sent today",
        "turns": [
            ("How many bills sent out today?",                                                        "E", ""),
        ],
    },

    # ── Thread 10: Calls today + state filter ────────────────────────────
    {
        "id": "T10_calls_today",
        "label": "Calls today → Tennessee filter",
        "turns": [
            ("How many calls today?",                                                                 "E", ""),
            ("….. in TN?",                                                                           "M", "tennessee"),
        ],
    },

    # ── Thread 11: Waiting time complaints + location ─────────────────────
    {
        "id": "T11_wait_complaints",
        "label": "Waiting time complaints → location breakdown",
        "turns": [
            ("How many customers have complained about waiting time yesterday?",                      "M", ""),
            ("… where are they located?",                                                            "M", "location"),
        ],
    },
]


def _run_thread(
    thread: dict,
    generate_cypher_with_conv,   # callable(question, conversation_id) -> str | dict
    neo4j_driver,
    metrics,
) -> list[dict]:
    """
    Run all turns in a thread sequentially, sharing a single conversation_id.
    Returns a list of per-turn metric records.
    """
    conversation_id = f"eval-{thread['id']}-{uuid.uuid4().hex[:8]}"
    records = []

    for turn_index, (turn_text, difficulty, rewrite_hint) in enumerate(thread["turns"]):
        rec: dict = {
            # identity
            "thread_id":        thread["id"],
            "thread_label":     thread["label"],
            "turn_index":       turn_index,
            "conversation_id":  conversation_id,
            # question
            "number":           f"{thread['id']}_t{turn_index}",
            "question":         turn_text,
            "difficulty":       difficulty,
            "category":         "Context Chain",
            # results
            "cypher":           None,
            "cypher_generated": False,
            "rewritten":        False,
            "rewritten_text":   None,
            "rewrite_hint_ok":  True,   # vacuously true when hint is ""
            "guardrail_ok":     False,
            "guardrail_reason": None,
            "executable":       False,
            "execution_tested": False,
            "execution_error":  None,
            "latency_s":        None,
            "error":            None,
        }

        t0 = time.perf_counter()
        try:
            result = generate_cypher_with_conv(turn_text, conversation_id)
            if isinstance(result, dict):
                cypher          = result.get("cypher", "")
                rewritten_text  = result.get("rewritten_question")
                rec["server_latency_s"] = result.get("latency_s")
            else:
                cypher         = result
                rewritten_text = None
        except Exception as exc:
            rec["error"]    = str(exc)
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            metrics.record(rec)
            records.append(rec)
            continue   # don't abort the thread — record and continue

        rec["latency_s"]        = round(time.perf_counter() - t0, 3)
        rec["cypher"]           = cypher
        rec["cypher_generated"] = bool(cypher and cypher.strip())

        if rewritten_text and rewritten_text != turn_text:
            rec["rewritten"]      = True
            rec["rewritten_text"] = rewritten_text

        if rewrite_hint and rec["rewritten"]:
            rec["rewrite_hint_ok"] = rewrite_hint.lower() in (rewritten_text or "").lower()

        if rec["cypher_generated"]:
            rec["guardrail_ok"], rec["guardrail_reason"] = _guardrail_check(cypher)
            if neo4j_driver is not None:
                rec["execution_tested"] = True
                rec["executable"], rec["execution_error"] = _explain(neo4j_driver, cypher)

        metrics.record(rec)
        records.append(rec)

    return records


@pytest.mark.context_chain
class TestContextChain:
    """
    One test method per thread. Each method runs all turns in sequence,
    sharing a conversation_id, and asserts that every turn produced Cypher.

    Mark: pytest -k context_chain  to run only these.
    Skip: pytest -k "not context_chain"  to skip.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, generate_cypher_with_conv, neo4j_driver, metrics):
        self._gen  = generate_cypher_with_conv
        self._db   = neo4j_driver
        self._metrics = metrics

    def _run(self, thread: dict) -> list[dict]:
        return _run_thread(thread, self._gen, self._db, self._metrics)

    def _assert_thread(self, records: list[dict], thread: dict) -> None:
        """Fail with a summary if any turn in the thread failed."""
        failed = [r for r in records if not r["cypher_generated"]]
        if failed:
            summary = "\n".join(
                f"  turn {r['turn_index']}: {r['question']!r}"
                + (f" — {r['error']}" if r.get("error") else " — empty output")
                for r in failed
            )
            pytest.fail(
                f"Thread {thread['id']} ({thread['label']}): "
                f"{len(failed)}/{len(records)} turns failed:\n{summary}",
                pytrace=False,
            )

    # ── One test per thread ───────────────────────────────────────────────

    def test_T01_location_volume(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T01_location_volume")
        self._assert_thread(self._run(t), t)

    def test_T02_invoice_by_name(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T02_invoice_by_name")
        self._assert_thread(self._run(t), t)

    def test_T03_unpaid_by_name(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T03_unpaid_by_name")
        self._assert_thread(self._run(t), t)

    def test_T04_unpaid_by_number(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T04_unpaid_by_number")
        self._assert_thread(self._run(t), t)

    def test_T05_unpaid_to_may(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T05_unpaid_to_may")
        self._assert_thread(self._run(t), t)

    def test_T06_full_profile(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T06_full_profile")
        self._assert_thread(self._run(t), t)

    def test_T07_latest_to_calls(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T07_latest_to_calls")
        self._assert_thread(self._run(t), t)

    def test_T08_responsible_party(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T08_responsible_party")
        self._assert_thread(self._run(t), t)

    def test_T09_bills_today(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T09_bills_today")
        self._assert_thread(self._run(t), t)

    def test_T10_calls_today(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T10_calls_today")
        self._assert_thread(self._run(t), t)

    def test_T11_wait_complaints(self):
        t = next(t for t in CONTEXT_THREADS if t["id"] == "T11_wait_complaints")
        self._assert_thread(self._run(t), t)