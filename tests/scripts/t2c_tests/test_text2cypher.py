"""Regression suite: one test per question in questions.yaml.

    pytest --no-api -v -c pytest.ini --rootdir=.          # validate ground truth only
    pytest -v -s -c pytest.ini --rootdir=. --ids Q01,Q05   # a few, with console output
    pytest -v -c pytest.ini --rootdir=.                    # all 64

Verdicts per question:
    cypher    - rows from the generated Cypher match ground truth   -> PASS/FAIL
    narration - description is faithful to those rows               -> PASS/xfail (reported, never masks cypher)
"""
import time

import pytest

from harness import Question, ask_chat, check_narration, compare


def _print_rows(title, rows, n=5):
    print(f"--- {title} ({len(rows)} rows) ---")
    for row in rows[:n]:
        print("  ", row)


def test_question(q: Question, graph, results, no_api):
    # 1. ground truth
    t0 = time.perf_counter()
    try:
        want = graph.run(q.expected_cypher)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{q.id}: ground-truth Cypher failed ({type(e).__name__}) - "
                    f"fix questions.yaml if this is a Cypher error\n{q.expected_cypher}\n{e}")
    gt_latency = time.perf_counter() - t0

    if no_api:
        print(f"\n[{q.id}] {q.question}\n--- expected cypher ---\n{q.expected_cypher}")
        _print_rows("first rows", want)
        results.add(dict(id=q.id, category=q.category, question=q.question, passed=True,
                         detail=f"ground truth ok ({len(want)} rows)", latency_s=gt_latency))
        return

    # 2. ask the models
    try:
        res = ask_chat(q.question, graph)
    except Exception as e:  # noqa: BLE001
        results.add(dict(id=q.id, category=q.category, question=q.question, passed=False,
                         detail=f"backend error: {e}", latency_s=0))
        pytest.fail(f"{q.id}: backend error: {e}")

    # 3. rows from the generated Cypher
    got = res.rows
    exec_error = res.raw.get("exec_error") if isinstance(res.raw, dict) else None
    if got is None and res.cypher and not exec_error:          # api mode returned only cypher
        try:
            got = graph.run(res.cypher)
        except Exception as e:  # noqa: BLE001
            exec_error = str(e)
    got = got or []

    print(f"\n[{q.id}] {q.question}  ({res.latency_s:.1f}s)")
    print(f"--- generated cypher ---\n{res.cypher}")
    if exec_error:
        print(f"--- execution error ---\n{exec_error}")
    _print_rows("rows from generated cypher", got)
    _print_rows("rows from ground truth", want)
    if res.answer:
        print(f"--- answer ---\n{res.answer[:400]}")

    # 4. verdicts
    if exec_error:
        passed, detail = False, f"generated Cypher failed: {exec_error[:200]}"
    else:
        passed, detail = compare(q.compare, got, want, q)
    narr_ok, narr_detail = (None, None) if res.answer is None else check_narration(res.answer, got, q)

    results.add(dict(
        id=q.id, category=q.category, question=q.question,
        passed=passed, detail=detail,
        narration_passed=narr_ok, narration_detail=narr_detail,
        latency_s=res.latency_s, generated_cypher=res.cypher, answer=res.answer,
        got_rows=got[:20], want_rows=want[:20], stats=res.raw, note=q.note,
    ))

    assert passed, (
        f"{q.id} [{q.category}] {q.question}\n"
        f"  compare={q.compare}: {detail}\n"
        f"  generated cypher:\n{res.cypher}\n"
        f"  expected cypher:\n{q.expected_cypher}\n"
        f"  answer: {res.answer}"
    )
    if narr_ok is False:
        pytest.xfail(f"narration: {narr_detail}")
