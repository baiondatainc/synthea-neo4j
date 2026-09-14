"""Run every question in questions.yaml through the text2cypher model and record the response.

    python run_t2c.py                     # all 64
    python run_t2c.py --ids Q01,Q05       # subset
    python run_t2c.py --no-exec           # only generate, don't run the Cypher
    python run_t2c.py --compare           # also compare rows with ground truth

Output: results/t2c-<model>-<stamp>.json   (one record per question) + a console summary.
Uses the same .env / harness.py as the pytest suite (T2C_MODEL, OLLAMA_URL, NEO4J_*).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import CFG, RESULTS_DIR, Graph, _clean_cypher, _ollama_generate, compare, load_questions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--category")
    ap.add_argument("--no-exec", action="store_true", help="generate only, do not execute on Neo4j")
    ap.add_argument("--compare", action="store_true", help="also compare rows with ground-truth Cypher")
    ap.add_argument("--model", default=CFG.t2c_model)
    args = ap.parse_args()

    qs = load_questions()
    if args.category:
        qs = [q for q in qs if q.category == args.category]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        qs = [q for q in qs if q.id in wanted]

    graph = None
    if not args.no_exec:
        graph = Graph(CFG)
        graph.verify()

    # warm-up: loads model + prefills system prompt once
    print(f"model: {args.model} @ {CFG.ollama_url}   questions: {len(qs)}")
    _ollama_generate(args.model, "How many policies?", CFG)

    records = []
    for q in qs:
        rec = {"id": q.id, "category": q.category, "question": q.question,
               "generated": False, "executed": None, "rows": None, "cypher": None,
               "raw_response": None, "error": None, "gen_s": None, "exec_s": None}
        t0 = time.perf_counter()
        try:
            gen = _ollama_generate(args.model, q.question, CFG)
            rec["gen_s"] = round(time.perf_counter() - t0, 2)
            rec["raw_response"] = gen.get("response")
            rec["cypher"] = _clean_cypher(gen.get("response", ""))
            rec["generated"] = bool(rec["cypher"])
            rec["tokens"] = {k: gen.get(k) for k in ("prompt_eval_count", "eval_count")}
            rec["eval_tok_s"] = round(gen["eval_count"] * 1e9 / gen["eval_duration"], 1) \
                if gen.get("eval_duration") else None
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"generation: {e}"

        if rec["generated"] and graph is not None:
            t1 = time.perf_counter()
            try:
                rows = graph.run(rec["cypher"])
                rec["executed"] = True
                rec["rows"] = len(rows)
                rec["sample"] = rows[:5]
                if args.compare:
                    want = graph.run(q.expected_cypher)
                    ok, detail = compare(q.compare, rows, want, q)
                    rec["correct"], rec["compare_detail"] = ok, detail
            except Exception as e:  # noqa: BLE001
                rec["executed"] = False
                rec["error"] = f"execution: {str(e).splitlines()[0][:200]}"
            rec["exec_s"] = round(time.perf_counter() - t1, 2)

        flag = ("GEN-FAIL" if not rec["generated"] else
                "ok" if rec["executed"] is None else
                "EXEC-FAIL" if rec["executed"] is False else
                ("ok" if not args.compare else ("CORRECT" if rec.get("correct") else "WRONG")))
        print(f"{q.id:4} {flag:9} {rec['gen_s'] or 0:5.1f}s  rows={rec['rows']!s:>5}  {rec['cypher'] or rec['error']}"
              .replace("\n", " ")[:170])
        records.append(rec)

    if graph:
        graph.close()

    # ---- summary + file
    n = len(records)
    gen_ok = sum(r["generated"] for r in records)
    exec_ok = sum(1 for r in records if r["executed"])
    summary = {
        "model": args.model, "run": dt.datetime.now().isoformat(timespec="seconds"),
        "questions": n, "generated": gen_ok,
        "executed": exec_ok if not args.no_exec else None,
        "correct": sum(1 for r in records if r.get("correct")) if args.compare else None,
        "avg_gen_s": round(sum(r["gen_s"] or 0 for r in records) / n, 2) if n else None,
        "failures": [{"id": r["id"], "error": r["error"]} for r in records if r["error"]],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"t2c-{args.model.replace(':', '_').replace('/', '_')}-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": records}, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"generated : {gen_ok}/{n}")
    if not args.no_exec:
        print(f"executed  : {exec_ok}/{n}")
    if args.compare:
        print(f"correct   : {summary['correct']}/{n}")
    print(f"avg gen   : {summary['avg_gen_s']}s")
    for f in summary["failures"]:
        print(f"  {f['id']}: {f['error']}")
    print(f"report    : {out}")
    return 0 if (gen_ok == n and (args.no_exec or exec_ok == n)) else 1


if __name__ == "__main__":
    sys.exit(main())