"""Run every question in questions.yaml through the text2cypher model and record the response.

    python run_t2c.py                     # all 64
    python run_t2c.py --ids Q01,Q05       # subset
    python run_t2c.py --no-exec           # only generate, don't run the Cypher
    python run_t2c.py --compare           # also compare rows with ground truth

Output: results/t2c-<model>-<stamp>.json   (one record per question)
        results/t2c-<model>-<stamp>.html   (browsable success/failure report)
        + a console summary.
Uses the same .env / harness.py as the pytest suite (T2C_MODEL, OLLAMA_URL, NEO4J_*).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import CFG, RESULTS_DIR, Graph, _clean_cypher, _ollama_generate, compare, load_questions  # noqa: E402


# ---------------------------------------------------------------- status logic
def status_of(rec: dict, compared: bool) -> str:
    """Single source of truth for a record's outcome flag."""
    if not rec["generated"]:
        return "GEN-FAIL"
    if rec["executed"] is None:          # --no-exec run
        return "OK"
    if rec["executed"] is False:
        return "EXEC-FAIL"
    if not compared:
        return "OK"
    return "CORRECT" if rec.get("correct") else "WRONG"


# ---------------------------------------------------------------- HTML report
_STATUS_META = {
    "CORRECT":   ("pass", "Correct"),
    "OK":        ("pass", "Ran ok"),
    "WRONG":     ("warn", "Wrong rows"),
    "EXEC-FAIL": ("fail", "Execution failed"),
    "GEN-FAIL":  ("fail", "Generation failed"),
}

_CSS = """
:root{--ink:#1c2230;--mut:#5c6575;--line:#dfe3ea;--bg:#f7f8fa;--card:#ffffff;
      --pass:#1a7f4b;--pass-bg:#e4f4ea;--warn:#9a6700;--warn-bg:#fbf0d7;
      --fail:#b3261e;--fail-bg:#fbe4e2;--code:#f0f2f6;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 "Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--bg);padding:28px 4vw 60px}
h1{font-size:22px;margin:0 0 2px}
.meta{color:var(--mut);font-size:13px;margin-bottom:22px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 18px;min-width:130px}
.card b{display:block;font-size:24px;font-weight:600}
.card span{color:var(--mut);font-size:13px}
.bar{height:8px;border-radius:4px;background:var(--fail-bg);overflow:hidden;margin:6px 0 26px;max-width:640px}
.bar i{display:block;height:100%;background:var(--pass)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;vertical-align:top;border-top:1px solid var(--line);font-size:14px}
th{background:#eef0f4;border-top:none;font-weight:600;font-size:13px}
tr.pass td:first-child{box-shadow:inset 3px 0 0 var(--pass)}
tr.warn td:first-child{box-shadow:inset 3px 0 0 var(--warn)}
tr.fail td:first-child{box-shadow:inset 3px 0 0 var(--fail)}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;font-weight:600;white-space:nowrap}
.badge.pass{color:var(--pass);background:var(--pass-bg)}
.badge.warn{color:var(--warn);background:var(--warn-bg)}
.badge.fail{color:var(--fail);background:var(--fail-bg)}
code,pre{font:12.5px/1.45 Consolas,Menlo,monospace}
pre{background:var(--code);border-radius:6px;padding:8px 10px;margin:6px 0 0;white-space:pre-wrap;word-break:break-word}
details summary{cursor:pointer;color:var(--mut);font-size:13px}
.err{color:var(--fail)}
td.num{text-align:right;white-space:nowrap;color:var(--mut)}
.q{max-width:420px}
.filters{margin:0 0 12px}
.filters button{border:1px solid var(--line);background:var(--card);border-radius:6px;padding:5px 12px;margin-right:6px;cursor:pointer;font-size:13px}
.filters button.on{border-color:var(--ink);font-weight:600}
"""

_JS = """
function flt(k,btn){
  document.querySelectorAll('.filters button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('tbody tr').forEach(r=>{
    r.style.display=(k==='all'||r.classList.contains(k))?'':'none';
  });
}
"""


def write_html(path: Path, summary: dict, records: list[dict], compared: bool) -> None:
    n = summary["questions"]
    passed = sum(1 for r in records if _STATUS_META[status_of(r, compared)][0] == "pass")
    pct = round(100 * passed / n) if n else 0

    def esc(v) -> str:
        return html.escape(str(v)) if v not in (None, "") else ""

    rows = []
    for r in records:
        st = status_of(r, compared)
        cls, label = _STATUS_META[st]
        detail = ""
        if r.get("cypher"):
            detail += f"<details><summary>cypher</summary><pre>{esc(r['cypher'])}</pre></details>"
        if r.get("sample"):
            detail += (f"<details><summary>sample rows ({r['rows']})</summary>"
                       f"<pre>{esc(json.dumps(r['sample'], indent=1, default=str)[:1500])}</pre></details>")
        if compared and r.get("compare_detail") and st == "WRONG":
            detail += f"<details open><summary>mismatch</summary><pre>{esc(r['compare_detail'])}</pre></details>"
        if r.get("error"):
            detail += f"<pre class='err'>{esc(r['error'])}</pre>"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{esc(r['id'])}</td>"
            f"<td>{esc(r['category'])}</td>"
            f"<td class='q'>{esc(r['question'])}</td>"
            f"<td><span class='badge {cls}'>{st}</span><br><span style='font-size:12px;color:var(--mut)'>{label}</span></td>"
            f"<td class='num'>{esc(r['gen_s'])}</td>"
            f"<td class='num'>{esc(r['exec_s'])}</td>"
            f"<td class='num'>{esc(r['rows'])}</td>"
            f"<td>{detail or '&mdash;'}</td>"
            f"</tr>"
        )

    correct_card = (f"<div class='card'><b>{summary['correct']}/{n}</b><span>correct</span></div>"
                    if compared else "")
    executed_card = (f"<div class='card'><b>{summary['executed']}/{n}</b><span>executed</span></div>"
                     if summary["executed"] is not None else "")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>text2cypher report &middot; {esc(summary['model'])}</title>
<style>{_CSS}</style></head><body>
<h1>text2cypher run report</h1>
<div class="meta">model {esc(summary['model'])} &nbsp;&bull;&nbsp; {esc(summary['run'])}
 &nbsp;&bull;&nbsp; avg generation {esc(summary['avg_gen_s'])}s</div>
<div class="cards">
  <div class="card"><b>{passed}/{n}</b><span>succeeded ({pct}%)</span></div>
  <div class="card"><b>{summary['generated']}/{n}</b><span>generated</span></div>
  {executed_card}{correct_card}
  <div class="card"><b>{len(summary['failures'])}</b><span>errors</span></div>
</div>
<div class="bar"><i style="width:{pct}%"></i></div>
<div class="filters">
  <button class="on" onclick="flt('all',this)">All ({n})</button>
  <button onclick="flt('pass',this)">Passed ({passed})</button>
  <button onclick="flt('warn',this)">Wrong ({sum(1 for r in records if status_of(r, compared) == 'WRONG')})</button>
  <button onclick="flt('fail',this)">Failed ({sum(1 for r in records if status_of(r, compared) in ('GEN-FAIL', 'EXEC-FAIL'))})</button>
</div>
<table>
<thead><tr><th>ID</th><th>Category</th><th>Question</th><th>Status</th>
<th>Gen&nbsp;s</th><th>Exec&nbsp;s</th><th>Rows</th><th>Detail</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<script>{_JS}</script>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--category")
    ap.add_argument("--no-exec", action="store_true", help="generate only, do not execute on Neo4j")
    ap.add_argument("--compare", action="store_true", help="also compare rows with ground-truth Cypher")
    ap.add_argument("--model", default=CFG.t2c_model)
    ap.add_argument("--no-html", action="store_true", help="skip the HTML report")
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

        rec["status"] = status_of(rec, args.compare)
        flag = rec["status"] if rec["status"] != "OK" else "ok"
        print(f"{q.id:4} {flag:9} {rec['gen_s'] or 0:5.1f}s  rows={rec['rows']!s:>5}  {rec['cypher'] or rec['error']}"
              .replace("\n", " ")[:170])
        records.append(rec)

    if graph:
        graph.close()

    # ---- summary + files
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
    stem = f"t2c-{args.model.replace(':', '_').replace('/', '_')}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = RESULTS_DIR / f"{stem}.json"
    out.write_text(json.dumps({"summary": summary, "results": records}, indent=2, default=str), encoding="utf-8")

    html_out = None
    if not args.no_html and n:
        html_out = RESULTS_DIR / f"{stem}.html"
        write_html(html_out, summary, records, args.compare)

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
    if html_out:
        print(f"html      : {html_out}")
    return 0 if (gen_ok == n and (args.no_exec or exec_ok == n)) else 1


if __name__ == "__main__":
    sys.exit(main())