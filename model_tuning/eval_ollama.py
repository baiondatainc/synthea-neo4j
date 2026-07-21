#!/usr/bin/env python3
"""
eval_ollama.py — evaluate the DEPLOYED Ollama model against the live graph.
───────────────────────────────────────────────────────────────────────────
Unlike eval_runner.py (which loads the adapter in Python), this evaluates the
actual production artifact: quantized base + ADAPTER + Modelfile SYSTEM +
template, via the same /api/generate endpoint the serving pipeline uses.

Scoring per question:
  1.0  ran on Neo4j AND returned >=1 row, all quality gates passed
  0.5  ran but returned 0 rows (suspect: wrong direction/hop/filter)
  0.25 gate failure (write clause, wrong property, markdown fence) even if runs
  0.0  Cypher syntax error / no MATCH / generation failure

Connection settings come from env or .env (NEO4J_URI, NEO4J_USER,
NEO4J_PASSWORD) — same pattern as build_dataset.py. Do NOT pass passwords
on the command line.

Usage:
    # eval the deployed fine-tune:
    python model_tuning/eval_ollama.py --model text2cypher-ft \\
        --eval data/eval/eval.json --out eval_results/ft_v2.json

    # side-by-side with the prompt-only base:
    python model_tuning/eval_ollama.py --model text2cypher-ft \\
        --compare text2cypher \\
        --eval data/eval/eval.json --out eval_results/ft_v2_vs_base.json

eval.json format (frozen eval set — NEVER add these to training):
    [{"question": "How many patients?"}, ...]
  or with reference cypher (enables zero-row forgiveness when the reference
  also returns zero):
    [{"question": "...", "cypher": "MATCH ..."}]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from prompt import WRITE_RE, WRONG_PROPS, AGG_RE  # same gates as training

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def load_env():
    p = Path(".env")
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def generate(model: str, question: str, timeout: int = 300) -> tuple[str, float]:
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model, "prompt": question,
        "stream": False, "keep_alive": -1,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip(), time.time() - t0


def gate(cypher: str) -> list[str]:
    problems = []
    u = cypher.upper()
    if "MATCH" not in u or "RETURN" not in u:
        problems.append("not_cypher")
    if WRITE_RE.search(cypher):
        problems.append("write_clause")
    if WRONG_PROPS.search(cypher):
        problems.append("wrong_property")
    if "```" in cypher:
        problems.append("markdown_fence")
    if "SELECT " in u:
        problems.append("sql")
    return problems


def run_model(model: str, cases: list[dict], session) -> dict:
    results, scores = [], []
    print(f"\n━━ {model} ━━")
    for i, case in enumerate(cases, 1):
        q = case["question"]
        entry = {"question": q}
        try:
            cypher, latency = generate(model, q)
        except Exception as e:
            entry.update({"score": 0.0, "error": f"generation: {e}"})
            results.append(entry); scores.append(0.0)
            print(f"  [{i:2d}] 0.00  GEN-FAIL  {q[:50]}")
            continue

        entry["cypher"] = cypher
        entry["latency_s"] = round(latency, 1)
        problems = gate(cypher)

        row_count, exec_error = None, None
        if "not_cypher" not in problems and "sql" not in problems:
            try:
                clean = re.sub(r"^```\w*\n?|```$", "", cypher, flags=re.M).strip()
                rows = session.run(clean).data()
                row_count = len(rows)
            except Exception as e:
                exec_error = str(e)[:150]

        if exec_error or "not_cypher" in problems or "sql" in problems:
            score = 0.0
        elif problems:
            score = 0.25
        elif row_count == 0:
            # forgive zero rows if the reference query ALSO returns zero
            ref = case.get("cypher")
            ref_zero = False
            if ref:
                try:
                    ref_zero = len(session.run(ref).data()) == 0
                except Exception:
                    pass
            score = 1.0 if ref_zero else 0.5
        else:
            score = 1.0

        entry.update({"score": score, "row_count": row_count,
                      "problems": problems, "exec_error": exec_error})
        results.append(entry); scores.append(score)
        flag = {1.0: "OK  ", 0.5: "ZERO", 0.25: "GATE", 0.0: "FAIL"}[score]
        print(f"  [{i:2d}] {score:.2f}  {flag}  {latency:4.0f}s  {q[:50]}")

    mean = sum(scores) / max(len(scores), 1)
    print(f"  ── mean score: {mean:.3f}  "
          f"({sum(s == 1.0 for s in scores)}/{len(scores)} clean)")
    return {"model": model, "mean_score": round(mean, 3), "results": results}


def main():
    ap = argparse.ArgumentParser(description="Evaluate deployed Ollama model on live Neo4j")
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. text2cypher-ft")
    ap.add_argument("--compare", default=None,
                    help="Second tag to run side-by-side, e.g. text2cypher")
    ap.add_argument("--eval", default="data/eval/eval.json")
    ap.add_argument("--out",  default="eval_results/eval.json")
    args = ap.parse_args()

    load_env()
    uri  = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pw   = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        sys.exit("✗ NEO4J_PASSWORD not set — put it in .env, not on the CLI")

    eval_path = Path(args.eval)
    if not eval_path.exists():
        sys.exit(f"✗ {eval_path} not found. Create your frozen eval set first "
                 f"(list of {{'question': ...}} objects).")
    cases = json.loads(eval_path.read_text())
    # Accept both formats: ["question", ...] and [{"question": ..., "cypher": ...}]
    cases = [{"question": c} if isinstance(c, str) else c for c in cases]
    print(f"📋 {len(cases)} eval questions  ·  Ollama {OLLAMA_URL}  ·  Neo4j {uri}")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, pw))
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "runs": []}
    with driver.session() as session:
        report["runs"].append(run_model(args.model, cases, session))
        if args.compare:
            report["runs"].append(run_model(args.compare, cases, session))
    driver.close()

    if args.compare and len(report["runs"]) == 2:
        a, b = report["runs"]
        print(f"\n═══ {a['model']}: {a['mean_score']}   vs   "
              f"{b['model']}: {b['mean_score']} ═══")
        for ra, rb in zip(a["results"], b["results"]):
            if ra["score"] != rb["score"]:
                print(f"  Δ {ra['score']} vs {rb['score']}: {ra['question'][:60]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n💾 {out}")


if __name__ == "__main__":
    main()