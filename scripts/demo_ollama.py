#!/usr/bin/env python3
"""
demo_ollama.py  v3
──────────────────
Uses Ollama /api/chat endpoint which correctly applies the Modelfile
system prompt. This is why text2cypher works in `ollama run` but not
in the previous version that used /api/generate with a custom prompt.

Usage:
    python demo_ollama.py --model text2cypher \\
        --uri bolt://localhost:7687 --user neo4j --password <pw> --demo

    python demo_ollama.py --model text2cypher \\
        --question "How many patients are in TX?" \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>

    python demo_ollama.py --model text2cypher \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>
"""

import argparse, json, os, re, sys, urllib.request
from typing import Optional


def call_ollama_chat(question: str, model: str,
                     host: str = "http://localhost:11434") -> Optional[str]:
    """
    Use /api/chat — this respects the Modelfile SYSTEM prompt,
    exactly as `ollama run` does.
    """
    payload = json.dumps({
        "model":  model,
        "stream": False,
        "options": {
            "temperature":    0,
            "top_k":          1,
            "repeat_penalty": 1.05,
            "num_predict":    300,
        },
        "messages": [
            {"role": "user", "content": question},
        ],
    }).encode()

    try:
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def clean_cypher(raw: str) -> str:
    """Strip markdown fences, extract Cypher block."""
    raw = re.sub(r"```(?:cypher|sql)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"```", "", raw).strip()
    for kw in ("MATCH", "CALL", "WITH", "UNWIND", "OPTIONAL"):
        if kw in raw.upper():
            return raw[raw.upper().index(kw):].strip()
    return raw


WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.I)
LIMIT_RE  = re.compile(r"\bLIMIT\s+\d+", re.I)
PII_KEYS  = {"first_name","last_name","middle_name","dob","email",
             "phone","phone_norm","cell_norm","patient_id"}

def safety_check(cypher: str) -> tuple[bool, Optional[str]]:
    if not cypher or len(cypher.strip()) < 5:
        return False, "empty response"
    if WRITE_RE.search(cypher):
        return False, "write keyword blocked"
    if not any(k in cypher.upper() for k in ("MATCH","CALL","WITH","UNWIND")):
        return False, "no Cypher clause found"
    return True, None

def inject_limit(cypher: str, cap: int = 100) -> str:
    return cypher if LIMIT_RE.search(cypher) else cypher.rstrip(";") + f"\nLIMIT {cap}"

def redact(rows: list) -> list:
    return [{k: "[redacted]" if k.lower() in PII_KEYS else v
             for k, v in row.items()} for row in rows]


def run_pipeline(question: str, driver, model: str, host: str,
                 verbose: bool = True) -> dict:
    result = {"question": question, "cypher": None,
              "rows": [], "status": None, "error": None}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Q: {question}")

    raw = call_ollama_chat(question, model, host)
    if not raw:
        result["status"] = "ollama_failed"
        if verbose: print("  ✗ No response from Ollama")
        return result

    cypher = clean_cypher(raw)
    result["cypher"] = cypher

    if verbose:
        print(f"\n  Cypher:")
        for line in cypher.split("\n"):
            print(f"    {line}")

    ok, err = safety_check(cypher)
    if not ok:
        result["status"] = "blocked"
        result["error"]  = err
        if verbose: print(f"  ✗ Safety: {err}")
        return result

    # EXPLAIN
    try:
        with driver.session() as s:
            s.run(f"EXPLAIN {cypher}")
        if verbose: print("  ✓ EXPLAIN passed")
    except Exception as e:
        result["status"] = "explain_failed"
        result["error"]  = str(e)[:150]
        if verbose: print(f"  ✗ EXPLAIN: {str(e)[:80]}")
        return result

    # Execute
    try:
        with driver.session() as s:
            rows = s.run(inject_limit(cypher)).data()
        rows = redact(rows)
        result["rows"]   = rows
        result["status"] = "success"
        if verbose:
            print(f"  ✓ {len(rows)} rows returned")
            for i, row in enumerate(rows[:8], 1):
                print(f"    {i}. {row}")
            if len(rows) > 8:
                print(f"    ... {len(rows)-8} more")
    except Exception as e:
        result["status"] = "exec_failed"
        result["error"]  = str(e)[:150]
        if verbose: print(f"  ✗ Execute: {str(e)[:80]}")

    return result


DEMO_QUESTIONS = [
    "How many patients have an outstanding balance?",
    "What is the total outstanding balance by payor cohort?",
    "How many patients are self-pay?",
    "Show the top agents by RingCentral call volume",
    "What is the total amount collected via IVR payments?",
    "Which locations have the highest Birdeye rating?",
    "What are the most common diagnosis codes by charge count?",
    "Show propensity grade breakdown with average outstanding balance",
    "How many patients in Tennessee have a balance over $1000?",
    "What is the statement level distribution?",
    "What is the SLA breakdown for RingCentral calls?",
    "Show the top 10 charges by amount with their procedure code",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="text2cypher")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--uri",         default="bolt://localhost:7687")
    parser.add_argument("--user",        default="neo4j")
    parser.add_argument("--password",    default=None)
    parser.add_argument("--question",    default=None)
    parser.add_argument("--demo",        action="store_true")
    args = parser.parse_args()

    pw = args.password or os.getenv("NEO4J_PASSWORD", "")
    if not pw:
        print("NEO4J_PASSWORD not set.")
        sys.exit(1)

    print("=" * 60)
    print(f"  RP Knowledge Graph — Text2Cypher via Ollama")
    print(f"  Model: {args.model}")
    print("=" * 60)

    # Quick connectivity check
    try:
        req = urllib.request.Request(f"{args.ollama_host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data   = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            match  = next((m for m in models if args.model in m), None)
            if not match:
                print(f"  ⚠  '{args.model}' not found. Available: {models}")
                print(f"  Using first available: {models[0] if models else 'none'}")
            else:
                print(f"  ✓ Ollama: {match}")
    except Exception as e:
        print(f"  ✗ Ollama not reachable: {e}")
        sys.exit(1)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))
    try:
        with driver.session() as s:
            s.run("RETURN 1")
        print(f"  ✓ Neo4j connected")
    except Exception as e:
        print(f"  ✗ Neo4j: {e}")
        sys.exit(1)

    if args.question:
        run_pipeline(args.question, driver, args.model, args.ollama_host)
        driver.close()
        return

    if args.demo:
        passed = 0
        for q in DEMO_QUESTIONS:
            r = run_pipeline(q, driver, args.model, args.ollama_host)
            if r["status"] == "success":
                passed += 1
        print(f"\n{'='*60}")
        print(f"  {passed}/{len(DEMO_QUESTIONS)} questions succeeded")
        print("=" * 60)
        driver.close()
        return

    # Interactive
    print(f"\nType your question (quit to exit)\n")
    while True:
        try:
            q = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q", ""):
            break
        run_pipeline(q, driver, args.model, args.ollama_host)

    driver.close()
    print("\nDone.")

if __name__ == "__main__":
    main()