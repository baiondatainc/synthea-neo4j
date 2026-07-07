"""
Run this on your server:
  python /tmp/diagnose.py --input data/raw/generated_pairs.parquet \
      --uri bolt://localhost:7687 --user neo4j --password <pw>

It shows exactly WHY each pair fails at each step.
"""
import argparse, re, sys
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

VALID_LABELS = {
    "Patient","Visit","Charge","Transaction","Statement","PhoneBridge",
    "RCCall","IVRInbound","DiallerCall","Location","InsurancePlan","Practice",
    "Campaign","DiagnosisCode","ProcedureCode","BirdeyeReview",
}
VALID_RELS = {
    "HAD_VISIT","HAS_CHARGE","HAS_TRANSACTION","RECEIVED_STATEMENT",
    "IDENTIFIED_BY_PHONE","REGISTERED_AT","CALLED_IVR","CONTACTED_BY_DIALLER",
    "BRIDGES_TO_PATIENT","ATTRIBUTED_TO_PHONE","PART_OF_CAMPAIGN","PART_OF_VISIT",
    "AT_LOCATION","DIAGNOSED_WITH","USES_PROCEDURE","SETTLES","PERFORMED_AT",
    "UNDER_PLAN","BELONGS_TO_PRACTICE","ISSUED_BY_PRACTICE","REVIEWS","RUN_BY",
}
WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", re.I)

def schema_check(cypher):
    u = cypher.upper()
    if "MATCH" not in u:  return False, "no MATCH"
    if "RETURN" not in u: return False, "no RETURN"
    if "LIMIT" not in u:  return False, "no LIMIT"
    if WRITE_RE.search(cypher): return False, "write keyword"
    for lbl in re.findall(r"\([\w]*:(\w+)\)", cypher):
        if lbl not in VALID_LABELS:
            return False, f"bad_label:{lbl}"
    for rel in re.findall(r"\[:(\w+)\]", cypher):
        if rel not in VALID_RELS:
            return False, f"bad_rel:{rel}"
    return True, "ok"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/generated_pairs.parquet")
    parser.add_argument("--uri",   default="bolt://localhost:7687")
    parser.add_argument("--user",  default="neo4j")
    parser.add_argument("--password", default=None)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()

    import os
    pw = args.password or os.getenv("NEO4J_PASSWORD","")

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} pairs\n")

    # ── Step 1: schema-only diagnosis ─────────────────────────────────────────
    schema_fails = Counter()
    schema_fail_examples = defaultdict(list)
    for _, row in df.iterrows():
        ok, reason = schema_check(row["cypher"])
        if not ok:
            schema_fails[reason] += 1
            if len(schema_fail_examples[reason]) < 3:
                schema_fail_examples[reason].append(row["cypher"][:120])

    print("=== SCHEMA FAILURES (no Neo4j needed) ===")
    for reason, count in schema_fails.most_common():
        print(f"  {count:4d}  {reason}")
        for ex in schema_fail_examples[reason]:
            print(f"         → {ex}")
    print()

    if args.schema_only or not pw:
        print("Pass --password to also run Neo4j EXPLAIN + guardrail diagnosis.")
        return

    # ── Step 2: EXPLAIN failures ───────────────────────────────────────────────
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))

    explain_fails = Counter()
    explain_examples = defaultdict(list)
    guardrail_fails = Counter()
    guard_examples = defaultdict(list)
    zero_row_cats = Counter()

    sys.path.insert(0, ".")
    try:
        from runtime.guardrails import check_cypher
        has_guardrail = True
    except Exception:
        has_guardrail = False
        print("Warning: could not import check_cypher — skipping guardrail diagnosis")

    schema_pass = [row for _, row in df.iterrows() if schema_check(row["cypher"])[0]]
    print(f"Schema pass: {len(schema_pass)} / {len(df)}")

    for row in schema_pass:
        cypher = row["cypher"]
        cat    = row.get("category","?")

        # EXPLAIN
        try:
            with driver.session() as s:
                s.run(f"EXPLAIN {cypher}")
        except Exception as e:
            msg = str(e)[:80]
            explain_fails[msg] += 1
            if len(explain_examples[msg]) < 2:
                explain_examples[msg].append(cypher[:150])
            continue

        # Execute
        try:
            with driver.session() as s:
                limited = cypher if "LIMIT" in cypher.upper() else f"{cypher} LIMIT 100"
                rows = s.run(limited).data()
            row_count = len(rows)
        except Exception as e:
            explain_fails["execute_err:"+str(e)[:60]] += 1
            continue

        if row_count == 0:
            zero_row_cats[cat] += 1

        # Guardrail
        if has_guardrail:
            ok, err = check_cypher(cypher)
            if not ok:
                guardrail_fails[err] += 1
                if len(guard_examples[err]) < 2:
                    guard_examples[err].append(cypher[:150])

    driver.close()

    print("\n=== EXPLAIN / EXECUTE FAILURES ===")
    for reason, count in explain_fails.most_common(15):
        print(f"  {count:4d}  {reason}")
        for ex in explain_examples.get(reason, []):
            print(f"         → {ex}")

    print("\n=== GUARDRAIL FAILURES ===")
    for reason, count in guardrail_fails.most_common(15):
        print(f"  {count:4d}  {reason}")
        for ex in guard_examples.get(reason, []):
            print(f"         → {ex}")

    print("\n=== ZERO-ROW BY CATEGORY ===")
    for cat, count in zero_row_cats.most_common():
        print(f"  {count:4d}  {cat}")

if __name__ == "__main__":
    main()