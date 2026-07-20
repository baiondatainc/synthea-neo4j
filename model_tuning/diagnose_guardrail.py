#!/usr/bin/env python3
"""
Show exactly what the model generated and why the guardrail rejected it.

Run from project root:
    python diagnose_guardrail2.py
"""
import json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")

results_path = Path("eval_results/latest.json")
if not results_path.exists():
    print("Run eval_runner.py first to generate eval_results/latest.json")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

results = data["results"]

print(f"Total: {len(results)}")
print(f"Generated: {sum(1 for r in results if r['generated'])}")
print(f"Guardrail pass: {sum(1 for r in results if r['guardrail_pass'])}")

# Show guardrail errors
errors = Counter()
for r in results:
    if r["generated"] and not r["guardrail_pass"]:
        err = r.get("guardrail_error", "unknown")
        errors[err] += 1

print(f"\n=== GUARDRAIL REJECTION REASONS ===")
for reason, count in errors.most_common(15):
    print(f"  {count:3d}  {reason}")

# Show sample generated Cypher for failing cases
print(f"\n=== SAMPLE FAILING OUTPUTS (first 5) ===")
shown = 0
for r in results:
    if r["generated"] and not r["guardrail_pass"]:
        print(f"\n  Q: {r['question'][:70]}")
        print(f"  Error: {r['guardrail_error']}")
        cypher = r.get("generated_cypher", "")
        print(f"  Cypher: {cypher[:300]}")
        shown += 1
        if shown >= 5:
            break

# Show the 3 that DID pass guardrail
print(f"\n=== GUARDRAIL PASSING OUTPUTS ===")
for r in results:
    if r["guardrail_pass"]:
        print(f"\n  Q: {r['question'][:70]}")
        print(f"  Cypher: {r.get('generated_cypher','')[:200]}")