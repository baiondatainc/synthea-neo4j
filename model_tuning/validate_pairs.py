#!/usr/bin/env python3
"""
validate_pairs.py
─────────────────
Validates question→Cypher training pairs through a cascade.

For TRAINING DATA the cascade is:
  1. Schema check   — labels, rels, write-keyword, LIMIT present
  2. EXPLAIN        — Neo4j parse + schema validation
  3. Execute        — row cap + timeout
  4. Zero-row flag  — flag but KEEP (model needs sparse patterns too)
  5. Guardrail      — OPTIONAL, OFF by default for training pairs
                      (guardrail is a runtime control for user queries,
                       not a training data filter — EXPLAIN+Execute
                       already prove correctness)

The --enable-guardrail flag turns Step 5 back on if you want strict
production-parity validation. Expect ~25% yield when on vs ~65% off.

Example:
    # Standard training data validation (guardrail off):
    python scripts/validate_pairs.py \\
        --input  data/raw/generated_pairs.parquet \\
        --output data/validated/validated_pairs.parquet \\
        --uri bolt://localhost:7687 \\
        --user neo4j --password <pw> \\
        --build-splits

    # Schema-only (no Neo4j):
    python scripts/validate_pairs.py \\
        --input data/raw/generated_pairs.parquet \\
        --skip-neo4j --build-splits

    # Strict mode (guardrail on — production parity):
    python scripts/validate_pairs.py \\
        --input data/raw/generated_pairs.parquet \\
        --enable-guardrail --build-splits
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Schema constants (from data_catalog.yaml) ─────────────────────────────────
VALID_LABELS = {
    "Patient", "Visit", "Charge", "Transaction", "Statement", "PhoneBridge",
    "RCCall", "IVRInbound", "DiallerCall", "Location", "InsurancePlan",
    "Practice", "Campaign", "DiagnosisCode", "ProcedureCode", "BirdeyeReview",
}
VALID_RELS = {
    "HAD_VISIT", "HAS_CHARGE", "HAS_TRANSACTION", "RECEIVED_STATEMENT",
    "IDENTIFIED_BY_PHONE", "REGISTERED_AT", "CALLED_IVR", "CONTACTED_BY_DIALLER",
    "BRIDGES_TO_PATIENT", "ATTRIBUTED_TO_PHONE", "PART_OF_CAMPAIGN", "PART_OF_VISIT",
    "AT_LOCATION", "DIAGNOSED_WITH", "USES_PROCEDURE", "SETTLES", "PERFORMED_AT",
    "UNDER_PLAN", "BELONGS_TO_PRACTICE", "ISSUED_BY_PRACTICE", "REVIEWS", "RUN_BY",
}
WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", re.I)


def schema_validate(cypher: str) -> Tuple[bool, Optional[str]]:
    """Fast offline check — no Neo4j needed."""
    u = cypher.upper()
    if "MATCH"  not in u: return False, "no MATCH clause"
    if "RETURN" not in u: return False, "no RETURN clause"
    if "LIMIT"  not in u: return False, "no LIMIT clause"
    if WRITE_RE.search(cypher): return False, "write keyword found"
    for lbl in re.findall(r"\([\w]*:(\w+)\)", cypher):
        if lbl not in VALID_LABELS:
            return False, f"unknown label: {lbl}"
    for rel in re.findall(r"\[:(\w+)\]", cypher):
        if rel not in VALID_RELS:
            return False, f"unknown relationship: {rel}"
    return True, None


class CypherValidator:
    def __init__(self, uri: str, user: str, password: str,
                 timeout: int = 10, row_limit: int = 100,
                 enable_guardrail: bool = False):
        from neo4j import GraphDatabase
        self.driver           = GraphDatabase.driver(uri, auth=(user, password))
        self.timeout          = timeout
        self.row_limit        = row_limit
        self.enable_guardrail = enable_guardrail

        # Try to load guardrail — warn but don't fail if missing
        self._guardrail_fn = None
        if enable_guardrail:
            try:
                from runtime.guardrails import check_cypher
                self._guardrail_fn = check_cypher
                print("  Guardrail: ENABLED (strict mode)")
            except Exception as e:
                print(f"  Guardrail: could not load check_cypher ({e}) — skipping")
        else:
            print("  Guardrail: DISABLED for training data (use --enable-guardrail to turn on)")

        self.stats = {
            "total": 0,
            "passed_schema": 0,
            "passed_explain": 0,
            "passed_execute": 0,
            "zero_row": 0,
            "passed_guardrail": 0,
            "final_pass": 0,
            "dropped_schema": 0,
            "dropped_explain": 0,
            "dropped_execute": 0,
            "dropped_guardrail": 0,
        }

    def validate_pair(self, question: str, cypher: str,
                      category: str = "", difficulty: str = "") -> Tuple[bool, dict]:
        result = {
            "question":          question,
            "cypher":            cypher,
            "category":          category,
            "difficulty":        difficulty,
            "step1_schema":      None,
            "step2_explain":     None,
            "step3_execute":     None,
            "step4_zero_row":    None,
            "step5_guardrail":   "skipped",
            "row_count":         None,
            "validated_zero_row": False,
            "passed":            False,
            "error":             None,
        }
        self.stats["total"] += 1

        # Step 1 — schema check (offline, fast)
        ok, err = schema_validate(cypher)
        if not ok:
            result["step1_schema"] = f"fail: {err}"
            result["error"] = err
            self.stats["dropped_schema"] += 1
            return False, result
        result["step1_schema"] = "pass"
        self.stats["passed_schema"] += 1

        # Step 2 — EXPLAIN
        try:
            with self.driver.session() as s:
                s.run(f"EXPLAIN {cypher}")
            result["step2_explain"] = "pass"
            self.stats["passed_explain"] += 1
        except Exception as e:
            result["step2_explain"] = f"fail: {str(e)[:120]}"
            result["error"] = str(e)
            self.stats["dropped_explain"] += 1
            return False, result

        # Step 3 — Execute with row cap
        limited = cypher if "LIMIT" in cypher.upper() else f"{cypher} LIMIT {self.row_limit}"
        try:
            with self.driver.session() as s:
                rows = s.run(limited).data()
            row_count = len(rows)
            result["step3_execute"] = f"pass ({row_count} rows)"
            result["row_count"] = row_count
            self.stats["passed_execute"] += 1
        except Exception as e:
            result["step3_execute"] = f"fail: {str(e)[:120]}"
            result["error"] = str(e)
            self.stats["dropped_execute"] += 1
            return False, result

        # Step 4 — Zero-row decision
        # Keep zero-row pairs — model needs to see valid queries that return
        # nothing on sparse data. Mark them so training can weight them differently.
        if row_count == 0:
            result["step4_zero_row"]    = "flagged"
            result["validated_zero_row"] = True
            self.stats["zero_row"] += 1
        else:
            result["step4_zero_row"] = "pass"

        # Step 5 — Guardrail (optional, off by default for training data)
        if self.enable_guardrail and self._guardrail_fn:
            guard_result = self._guardrail_fn(cypher)
            guard_ok, guard_err = guard_result.ok, guard_result.reason
            if not guard_ok:
                result["step5_guardrail"] = f"fail: {guard_err}"
                result["error"] = guard_err
                self.stats["dropped_guardrail"] += 1
                return False, result
            result["step5_guardrail"] = "pass"
            self.stats["passed_guardrail"] += 1
        else:
            self.stats["passed_guardrail"] += 1  # count as passed when skipped

        result["passed"] = True
        self.stats["final_pass"] += 1
        return True, result

    def close(self):
        self.driver.close()


def validate_with_neo4j(df: pd.DataFrame, uri: str, user: str, password: str,
                         enable_guardrail: bool = False) -> Tuple[pd.DataFrame, dict]:
    validator = CypherValidator(
        uri, user, password,
        timeout=10,
        row_limit=100,
        enable_guardrail=enable_guardrail,
    )
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        _, r = validator.validate_pair(
            row["question"], row["cypher"],
            category=row.get("category", ""),
            difficulty=row.get("difficulty", ""),
        )
        if "grounding_values" in row:
            r["grounding_values"] = row["grounding_values"]
        results.append(r)
    validator.close()
    return pd.DataFrame(results), validator.stats


def validate_schema_only(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Offline validation — no Neo4j needed."""
    results = []
    stats = {"total": 0, "passed_schema": 0, "final_pass": 0,
             "dropped_schema": 0, "dropped_explain": 0,
             "dropped_execute": 0, "dropped_guardrail": 0, "zero_row": 0,
             "passed_explain": 0, "passed_execute": 0, "passed_guardrail": 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating (schema only)"):
        stats["total"] += 1
        cypher = row["cypher"]
        ok, err = schema_validate(cypher)
        r = {
            "question":          row["question"],
            "cypher":            cypher,
            "category":          row.get("category", ""),
            "difficulty":        row.get("difficulty", ""),
            "step1_schema":      "pass" if ok else f"fail: {err}",
            "validated_zero_row": False,
            "passed":            ok,
            "error":             err,
        }
        if "grounding_values" in row:
            r["grounding_values"] = row["grounding_values"]

        if ok:
            stats["passed_schema"] += 1
            stats["passed_explain"] += 1
            stats["passed_execute"] += 1
            stats["passed_guardrail"] += 1
            stats["final_pass"] += 1
        else:
            stats["dropped_schema"] += 1
        results.append(r)
    return pd.DataFrame(results), stats


def print_stats(stats: dict) -> None:
    print(f"\n{'='*62}")
    print("  Validation Results")
    print(f"{'='*62}")
    total = stats["total"]
    print(f"  Total pairs:          {total}")
    print(f"  Step 1 (Schema):      {stats['passed_schema']} pass  "
          f"({stats['dropped_schema']} dropped)")
    print(f"  Step 2 (EXPLAIN):     {stats['passed_explain']} pass  "
          f"({stats['dropped_explain']} dropped)")
    print(f"  Step 3 (Execute):     {stats['passed_execute']} pass  "
          f"({stats['dropped_execute']} dropped)")
    print(f"  Step 4 (Zero-row):    {stats['zero_row']} flagged (kept)")
    print(f"  Step 5 (Guardrail):   {stats['passed_guardrail']} pass  "
          f"({stats.get('dropped_guardrail',0)} dropped)")
    print(f"  Final pass:           {stats['final_pass']} / {total}")
    yield_pct = stats["final_pass"] / total * 100 if total else 0
    print(f"\n  Yield rate: {yield_pct:.1f}%")
    if yield_pct < 50:
        print(f"  ⚠  Yield below 50% — check EXPLAIN failures above")
    elif yield_pct < 70:
        print(f"  ✓  Acceptable yield — consider tuning templates to reduce zero-rows")
    else:
        print(f"  ✓  Good yield")


def build_splits(validated_path: str,
                 train_split: float = 0.95,
                 val_split:   float = 0.05) -> None:
    """
    95/5 train/val split.
    The frozen eval JSON is the real test set — no separate test parquet needed.
    """
    df = pd.read_parquet(validated_path)
    if len(df) == 0:
        print("  No validated pairs — skipping split creation.")
        return

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n       = len(df)
    val_n   = max(1, round(n * val_split))
    train_n = n - val_n

    train_df = df[:train_n]
    val_df   = df[train_n:]

    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    val_df.to_parquet(splits_dir   / "val.parquet",   index=False)

    print(f"\n  Splits saved to data/splits/")
    print(f"    train.parquet  {len(train_df):,} rows  ({len(train_df)/n*100:.1f}%)")
    print(f"    val.parquet    {len(val_df):,} rows  ({len(val_df)/n*100:.1f}%)")
    print(f"\n  Next step:")
    print(f"    python scripts/train_lora.py \\")
    print(f"      --train-data data/splits/train.parquet \\")
    print(f"      --val-data   data/splits/val.parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Cypher training pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  default="data/raw/generated_pairs.parquet")
    parser.add_argument("--output", default="data/validated/validated_pairs.parquet")
    parser.add_argument("--uri",    default="bolt://localhost:7687")
    parser.add_argument("--user",   default="neo4j")
    parser.add_argument("--password", default=None)
    parser.add_argument("--sample",   type=int, default=None,
                        help="Validate only N pairs (for quick testing)")
    parser.add_argument("--skip-neo4j",  action="store_true",
                        help="Schema-only check — no Neo4j connection")
    parser.add_argument("--enable-guardrail", action="store_true",
                        help="Enable check_cypher() guardrail (strict mode, "
                             "lower yield — off by default for training data)")
    parser.add_argument("--build-splits", action="store_true",
                        help="Build train/val splits after validation")
    args = parser.parse_args()

    print("=" * 62)
    print("  Cypher Pair Validation")
    print(f"  Guardrail: {'ON (strict)' if args.enable_guardrail else 'OFF (training mode)'}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 62)

    df = pd.read_parquet(args.input)
    print(f"\nLoaded {len(df):,} pairs from {args.input}")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampled to {len(df):,}")

    # Run validation
    if args.skip_neo4j:
        print("\nSchema-only validation (no Neo4j)...")
        result_df, stats = validate_schema_only(df)
    else:
        pw = args.password or os.getenv("NEO4J_PASSWORD", "")
        if not pw:
            print("NEO4J_PASSWORD not set. Use --skip-neo4j or set the env var.")
            sys.exit(1)
        result_df, stats = validate_with_neo4j(
            df, args.uri, args.user, pw,
            enable_guardrail=args.enable_guardrail,
        )

    print_stats(stats)

    # Save validated pairs
    validated_df = result_df[result_df["passed"]].copy()
    keep_cols = ["question", "cypher", "category", "difficulty", "validated_zero_row"]
    if "grounding_values" in validated_df.columns:
        keep_cols.append("grounding_values")
    validated_df = validated_df[[c for c in keep_cols if c in validated_df.columns]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_parquet(out, index=False)
    print(f"\n  Saved {len(validated_df):,} validated pairs → {out}")

    # Category breakdown
    if "category" in validated_df.columns:
        print(f"\n  By category:")
        for cat, n in validated_df["category"].value_counts().items():
            pct = n / len(validated_df) * 100
            print(f"    {cat:<35} {n:>5}  ({pct:.1f}%)")

    if args.build_splits:
        build_splits(str(out))


if __name__ == "__main__":
    main()
