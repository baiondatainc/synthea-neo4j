#!/usr/bin/env python3
"""
build_dataset.py  v2 — schema-aware dataset builder for RP Text2Cypher
──────────────────────────────────────────────────────────────────────
Builds train/val/test splits from validated (question, cypher) pairs, with
awareness of the 16-label graph schema.

What changed vs v1:
  1. SCHEMA TAGGING — every pair is auto-tagged from its Cypher:
       - node labels used         (Patient, Visit, BirdeyeReview, ...)
       - relationships used       (HAD_VISIT, REVIEWS, SETTLES, ...)
       - hop_count                (# of relationship patterns)
       - has_aggregation, has_time (count/sum/avg, .year/.month/substring)
       - derived `category` and `difficulty` if not already present
  2. DEDUPLICATION — exact + normalized-question dedup BEFORE splitting.
  3. LEAKAGE-SAFE SPLIT — paraphrases of the same intent are grouped
     (by their generated cypher signature) and a whole group goes to ONE
     split. Otherwise "Monthly charge trend" in train and "charge trend
     by month" in val would inflate eval scores.
  4. FIXED STRATIFICATION — v1's stratify branch concatenated
     reset-index frames with original-index lookups → silently misaligned
     question/cypher rows. Now uses sklearn on the whole dataframe.
  5. COVERAGE REPORT — per-label and per-relationship counts, and a hard
     warning if any node label is missing from val (a label the model is
     never evaluated on is a label you can't trust).

Example:
    python scripts/build_dataset.py \\
        --input data/validated/validated_pairs.parquet \\
        --output data/splits/ \\
        --train-split 0.8 --val-split 0.1 \\
        --min-per-label 30
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# By default fetched LIVE from Neo4j (db.labels / db.relationshipTypes).
# The hardcoded sets below are only the OFFLINE FALLBACK — keep them in sync
# with Modelfile.qwen3-4b for runs without a database connection.
# ─────────────────────────────────────────────────────────────────────────────

NODE_LABELS = {
    "Patient", "Practice", "Location", "InsurancePlan", "Visit", "Charge",
    "Transaction", "Statement", "RCCall", "IVRInbound", "DiallerCall",
    "PhoneBridge", "Campaign", "BirdeyeReview", "DiagnosisCode", "ProcedureCode",
}

RELATIONSHIPS = {
    "REGISTERED_AT", "HAD_VISIT", "HAS_CHARGE", "HAS_TRANSACTION",
    "RECEIVED_STATEMENT", "CALLED_IVR", "CONTACTED_BY_DIALLER",
    "IDENTIFIED_BY_PHONE", "BELONGS_TO_PRACTICE", "ISSUED_BY_PRACTICE",
    "PERFORMED_AT", "UNDER_PLAN", "PART_OF_VISIT", "AT_LOCATION",
    "DIAGNOSED_WITH", "USES_PROCEDURE", "SETTLES", "PART_OF_CAMPAIGN",
    "ATTRIBUTED_TO_PHONE", "REVIEWS", "RUN_BY",
}


def fetch_schema_from_neo4j(uri: str, user: str, password: str,
                            database: str = "neo4j") -> tuple[set, set, list]:
    """
    Pull the LIVE schema from Neo4j:
      - db.labels()             → node labels that exist on ≥1 node
      - db.relationshipTypes()  → relationship types that exist on ≥1 edge
      - db.schema.visualization() → directed (Start)-[REL]->(End) patterns,
        returned for printing so you can eyeball direction correctness against
        the Modelfile's RELATIONSHIP DIRECTION block.

    NOTE: these procedures only report what's IN the graph. A label that was
    modeled but never loaded won't appear — which is exactly what you want
    for validation (a training pair querying an unloaded label would return
    empty results anyway), but it means an incomplete load silently shrinks
    the schema. The caller prints a diff vs the static fallback to catch this.
    """
    from neo4j import GraphDatabase  # pip install neo4j

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as s:
            labels = {r["label"] for r in
                      s.run("CALL db.labels() YIELD label RETURN label")}
            rels = {r["relationshipType"] for r in
                    s.run("CALL db.relationshipTypes() YIELD relationshipType "
                          "RETURN relationshipType")}
            patterns = []
            try:
                rec = s.run("CALL db.schema.visualization()").single()
                node_names = {n.element_id: next(iter(n.labels), "?")
                              for n in rec["nodes"]}
                for r in rec["relationships"]:
                    patterns.append(
                        f"({node_names.get(r.start_node.element_id, '?')})"
                        f"-[:{r.type}]->"
                        f"({node_names.get(r.end_node.element_id, '?')})"
                    )
            except Exception:
                pass  # visualization proc unavailable on some editions — optional
    finally:
        driver.close()
    return labels, rels, sorted(patterns)


def load_env_file(path: str | None) -> None:
    """
    Load NEO4J_* settings from a .env file into os.environ.
    Uses python-dotenv if installed; otherwise a minimal KEY=VALUE parser.
    Explicit --env-file wins; else auto-detects ./.env if present.
    Existing environment variables are NOT overwritten.
    """
    import os
    candidates = [path] if path else [".env"]
    for p in candidates:
        if not p or not Path(p).exists():
            if path:  # explicitly requested but missing → loud failure
                print(f"❌ --env-file not found: {path}")
                sys.exit(1)
            continue
        try:
            from dotenv import load_dotenv          # pip install python-dotenv
            load_dotenv(p, override=False)
        except ImportError:
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                os.environ.setdefault(k, v)
        print(f"🔐 Loaded env file: {p}")
        return


def resolve_schema(args) -> None:
    """Set module-level NODE_LABELS / RELATIONSHIPS, live if possible.

    Connection settings resolve in this order:
      CLI flag  >  environment / .env  >  built-in default
    Env vars: NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME),
              NEO4J_PASSWORD, NEO4J_DATABASE
    """
    global NODE_LABELS, RELATIONSHIPS
    import os

    uri      = args.neo4j_uri      or os.environ.get("NEO4J_URI")
    user     = args.neo4j_user     or os.environ.get("NEO4J_USER") \
                                   or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
    database = args.neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")

    if not uri:
        print("🗂️  Schema: static fallback (no --neo4j-uri / NEO4J_URI set)")
        return
    try:
        labels, rels, patterns = fetch_schema_from_neo4j(
            uri, user, password, database)
    except Exception as e:
        print(f"⚠️  Neo4j schema fetch failed ({e}) — using static fallback")
        return

    print(f"🗂️  Schema: LIVE from {uri} "
          f"({len(labels)} labels, {len(rels)} relationship types)")
    # Diff vs fallback — catches incomplete loads and Modelfile drift
    for name, live, static in (("labels", labels, NODE_LABELS),
                               ("relationships", rels, RELATIONSHIPS)):
        only_static = static - live
        only_live   = live - static
        if only_static:
            print(f"   ⚠ {name} in Modelfile fallback but NOT in graph "
                  f"(unloaded?): {sorted(only_static)}")
        if only_live:
            print(f"   ⚠ {name} in graph but NOT in Modelfile fallback "
                  f"(prompt drift?): {sorted(only_live)}")
    if patterns:
        print("   Directed patterns in graph:")
        for p in patterns:
            print(f"     {p}")

    NODE_LABELS = labels
    RELATIONSHIPS = rels


LABEL_RE = re.compile(r"\(\s*\w*\s*:\s*([A-Za-z]\w*)")
REL_RE   = re.compile(r"\[\s*\w*\s*:\s*([A-Z_]+)")
AGG_RE   = re.compile(r"\b(count|sum|avg|min|max|collect)\s*\(", re.I)
TIME_RE  = re.compile(r"\.(year|month|day|quarter)\b|substring\s*\(", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# TAGGING
# ─────────────────────────────────────────────────────────────────────────────

def tag_pair(cypher: str) -> dict:
    """Extract schema features from a Cypher query."""
    labels = sorted(set(LABEL_RE.findall(cypher)) & NODE_LABELS)
    rels   = sorted(set(REL_RE.findall(cypher)) & RELATIONSHIPS)
    unknown_labels = sorted(set(LABEL_RE.findall(cypher)) - NODE_LABELS)
    unknown_rels   = sorted(set(REL_RE.findall(cypher)) - RELATIONSHIPS)

    hop_count = len(re.findall(r"\[\s*:?\w*\s*:?[A-Z_]*\s*\]?[^\]]*?->|<-", cypher))
    # simpler robust proxy: count relationship-type occurrences
    hop_count = len(REL_RE.findall(cypher))

    has_agg  = bool(AGG_RE.search(cypher))
    has_time = bool(TIME_RE.search(cypher))

    # primary label = first label in the MATCH (what the question is "about")
    primary = labels[0] if labels else "NONE"
    m = LABEL_RE.search(cypher)
    if m and m.group(1) in NODE_LABELS:
        primary = m.group(1)

    # difficulty heuristic
    if hop_count == 0 and not has_agg:
        difficulty = "simple"
    elif hop_count <= 1:
        difficulty = "medium"
    elif hop_count == 2 or (has_agg and has_time):
        difficulty = "hard"
    else:
        difficulty = "very_hard"

    # category = primary label + query shape (used for stratification)
    shape = "trend" if (has_time and has_agg) else ("agg" if has_agg else "list")
    category = f"{primary}:{shape}"

    return {
        "labels": ",".join(labels),
        "relationships": ",".join(rels),
        "primary_label": primary,
        "hop_count": hop_count,
        "has_aggregation": has_agg,
        "has_time": has_time,
        "difficulty": difficulty,
        "derived_category": category,
        "unknown_labels": ",".join(unknown_labels),
        "unknown_rels": ",".join(unknown_rels),
    }


def normalize_question(q: str) -> str:
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def cypher_signature(cypher: str) -> str:
    """
    Structural signature of a query — same signature ⇒ same intent template.
    Strips variable names, literals, and whitespace so paraphrases that were
    generated from one template collapse to one group. Whole groups are
    assigned to a single split to prevent train→val leakage.
    """
    s = cypher.upper()
    s = re.sub(r"'[^']*'", "'X'", s)          # string literals
    s = re.sub(r"\b\d+(\.\d+)?\b", "N", s)    # numeric literals
    s = re.sub(r"\b[A-Z]\w*\.(?=\w)", "V.", s)  # variable aliases before dot
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    input_path: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    stratify_col: str = "derived_category",
    min_per_label: int = 30,
    seed: int = 42,
):
    print("=" * 70)
    print("📊 Building Training Dataset (schema-aware v2)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if not Path(input_path).exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)
    df = pd.read_parquet(input_path)
    df["question"] = df["question"].astype(str)
    df["cypher"]   = df["cypher"].astype(str)
    print(f"📥 {len(df):,} validated pairs loaded")

    # ── 2. Deduplicate ────────────────────────────────────────────────────────
    before = len(df)
    df["_qnorm"] = df["question"].map(normalize_question)
    df = df.drop_duplicates(subset="_qnorm").reset_index(drop=True)
    print(f"🧹 Dedup: {before:,} → {len(df):,}  (-{before - len(df)} duplicate questions)")

    # ── 3. Tag with schema features ───────────────────────────────────────────
    print("🏷️  Tagging pairs with schema features...")
    tags = df["cypher"].map(tag_pair).apply(pd.Series)
    # The input parquet may already carry same-named columns from the pair
    # generator (e.g. 'difficulty', 'category'). Duplicate column names make
    # df["difficulty"] 2-dimensional and crash value_counts/groupby.
    # Keep the originals as <col>_src; the derived (Cypher-based) values win.
    overlap = [c for c in tags.columns if c in df.columns]
    if overlap:
        print(f"   (input already had {overlap} — kept as *_src, using derived)")
        df = df.rename(columns={c: f"{c}_src" for c in overlap})
    df = pd.concat([df, tags], axis=1)
    df["_group"] = df["cypher"].map(cypher_signature)

    bad = df[(df["unknown_labels"] != "") | (df["unknown_rels"] != "")]
    if len(bad):
        print(f"   ⚠  {len(bad)} pairs reference labels/relationships NOT in the "
              f"schema — dropping them:")
        for _, r in bad.head(5).iterrows():
            print(f"      {r['question'][:60]!r} → {r['unknown_labels']}{r['unknown_rels']}")
        df = df[(df["unknown_labels"] == "") & (df["unknown_rels"] == "")].reset_index(drop=True)

    # ── 4. Coverage report ────────────────────────────────────────────────────
    print(f"\n📈 Schema coverage ({len(df):,} pairs):")
    label_counts = Counter()
    for s in df["labels"]:
        label_counts.update(l for l in s.split(",") if l)
    print("   Per node label:")
    for lbl in sorted(NODE_LABELS):
        c = label_counts.get(lbl, 0)
        flag = "  ⚠ BELOW MINIMUM — generate more pairs for this label" \
               if c < min_per_label else ""
        print(f"     {lbl:15s} {c:6d}{flag}")

    rel_counts = Counter()
    for s in df["relationships"]:
        rel_counts.update(r for r in s.split(",") if r)
    missing_rels = [r for r in RELATIONSHIPS if rel_counts.get(r, 0) == 0]
    if missing_rels:
        print(f"   ⚠ Relationships with ZERO training examples: {missing_rels}")

    print("   Difficulty:", dict(df["difficulty"].value_counts()))

    # ── 5. Group-aware, stratified split ──────────────────────────────────────
    # One row per group; the group inherits the modal stratification class.
    test_split = round(1.0 - train_split - val_split, 4)
    print(f"\n🔀 Split: {train_split:.0%} train / {val_split:.0%} val / "
          f"{test_split:.0%} test  (grouped by cypher signature)")

    groups = (
        df.groupby("_group")[stratify_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={stratify_col: "_strat"})
    )
    print(f"   {len(groups):,} unique query templates "
          f"({len(df)/len(groups):.1f} paraphrases/template avg)")

    # Merge rare stratification classes (<10 groups) into 'other' so sklearn
    # stratify doesn't fail on singleton classes.
    strat_counts = groups["_strat"].value_counts()
    rare = set(strat_counts[strat_counts < 10].index)
    if rare:
        groups.loc[groups["_strat"].isin(rare), "_strat"] = "other"
        print(f"   (merged {len(rare)} rare categories into 'other' for stratification)")

    g_trainval, g_test = train_test_split(
        groups, test_size=max(test_split, 0.01),
        stratify=groups["_strat"], random_state=seed,
    )
    g_train, g_val = train_test_split(
        g_trainval, test_size=val_split / (train_split + val_split),
        stratify=g_trainval["_strat"], random_state=seed,
    )

    train_df = df[df["_group"].isin(g_train["_group"])].sample(frac=1, random_state=seed)
    val_df   = df[df["_group"].isin(g_val["_group"])].sample(frac=1, random_state=seed)
    test_df  = df[df["_group"].isin(g_test["_group"])].sample(frac=1, random_state=seed)

    # ── 6. Post-split label coverage check ───────────────────────────────────
    def labels_in(d):
        c = Counter()
        for s in d["labels"]:
            c.update(l for l in s.split(",") if l)
        return set(c)

    missing_in_val = (labels_in(train_df) - labels_in(val_df)) & NODE_LABELS
    if missing_in_val:
        print(f"\n   ⚠ Labels present in TRAIN but absent from VAL: {sorted(missing_in_val)}")
        print(f"     → eval loss will never see these; generate more pairs or "
              f"lower --min-per-label enforcement upstream.")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    drop_cols = ["_qnorm", "_group", "unknown_labels", "unknown_rels"]
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = out / f"{name}.parquet"
        d.drop(columns=drop_cols).reset_index(drop=True).to_parquet(p, index=False)
        print(f"   ✓ {name:5s}: {p}  ({len(d):,} pairs, "
              f"{d['_group'].nunique():,} templates)")

    n = len(df)
    print(f"\n✅ {len(train_df)/n:5.1%} / {len(val_df)/n:5.1%} / {len(test_df)/n:5.1%}"
          f"  of {n:,} pairs")
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser(description="Schema-aware dataset builder")
    ap.add_argument("--input",  default="data/validated/validated_pairs.parquet")
    ap.add_argument("--output", default="data/splits/")
    ap.add_argument("--train-split", type=float, default=0.8)
    ap.add_argument("--val-split",   type=float, default=0.1)
    ap.add_argument("--stratify",    default="derived_category",
                    help="Column to stratify groups by (derived_category, "
                         "difficulty, primary_label)")
    ap.add_argument("--min-per-label", type=int, default=30,
                    help="Warn if a node label has fewer total pairs than this")
    ap.add_argument("--env-file", default=None,
                    help="Path to .env with NEO4J_URI / NEO4J_USER / "
                         "NEO4J_PASSWORD / NEO4J_DATABASE. If omitted, "
                         "./.env is auto-loaded when present.")
    ap.add_argument("--neo4j-uri", default=None,
                    help="e.g. bolt://localhost:7687 — fetch schema live; "
                         "overrides NEO4J_URI from env/.env")
    ap.add_argument("--neo4j-user", default=None,
                    help="overrides NEO4J_USER (default: neo4j)")
    ap.add_argument("--neo4j-password", default=None,
                    help="overrides NEO4J_PASSWORD — prefer .env over CLI "
                         "so the password stays out of shell history")
    ap.add_argument("--neo4j-database", default=None,
                    help="overrides NEO4J_DATABASE (default: neo4j)")
    args = ap.parse_args()

    load_env_file(args.env_file)
    resolve_schema(args)

    build_splits(
        args.input, args.output,
        args.train_split, args.val_split,
        stratify_col=args.stratify,
        min_per_label=args.min_per_label,
    )

    print(f"\nNext step:")
    print(f"  python scripts/train_lora.py \\")
    print(f"    --train-data {args.output}/train.parquet \\")
    print(f"    --val-data   {args.output}/val.parquet")


if __name__ == "__main__":
    main()