"""
Backfill missing patient_navigation_map columns onto Patient nodes,
restricted to patients that ALREADY EXIST in Neo4j.

Flow:
  1. Pull all Patient.patientId values from the graph.
  2. Read only the needed columns from the parquet, build the same
     composite key (Source_Database_Code + ':' + PatientID).
  3. Filter the parquet to rows whose key exists in the graph.
  4. Batch-update those nodes with SET p += props.

Usage:
    pip install neo4j pandas pyarrow
    python backfill_patient_props.py

Idempotent: re-running just re-SETs the same values.
"""

import math
import pandas as pd
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "patient_navigation_map.parquet"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")          # <-- change
NEO4J_DB = "neo4j"
BATCH_SIZE = 5000
INCLUDE_LOW_TIER = False                     # first_email_sent etc.

# ── Column tiers ─────────────────────────────────────────────────────────────
DATE_COLS = [
    "first_visit_date", "last_visit_date",
    "first_statement_date", "last_statement_date",
]
INT_COLS = [
    "charge_count", "transaction_count", "statement_count",
    "in_window_charge_count", "in_window_statement_count",
    "rv_in_calls_window", "rv_out_calls_window",
    "rc_calls_window", "rc_attributed_calls",
    "total_calls_window",
]
FLOAT_COLS = [
    # lifetime financial rollups
    "total_charged", "total_paid", "total_adjusted", "outstanding_balance",
    # 12-month operational window
    "in_window_charged", "in_window_paid",
    # full adjustment bucket set (adj_bad_debt may already be loaded; harmless)
    "adj_contractual", "adj_bad_debt", "adj_collection_agency",
    "adj_charity_care", "adj_refund_reversal", "adj_payment_plan", "adj_other",
]
BOOL_COLS = [
    "has_visits", "has_charges", "has_transactions",
    "has_statements", "has_email", "has_phone",
]
# Source column -> graph property. Plan fields are renamed with a primary_
# prefix: they are a one-per-patient snapshot (primary/most recent plan) and
# must not collide with InsurancePlan node modeling or the YAML plan_type
# recode (HMO/PPO/...) — these carry raw source values (e.g. MCDASSIGN, COMM).
STR_COLS = {
    "PlanName": "primary_plan_name",
    "Carrier_Name": "primary_carrier_name",
    "PlanType": "primary_plan_type",
}

LOW_DATE_COLS = ["first_email_sent", "first_text_sent"]
LOW_BOOL_COLS = [
    "has_inbound_calls", "has_outbound_calls", "has_ringcentral",
    "has_campaign_assignment", "has_in_window_activity",
    "has_birdeye_at_primary_location",
]

if INCLUDE_LOW_TIER:
    DATE_COLS += LOW_DATE_COLS
    BOOL_COLS += LOW_BOOL_COLS

ALL_PROP_COLS = DATE_COLS + INT_COLS + FLOAT_COLS + BOOL_COLS + list(STR_COLS)
KEY_COLS = ["Source_Database_Code", "PatientID"]

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# ── 1. What's actually in the graph? ─────────────────────────────────────────
print("Fetching existing Patient IDs from Neo4j ...")
with driver.session(database=NEO4J_DB) as session:
    existing_ids = set(
        session.run(
            "MATCH (p:Patient) WHERE p.patientId IS NOT NULL "
            "RETURN p.patientId AS id"
        ).value("id")
    )
print(f"  {len(existing_ids):,} Patient nodes in graph")

if not existing_ids:
    raise SystemExit("No Patient nodes found — nothing to backfill.")

# Sanity peek: key format + practice codes actually loaded
sample = list(existing_ids)[:5]
graph_codes = {i.split(":", 1)[0] for i in existing_ids if ":" in i}
print(f"  sample keys: {sample}")
print(f"  practice codes in graph: {sorted(graph_codes)}")

# ── 2. Read parquet (only needed columns) ────────────────────────────────────
print(f"\nReading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH, columns=KEY_COLS + ALL_PROP_COLS)
print(f"  {len(df):,} parquet rows")

df = df[df["Source_Database_Code"].notna() & df["PatientID"].notna()]
df["patientId"] = (
    df["Source_Database_Code"].astype(str).str.strip()
    + ":"
    + df["PatientID"].astype(str).str.strip()
)

parquet_codes = set(df["Source_Database_Code"].astype(str).str.strip().unique())

# ── 3. Filter to graph-resident patients only ────────────────────────────────
df = df[df["patientId"].isin(existing_ids)]
df = df.drop_duplicates(subset="patientId", keep="last")
print(f"  {len(df):,} parquet rows match an existing Patient node")

coverage = len(df) / len(existing_ids) * 100
print(f"  coverage: {coverage:.1f}% of graph patients have a parquet row")

if coverage < 50:
    print("\n  WARNING: low coverage. Likely the practice-code mismatch from the")
    print("  dictionary diff (MD: PMR/ESR/ACRB/... vs YAML: RADM/RADH/...).")
    print(f"  Codes in graph:   {sorted(graph_codes)}")
    print(f"  Codes in parquet: {sorted(parquet_codes)}")
    print("  If these differ, add a translation map before the key build step.")

if df.empty:
    raise SystemExit("Nothing to update after filtering.")


# ── 4. Clean + batch update ──────────────────────────────────────────────────
def clean_value(col, v):
    """Convert pandas values to neo4j-driver-friendly Python types."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    if col in DATE_COLS:
        return pd.Timestamp(v).date()      # driver maps date -> Neo4j Date
    if col in INT_COLS:
        return int(v)
    if col in FLOAT_COLS:
        return float(v)
    if col in BOOL_COLS:
        return bool(int(v)) if not isinstance(v, bool) else v
    if col in STR_COLS:
        s = str(v).strip()
        return s if s else None            # drop empty strings
    return v


def make_rows(frame):
    for rec in frame.to_dict("records"):
        props = {}
        for col in ALL_PROP_COLS:
            val = clean_value(col, rec.get(col))
            if val is not None:            # skip nulls: don't erase, don't bloat
                props[STR_COLS.get(col, col)] = val   # apply rename if any
        if props:
            yield {"patientId": rec["patientId"], "props": props}


rows = list(make_rows(df))
print(f"\n{len(rows):,} rows with at least one non-null property to set")

CYPHER = """
UNWIND $rows AS row
MATCH (p:Patient {patientId: row.patientId})
SET p += row.props
RETURN count(p) AS matched
"""

matched_total = 0
with driver.session(database=NEO4J_DB) as session:
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        result = session.execute_write(
            lambda tx, b=batch: tx.run(CYPHER, rows=b).single()["matched"]
        )
        matched_total += result
        print(f"  batch {i // BATCH_SIZE + 1}: updated {result:,}/{len(batch):,}")

driver.close()

print(f"\nDone. Updated {matched_total:,} Patient nodes.")
print("\nVerify with:")
print("  MATCH (p:Patient) WHERE p.last_visit_date IS NOT NULL")
print("  RETURN count(p) AS with_dates")