"""
gen_derived.py
──────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates two derived/analytical tables found in schema CSVs
but not in table_inventory (they are built FROM core tables):

  • 03_supplementary/master_patient_index.parquet
  • 03_supplementary/patient_call_bridge.parquet

These are DERIVED tables — built by aggregating the core tables.
They are NOT source tables; they are outputs of analytical pipelines.

Schema sources:
  master_patient_index_schema.csv   (17 columns)
  patient_call_bridge_schema.csv    (5 columns)

Relationships:
  master_patient_index.PatientID    → patient.PatientID (unique humans)
  patient_call_bridge.PatientID     → patient.PatientID (FK)
  patient_call_bridge.AccountID     → rv_inbound.AccountID  (= PatientID per DQ-003)
                                    → rv_outbound.ACCOUNTID (= PatientID per DQ-003)

Run standalone:
    python gen_derived.py --out ./output
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from constants import SEED, REFRESH_DATE, PRACTICE_STATES
from helpers import rng


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PATIENT INDEX
# ─────────────────────────────────────────────────────────────────────────────

def generate_master_patient_index(
    patient_df: pd.DataFrame,
    visits_df: pd.DataFrame,
    charges_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    statements_df: pd.DataFrame,
    rv_inbound_df: pd.DataFrame,
    rv_outbound_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per unique human PatientID (collapsed across all practices).
    17 columns per master_patient_index_schema.csv.

    This is the cross-practice MPI — where multi-practice patients appear
    once, not once per practice.

    Columns:
      PatientID            int64    unique human
      n_source_databases   int64    how many practices they appear in
      source_database_codes str     comma-separated list e.g. "SAPA,PMR"
      is_multi_practice    str      "True" / "False"
      city                 str      most recent city from patient dim
      state                str      most recent state
      age                  float64  age in years at refresh date
      gender               str      from patient dim
      n_patient_rows       int64    total rows across all patient-practice records
      n_visits             int64    total visit rows
      n_charges            int64    total charge rows
      n_transactions       int64    total transaction rows
      n_statements         int64    total statement rows
      n_inbound_calls      int64    total rv_inbound calls
      n_outbound_calls     int64    total rv_outbound calls
      total_interactions   int64    sum of all above activity counts
      has_called           str      "True" if any calls, else "False"
    """
    # ── Per-patient aggregations ──────────────────────────────────────────────

    # Patient dimension: group by PatientID to get demographics + practice list
    pat_grp = patient_df.groupby("PatientID").agg(
        n_patient_rows   = ("PatientID", "count"),
        n_source_databases = ("Source_Database_Code", "nunique"),
        source_database_codes = ("Source_Database_Code", lambda x: ",".join(sorted(x.unique()))),
        city             = ("PatientCity",  "last"),
        state            = ("PatientState", "last"),
        gender           = ("PatientGender","last"),
        dob              = ("PatientDOB",   "last"),
    ).reset_index()

    # Activity counts from fact tables
    def count_by_pid(df, pid_col="PatientID"):
        if df is None or len(df) == 0:
            return pd.Series(dtype="int64", name="count")
        return df.groupby(pid_col).size().rename("count")

    visit_counts   = count_by_pid(visits_df)
    charge_counts  = count_by_pid(charges_df)
    txn_counts     = count_by_pid(transactions_df)
    stmt_counts    = count_by_pid(statements_df)

    # Call counts — AccountID = PatientID per DQ-003
    if rv_inbound_df is not None and len(rv_inbound_df) > 0:
        rvi_counts = rv_inbound_df.groupby(
            rv_inbound_df["AccountID"].astype(str).str.replace(r'\.0$', '', regex=True).astype("int64",errors="ignore")
        ).size().rename("count")
    else:
        rvi_counts = pd.Series(dtype="int64", name="count")

    if rv_outbound_df is not None and len(rv_outbound_df) > 0:
        rvo_col = rv_outbound_df["ACCOUNTID"].astype(str).str.replace(r'\.0$', '', regex=True)
        rvo_counts = rvo_col.groupby(rvo_col).size().rename("count")
    else:
        rvo_counts = pd.Series(dtype="int64", name="count")

    # Merge all into one frame
    df = pat_grp.copy()
    df = df.merge(visit_counts.rename("n_visits"),      on="PatientID", how="left")
    df = df.merge(charge_counts.rename("n_charges"),    on="PatientID", how="left")
    df = df.merge(txn_counts.rename("n_transactions"),  on="PatientID", how="left")
    df = df.merge(stmt_counts.rename("n_statements"),   on="PatientID", how="left")

    # Call counts need index reset
    rvi_df = rvi_counts.reset_index()
    rvi_df.columns = ["PatientID", "n_inbound_calls"]
    rvo_df = rvo_counts.reset_index()
    rvo_df.columns = ["PatientID", "n_outbound_calls"]

    df = df.merge(rvi_df, on="PatientID", how="left")
    df = df.merge(rvo_df, on="PatientID", how="left")

    # Fill nulls with 0 for counts
    count_cols = ["n_visits","n_charges","n_transactions","n_statements",
                  "n_inbound_calls","n_outbound_calls"]
    for col in count_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype("int64")

    # Derived columns
    df["is_multi_practice"]  = (df["n_source_databases"] > 1).astype(str)
    df["total_interactions"] = (
        df["n_visits"] + df["n_charges"] + df["n_transactions"] +
        df["n_statements"] + df["n_inbound_calls"] + df["n_outbound_calls"]
    ).astype("int64")
    df["has_called"] = (
        (df["n_inbound_calls"] > 0) | (df["n_outbound_calls"] > 0)
    ).astype(str)

    # Age: years from DOB to refresh date
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = ((pd.Timestamp(REFRESH_DATE) - df["dob"]).dt.days / 365.25).round(1)

    # Final column order per schema
    out = df[[
        "PatientID",
        "n_source_databases",
        "source_database_codes",
        "is_multi_practice",
        "city",
        "state",
        "age",
        "gender",
        "n_patient_rows",
        "n_visits",
        "n_charges",
        "n_transactions",
        "n_statements",
        "n_inbound_calls",
        "n_outbound_calls",
        "total_interactions",
        "has_called",
    ]].copy()

    # Cast types per schema
    out["PatientID"]           = out["PatientID"].astype("int64")
    out["n_source_databases"]  = out["n_source_databases"].astype("int64")
    out["n_patient_rows"]      = out["n_patient_rows"].astype("int64")
    out["n_visits"]            = out["n_visits"].astype("int64")
    out["n_charges"]           = out["n_charges"].astype("int64")
    out["n_transactions"]      = out["n_transactions"].astype("int64")
    out["n_statements"]        = out["n_statements"].astype("int64")
    out["n_inbound_calls"]     = out["n_inbound_calls"].astype("int64")
    out["n_outbound_calls"]    = out["n_outbound_calls"].astype("int64")
    out["total_interactions"]  = out["total_interactions"].astype("int64")
    out["age"]                 = out["age"].astype("float64")

    for col in ["source_database_codes","is_multi_practice","city","state",
                "gender","has_called"]:
        out[col] = out[col].astype(str)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT CALL BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_patient_call_bridge(
    patient_df: pd.DataFrame,
    rv_inbound_df: pd.DataFrame,
    rv_outbound_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cross-reference table linking AccountID from call tables to PatientID.
    5 columns per patient_call_bridge_schema.csv.

    AccountID = PatientID (DQ-003 confirmed).
    bridge_source: 'rv_inbound' or 'rv_outbound'
    confidence: 'confirmed' (direct match) or 'inferred' (probabilistic)

    One row per (AccountID, PatientID, bridge_source) combination.
    """
    rows = []

    # From rv_inbound: AccountID directly = PatientID
    if rv_inbound_df is not None and len(rv_inbound_df) > 0:
        for _, r in rv_inbound_df[["AccountID","FacilityCode"]].drop_duplicates().iterrows():
            account_id = str(r["AccountID"]).replace(".0","")
            try:
                pid = int(float(account_id))
            except (ValueError, TypeError):
                continue
            src = str(r.get("FacilityCode",""))

            rows.append({
                "AccountID":     pid,
                "PatientID":     pid,
                "bridge_source": "rv_inbound",
                "facility_code": src,
                "confidence":    "confirmed",
            })

    # From rv_outbound: ACCOUNTID directly = PatientID
    if rv_outbound_df is not None and len(rv_outbound_df) > 0:
        for _, r in rv_outbound_df[["ACCOUNTID","SERVICELOC"]].drop_duplicates().iterrows():
            account_id = str(r["ACCOUNTID"]).replace(".0","")
            try:
                pid = int(float(account_id))
            except (ValueError, TypeError):
                continue
            src = str(r.get("SERVICELOC",""))

            rows.append({
                "AccountID":     pid,
                "PatientID":     pid,
                "bridge_source": "rv_outbound",
                "facility_code": src,
                "confidence":    "confirmed",
            })

    # Add a small set of inferred matches (~36.1% of RV calls don't match
    # directly — they are outside the 12-month patient window, per DQ-003)
    patient_ids = patient_df["PatientID"].unique()
    n_inferred  = int(len(rows) * 0.361 / 0.639)   # scale to match prod ratio
    for _ in range(min(n_inferred, 200)):
        pid = int(rng.choice(patient_ids))
        src = str(rng.choice(["ACRB","CRC","SAPA","PMR","ESR"]))
        rows.append({
            "AccountID":     pid,
            "PatientID":     pid,
            "bridge_source": "inferred",
            "facility_code": src,
            "confidence":    "inferred",
        })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["AccountID","PatientID","bridge_source"]
    ).reset_index(drop=True)

    # Cast types per schema
    df["AccountID"] = df["AccountID"].astype("int64")
    df["PatientID"] = df["PatientID"].astype("int64")
    for col in ["bridge_source","facility_code","confidence"]:
        df[col] = df[col].astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path,
        patient_df: pd.DataFrame = None,
        visits_df: pd.DataFrame = None,
        charges_df: pd.DataFrame = None,
        transactions_df: pd.DataFrame = None,
        statements_df: pd.DataFrame = None,
        rv_inbound_df: pd.DataFrame = None,
        rv_outbound_df: pd.DataFrame = None) -> dict:

    def _load(path):
        return pd.read_parquet(path) if path.exists() else None

    dims  = out_dir / "02_dims"
    facts = out_dir / "01_facts"
    supp  = out_dir / "03_supplementary"
    supp.mkdir(parents=True, exist_ok=True)

    if patient_df      is None: patient_df      = _load(dims  / "patient.parquet")
    if visits_df       is None: visits_df        = _load(facts / "visits.parquet")
    if charges_df      is None: charges_df       = _load(facts / "charges.parquet")
    if transactions_df is None: transactions_df  = _load(facts / "transactions.parquet")
    if statements_df   is None: statements_df    = _load(facts / "statements.parquet")
    if rv_inbound_df   is None: rv_inbound_df    = _load(facts / "rv_inbound.parquet")
    if rv_outbound_df  is None: rv_outbound_df   = _load(facts / "rv_outbound.parquet")

    print("Generating master_patient_index.parquet ...")
    mpi_df = generate_master_patient_index(
        patient_df, visits_df, charges_df, transactions_df,
        statements_df, rv_inbound_df, rv_outbound_df
    )
    mpi_df.to_parquet(supp / "master_patient_index.parquet", index=False)
    print(f"  ✓ {len(mpi_df):,} rows → {supp / 'master_patient_index.parquet'}")

    print("Generating patient_call_bridge.parquet ...")
    pcb_df = generate_patient_call_bridge(
        patient_df, rv_inbound_df, rv_outbound_df
    )
    pcb_df.to_parquet(supp / "patient_call_bridge.parquet", index=False)
    print(f"  ✓ {len(pcb_df):,} rows → {supp / 'patient_call_bridge.parquet'}")

    return {"master_patient_index": mpi_df, "patient_call_bridge": pcb_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output")
    args = parser.parse_args()
    run(Path(args.out))