"""
gen_supplementary.py
────────────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates supplementary / reference tables:

  • 03_supplementary/phone_bridge.parquet   (patient-phone × RC attribution)
  • 03_supplementary/campaign_map.parquet   (45-row practice → campaign reference)

Relationship chain:
  phone_bridge.Source_Database_Code + PatientID → patient.PatientID (FK)
  phone_bridge.phone_norm  → ringcentral.ANI_DIALNUM_norm  (attribution join)
  campaign_map.Source_Database_Code → practice  (lookup reference)

phone_bridge is the cross-join that enables patient attribution of RC calls.
Every phone number associated with a patient (PatientPhone, PatientCellPhone,
ResponsiblePartyPhone, ResponsiblePartyCellPhone) gets one row per phone type.
~6.5% of patients have RC calls attributed (rc_call_count not null).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    SEED, PRACTICE_CODES, CAMPAIGN_NAMES,
)
from helpers import rng, normalise_phone


# ─────────────────────────────────────────────────────────────────────────────
# PHONE BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

def generate_phone_bridge(patient_df: pd.DataFrame,
                           ringcentral_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (patient, phone_number, phone_type).
    Patient can have up to 4 phone types:
      patient / cell / responsible_party / responsible_party_cell

    RC attribution:
      For each phone, look up whether ANI_DIALNUM_norm in RC matches.
      ~6.5% match rate overall (mirrors prod 222k / 3.29M).
    """
    # Build RC phone → call count / campaign lookup
    rc_by_phone: dict = {}
    for _, rc in ringcentral_df.iterrows():
        ani = rc.get("ANI_DIALNUM_norm")
        if ani and not pd.isna(ani) and rc.get("_rc_attributable"):
            if ani not in rc_by_phone:
                rc_by_phone[ani] = {"call_count": 0, "campaigns": set()}
            rc_by_phone[ani]["call_count"] += 1
            cname = rc.get("Campaign_Name")
            if cname:
                rc_by_phone[ani]["campaigns"].add(str(cname))

    rows = []

    phone_types = [
        ("PatientPhone_norm",             "patient"),
        ("PatientCellPhone_norm",         "cell"),
        ("ResponsiblePartyPhone_norm",    "responsible_party"),
        ("ResponsiblePartyCellPhone_norm","responsible_party_cell"),
    ]

    for _, pat in patient_df.iterrows():
        pid = int(pat["PatientID"])
        src = str(pat["Source_Database_Code"])

        for norm_col, ptype in phone_types:
            phone_val = pat.get(norm_col)
            if phone_val is None or pd.isna(phone_val):
                continue
            phone_norm = str(phone_val)
            if len(phone_norm) < 7:
                continue

            rc_info = rc_by_phone.get(phone_norm)
            if rc_info:
                rc_calls   = float(rc_info["call_count"])
                campaigns  = sorted(rc_info["campaigns"])
                camp_count = float(len(campaigns))
                camp_str   = ", ".join(campaigns) if campaigns else None
                primary    = campaigns[0] if campaigns else None
            else:
                rc_calls   = None
                camp_count = None
                camp_str   = None
                primary    = None

            rows.append({
                "Source_Database_Code": src,
                "PatientID":            pid,
                "phone_norm":           phone_norm,
                "phone_type":           ptype,
                "rc_call_count":        rc_calls,
                "campaign_count":       camp_count,
                "campaigns_contacted":  camp_str,
                "primary_campaign":     primary,
            })

    df = pd.DataFrame(rows)
    df["PatientID"]       = df["PatientID"].astype("Int64")
    df["rc_call_count"]   = df["rc_call_count"].astype("float64")
    df["campaign_count"]  = df["campaign_count"].astype("float64")
    df["primary_campaign"] = df["primary_campaign"].astype("string")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CAMPAIGN MAP
# ─────────────────────────────────────────────────────────────────────────────

def generate_campaign_map() -> pd.DataFrame:
    """
    45-row reference table: Campaign_Name → Source_Database_Code.
    8.9% null Source_Database_Code (non-Imagine databases).
    Matches prod grain exactly.
    """
    rows = []

    for i, campaign in enumerate(CAMPAIGN_NAMES):
        # ~8.9% of campaigns are non-Imagine (no Source_Database_Code)
        src = str(rng.choice(PRACTICE_CODES)) if rng.random() > 0.089 else None
        note = "Non-Imagine database" if src is None else None
        rows.append({
            "Campaign Name":       campaign,
            "Source Database Code": src,
            "Notes":               note,
        })

    # Pad to 45 rows if needed
    while len(rows) < 45:
        rows.append({
            "Campaign Name":       f"CAMP{len(rows)+1:03d}",
            "Source Database Code": str(rng.choice(PRACTICE_CODES)),
            "Notes":               None,
        })

    df = pd.DataFrame(rows[:45])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path,
        patient_df: pd.DataFrame = None,
        ringcentral_df: pd.DataFrame = None) -> dict:

    if patient_df is None:
        patient_df = pd.read_parquet(out_dir / "02_dims" / "patient.parquet")
    if ringcentral_df is None:
        rc_path = out_dir / "01_facts" / "ringcentral.parquet"
        if rc_path.exists():
            ringcentral_df = pd.read_parquet(rc_path)
        else:
            from gen_calls import generate_ringcentral
            ringcentral_df = generate_ringcentral(patient_df)

    supp = out_dir / "03_supplementary"
    supp.mkdir(parents=True, exist_ok=True)

    print("Generating phone_bridge.parquet ...")
    pb_df = generate_phone_bridge(patient_df, ringcentral_df)
    pb_df.to_parquet(supp / "phone_bridge.parquet", index=False)
    print(f"  ✓ {len(pb_df):,} rows → {supp / 'phone_bridge.parquet'}")

    print("Generating campaign_map.parquet ...")
    cm_df = generate_campaign_map()
    cm_df.to_parquet(supp / "campaign_map.parquet", index=False)
    print(f"  ✓ {len(cm_df):,} rows → {supp / 'campaign_map.parquet'}")

    return {"phone_bridge": pb_df, "campaign_map": cm_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output")
    args = parser.parse_args()
    run(Path(args.out))
