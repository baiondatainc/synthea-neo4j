"""
gen_calls.py
────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates call-centre fact tables:

  • 01_facts/ringcentral.parquet    (agent call records, 1 row per Contact_ID)
  • 01_facts/rv_inbound.parquet     (IVR inbound, 1 row per ResponseID)
  • 01_facts/rv_outbound.parquet    (dialler outbound, 1 row per ACCOUNT)

Relationship chain:
  ringcentral.ANI_DIALNUM_norm → phone_bridge.phone_norm → patient.PatientID
  rv_inbound.AccountID         = patient.PatientID (confirmed DQ-003)
  rv_outbound.ACCOUNTID        = patient.PatientID (confirmed DQ-003)

Known data quality issues replicated:
  DQ-002: ringcentral.start_time is broken — collapsed to 2026-05-04
          (real field to use is Start_Date / Start_Date_clean)
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from constants import (
    SEED, REFRESH_DATE, WINDOW_START, PRACTICE_CODES, PRACTICE_STATES,
    CAMPAIGN_NAMES, SKILL_NAMES, TEAM_NAMES, AGENT_NAMES,
    DISP_NAMES, IVR_TYPES, RESULT_DESCS, STATE_AREA_CODES,
)
from helpers import (
    rng, pick_date_in_window, normalise_phone, fmt_phone, composite_key,
)

BROKEN_START_TIME = datetime(2026, 5, 4, 7, 59, 20)  # DQ-002 collapsed value


# ─────────────────────────────────────────────────────────────────────────────
# RINGCENTRAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_ringcentral(patient_df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    """
    Agent call records.
    ANI_DIALNUM_norm links to phone_bridge for patient attribution.
    ~6.5% of patients have RC call exposure (222k / 3.29M in prod).
    """
    rows = []

    # Pick a subset of patients to have calls attributed
    n_attributed = min(n, int(len(patient_df) * 0.065))
    attributed_pats = patient_df.sample(n=n_attributed, random_state=SEED)

    contact_counter = 593000000000

    for i in range(n):
        contact_id = str(contact_counter + i)
        src        = str(rng.choice(PRACTICE_CODES))
        campaign   = str(rng.choice(CAMPAIGN_NAMES))
        skill      = str(rng.choice(SKILL_NAMES))
        team       = str(rng.choice(TEAM_NAMES))
        agent      = str(rng.choice(AGENT_NAMES))

        # ~65% of calls are attributed to a known patient phone
        if i < n_attributed:
            pat_row = attributed_pats.iloc[i % len(attributed_pats)]
            state   = PRACTICE_STATES.get(str(pat_row["Source_Database_Code"]), "TN")
            ani     = pat_row.get("PatientPhone_norm") or pat_row.get("PatientPhone")
            if ani is None:
                area = str(rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES["TN"])))
                ani  = str(int(rng.integers(2000000000, 9999999999)))
            ani_norm = normalise_phone(str(ani)) if ani else None
            junk = False
        else:
            area    = str(rng.choice(["615","713","404","614","305","702"]))
            ani     = str(int(rng.integers(2000000000, 9999999999)))
            ani_norm = ani[-10:] if len(ani) >= 10 else None
            junk    = rng.random() < 0.02

        # Call timing
        call_date   = pick_date_in_window()
        start_date  = call_date.strftime("%Y-%m-%d")
        pre_queue   = int(rng.integers(0, 300))
        in_queue    = int(rng.integers(0, 30))
        agent_time  = int(rng.integers(60, 1200))
        post_queue  = 0
        acw         = int(rng.integers(0, 120))
        total_time  = pre_queue + in_queue + agent_time + acw
        abandon_t   = int(rng.integers(0, pre_queue)) if rng.random() < 0.1 else 0
        abandon     = "Y" if abandon_t > 0 else "N"
        sla         = 1 if in_queue <= 20 else 0

        disp     = str(rng.choice(DISP_NAMES))
        tag      = str(rng.choice(["First Time Call","Repeat Caller","Escalation",""])) if rng.random() > 0.113 else None

        rows.append({
            "Contact_ID":                  contact_id,
            "Master_Contact_ID":           contact_id,
            "Contact_Code":                str(int(rng.integers(10000000, 99999999))),
            "Media_Name":                  "Phone Call",
            "Contact_Name":                str(int(rng.integers(8000000000, 8999999999))),
            "ANI_DIALNUM":                 ani[-10:] if ani else "0000000000",
            "Skill_No":                    str(int(rng.integers(1000000, 9999999))),
            "Skill_Name":                  skill,
            "Campaign_No":                 str(int(rng.integers(1000000, 9999999))),
            "Campaign_Name":               campaign,
            "Agent_No":                    str(int(rng.integers(10000000, 99999999))),
            "Agent_Name":                  agent,
            "Team_No":                     str(int(rng.integers(1000000, 9999999))),
            "Team_Name":                   team,
            "SLA":                         sla,
            "Start_Date":                  start_date,
            "start_time":                  BROKEN_START_TIME,    # DQ-002: intentionally broken
            "PreQueue":                    pre_queue,
            "InQueue":                     in_queue,
            "Agent_Time":                  agent_time,
            "PostQueue":                   post_queue,
            "ACW_Time":                    acw,
            "Total_Time_Plus_Disposition": total_time,
            "Abandon_Time":                abandon_t,
            "Routing_Time":                int(rng.integers(0, 10)),
            "Abandon":                     abandon,
            "Callback_Time":               0,
            "Logged":                      "N",
            "Hold_Time":                   int(rng.integers(0, 60)),
            "Disp_Code":                   str(int(rng.integers(1000, 9999))),
            "Disp_Name":                   disp,
            "Disp_Comments":               None,
            "Tags":                        tag,
            "Start_Date_clean":            call_date.replace(hour=0, minute=0, second=0, microsecond=0),
            "ANI_DIALNUM_norm":            ani_norm,
            "_phone_is_junk":              junk,
            "_phone_is_dialler":           rng.random() < 0.02,
            "_rc_attributable":            not junk and ani_norm is not None,
        })

    df = pd.DataFrame(rows)
    df["Contact_ID"]      = df["Contact_ID"].astype("string")
    df["Master_Contact_ID"] = df["Master_Contact_ID"].astype("string")
    df["Contact_Code"]    = df["Contact_Code"].astype("string")
    df["Skill_No"]        = df["Skill_No"].astype("string")
    df["Campaign_No"]     = df["Campaign_No"].astype("string")
    df["Agent_No"]        = df["Agent_No"].astype("string")
    df["Team_No"]         = df["Team_No"].astype("string")
    df["Disp_Code"]       = df["Disp_Code"].astype("string")
    df["start_time"]      = pd.to_datetime(df["start_time"])
    df["Start_Date_clean"] = pd.to_datetime(df["Start_Date_clean"])
    df["SLA"]             = df["SLA"].astype("int64")
    df["PreQueue"]        = df["PreQueue"].astype("int64")
    df["InQueue"]         = df["InQueue"].astype("int64")
    df["Agent_Time"]      = df["Agent_Time"].astype("int64")
    df["PostQueue"]       = df["PostQueue"].astype("int64")
    df["ACW_Time"]        = df["ACW_Time"].astype("int64")
    df["Total_Time_Plus_Disposition"] = df["Total_Time_Plus_Disposition"].astype("int64")
    df["Abandon_Time"]    = df["Abandon_Time"].astype("int64")
    df["Routing_Time"]    = df["Routing_Time"].astype("int64")
    df["Callback_Time"]   = df["Callback_Time"].astype("int64")
    df["Hold_Time"]       = df["Hold_Time"].astype("int64")
    df["_phone_is_junk"]  = df["_phone_is_junk"].astype(bool)
    df["_phone_is_dialler"] = df["_phone_is_dialler"].astype(bool)
    df["_rc_attributable"]  = df["_rc_attributable"].astype(bool)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RV INBOUND (IVR)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rv_inbound(patient_df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """
    IVR inbound call records.
    AccountID = PatientID (DQ-003 confirmed).
    ~52.9% have payment made (AmountPaid not null).
    """
    rows = []
    resp_counter = 900000

    # Sample patients who made inbound calls
    callers = patient_df.sample(n=min(n, len(patient_df)), random_state=SEED + 1)

    for i in range(n):
        pat_row   = callers.iloc[i % len(callers)]
        pid       = int(pat_row["PatientID"])
        src       = str(pat_row["Source_Database_Code"])
        state     = PRACTICE_STATES.get(src, "TN")

        resp_id   = str(resp_counter + i)
        area      = str(rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES["TN"])))
        caller_id = f"1{area}{int(rng.integers(2000000, 9999999))}"

        balance   = round(float(rng.uniform(10, 3000)), 2)
        made_pay  = rng.random() > 0.529
        amt_paid  = round(float(rng.uniform(10, balance)), 2) if made_pay else None
        auth_ok   = made_pay or rng.random() > 0.3

        call_dt   = pick_date_in_window()
        duration  = round(float(rng.uniform(1.0, 15.0)), 1)
        transfer  = round(float(rng.uniform(0, duration * 0.5)), 1)

        first = str(pat_row.get("PatientFirstName", "UNKNOWN"))
        last  = str(pat_row.get("PatientLastName",  "UNKNOWN"))

        rows.append({
            "ResponseID":           resp_id,
            "CallerID":             caller_id,
            "AccountID":            str(pid),
            "Comment":              f"AccountNumber={pid}&DateOfBirth={call_dt.strftime('%m%d%Y')}",
            "PatLastName":          last,
            "PatFirstName":         first,
            "Balance":              balance,
            "FacilityCode":         src,
            "BusinessUnit":         f"{src} Radiology {src} 555-555-5555",
            "IVR":                  str(rng.choice(IVR_TYPES)),
            "AmountPaid":           amt_paid,
            "AuthenticationSuccess": auth_ok,
            "ResultDesc":           "Approved" if made_pay else "No Payment",
            "TransactionTypeDesc":  "Sale" if made_pay else None,
            "CallDateTime":         call_dt,
            "CallDuration":         duration,
            "TransferDuration":     transfer,
            # norm column
            "CallerID_norm":        caller_id[-10:] if len(caller_id) >= 10 else caller_id,
        })

    df = pd.DataFrame(rows)
    df["ResponseID"]    = df["ResponseID"].astype("string")
    df["AccountID"]     = df["AccountID"].astype("string")
    df["Balance"]       = df["Balance"].astype("float64")
    df["AmountPaid"]    = df["AmountPaid"].astype("float64")
    df["CallDateTime"]  = pd.to_datetime(df["CallDateTime"])
    df["AuthenticationSuccess"] = df["AuthenticationSuccess"].astype(bool)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RV OUTBOUND (dialler)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rv_outbound(patient_df: pd.DataFrame, n: int = 150) -> pd.DataFrame:
    """
    Dialler outbound call records.
    ACCOUNTID = PatientID.
    20.8% null CALLDATETIME in prod — replicated.
    """
    rows = []

    outbound_pats = patient_df.sample(n=min(n, len(patient_df)), random_state=SEED + 2)

    for i in range(n):
        pat_row  = outbound_pats.iloc[i % len(outbound_pats)]
        pid      = int(pat_row["PatientID"])
        src      = str(pat_row["Source_Database_Code"])
        state    = PRACTICE_STATES.get(src, "TN")

        area     = str(rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES["TN"])))
        pat_phone = int(rng.integers(2000000000, 9999999999)) if rng.random() > 0.013 else None
        resp_phone = int(rng.integers(2000000000, 9999999999)) if rng.random() > 0.015 else None

        balance  = round(float(rng.uniform(10, 5000)), 2)
        result   = str(rng.choice(RESULT_DESCS))
        account  = f"CRC{pid}"

        call_dt  = pick_date_in_window() if rng.random() > 0.208 else None

        pat_norm  = str(pat_phone)[-10:] if pat_phone else None
        resp_norm = str(resp_phone)[-10:] if resp_phone else None

        first = str(pat_row.get("PatientFirstName", "UNKNOWN"))
        last  = str(pat_row.get("PatientLastName",  "UNKNOWN"))

        rows.append({
            "ACCOUNT":           account,
            "RESPLASTNAME":      last  if rng.random() > 0.013 else None,
            "RESPFIRSTNAME":     first if rng.random() > 0.013 else None,
            "PATIENTPHONE":      float(pat_phone) if pat_phone else None,
            "RESPPHONE":         float(resp_phone) if resp_phone else None,
            "PATIENTBALANCE":    balance if rng.random() > 0.013 else None,
            "CALLDATETIME":      call_dt,
            "RESULTSDESC":       result,
            "ACCOUNTID":         str(float(pid)),
            "SERVICELOC":        src[:3],
            "PATIENTPHONE_norm": pat_norm,
            "RESPPHONE_norm":    resp_norm if resp_norm else str(int(rng.integers(7000000000, 8999999999))),
        })

    df = pd.DataFrame(rows)
    df["PATIENTPHONE"]  = df["PATIENTPHONE"].astype("float64")
    df["RESPPHONE"]     = df["RESPPHONE"].astype("float64")
    df["PATIENTBALANCE"] = df["PATIENTBALANCE"].astype("float64")
    df["CALLDATETIME"]  = pd.to_datetime(df["CALLDATETIME"])
    df["ACCOUNTID"]     = df["ACCOUNTID"].astype("string")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path, patient_df: pd.DataFrame = None) -> dict:
    if patient_df is None:
        patient_df = pd.read_parquet(out_dir / "02_dims" / "patient.parquet")

    facts = out_dir / "01_facts"
    facts.mkdir(parents=True, exist_ok=True)

    print("Generating ringcentral.parquet ...")
    rc_df = generate_ringcentral(patient_df, n=min(5000, len(patient_df) * 3))
    rc_df.to_parquet(facts / "ringcentral.parquet", index=False)
    print(f"  ✓ {len(rc_df):,} rows → {facts / 'ringcentral.parquet'}")

    print("Generating rv_inbound.parquet ...")
    rvi_df = generate_rv_inbound(patient_df, n=min(400, len(patient_df)))
    rvi_df.to_parquet(facts / "rv_inbound.parquet", index=False)
    print(f"  ✓ {len(rvi_df):,} rows → {facts / 'rv_inbound.parquet'}")

    print("Generating rv_outbound.parquet ...")
    rvo_df = generate_rv_outbound(patient_df, n=min(150, len(patient_df)))
    rvo_df.to_parquet(facts / "rv_outbound.parquet", index=False)
    print(f"  ✓ {len(rvo_df):,} rows → {facts / 'rv_outbound.parquet'}")

    return {"ringcentral": rc_df, "rv_inbound": rvi_df, "rv_outbound": rvo_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output")
    args = parser.parse_args()
    run(Path(args.out))
