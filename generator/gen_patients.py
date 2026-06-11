"""
gen_patients.py
───────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates:
  • 02_dims/patient.parquet              (demographics, one row per patient-practice)
  • 00_navigation/patient_navigation_map.parquet  (144-col master, enriched)

Relationship chain:
  patient.PatientID               ← primary key for all fact tables
  patient.Source_Database_Code    ← links to location, charges, visits, transactions
  nav map is a superset of patient — adds cohort flags, financial rollups,
  call tallies, location enrichment, and campaign exposure.

Multi-practice patients:
  ~15.8% of unique humans appear under 2+ practices (520k / 3.29M in prod).
  We simulate this by assigning a second Source_Database_Code to some patients.

Run standalone:
    python gen_patients.py --out ./output --n 5000
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from constants import (
    SEED, N_PATIENTS, PRACTICE_CODES, PRACTICE_STATES, MULTI_PRACTICE_RATE,
    CATASTROPHE_RATE, SELF_PAY_RATE, REFRESH_DATE, WINDOW_START, WINDOW_END,
    CARRIERS, CARRIER_WEIGHTS, PLAN_TYPES,
)
from helpers import (
    rng,
    pick_gender, pick_name, pick_suffix, pick_dob,
    pick_race, pick_ethnicity, pick_propensity, pick_ssn,
    pick_address, bad_address_indicator,
    pick_phone, pick_cell_phone, pick_email,
    normalise_phone, is_institutional_phone,
    composite_key, hashed_patient_id,
    pick_date_in_window, pick_date_historical,
    assign_payor_cohort, assign_call_tier, assign_cohort,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core patient record builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_patient_row(patient_id: int, src: str) -> dict:
    """Build one patient-practice demographics row (42 columns = patient.parquet)."""
    state  = PRACTICE_STATES.get(src, "TN")
    gender = pick_gender()
    first, middle, last = pick_name(gender)

    addr, addr2, city, _, zip_ = pick_address(state)
    rp_addr, rp_addr2, rp_city, _, rp_zip = pick_address(state)  # responsible party may differ

    dob      = pick_dob()
    rp_dob   = dob if rng.random() < 0.7 else pick_dob()   # often same person

    phone     = pick_phone(state)
    cell      = pick_cell_phone(state)
    rp_phone  = phone if rng.random() < 0.6 else pick_phone(state)
    rp_cell   = pick_cell_phone(state)
    email     = pick_email(first, last)

    phone_norm    = normalise_phone(phone)
    cell_norm     = normalise_phone(cell)
    rp_phone_norm = normalise_phone(rp_phone)
    rp_cell_norm  = normalise_phone(rp_cell)

    grade, desc = pick_propensity()
    race        = pick_race()
    ethnicity   = pick_ethnicity()
    ssn         = pick_ssn()
    rp_ssn      = ssn if rng.random() < 0.6 else pick_ssn()

    institutional = is_institutional_phone(phone_norm)

    return {
        "Source_Database_Code":                src,
        "PatientID":                           patient_id,
        "PatientFirstName":                    first,
        "PatientMiddleName":                   middle,
        "PatientLastName":                     last,
        "PatientAddress":                      addr if rng.random() > 0.004 else None,
        "PatientAddress2":                     addr2,
        "PatientCity":                         city if rng.random() > 0.004 else None,
        "PatientState":                        state if rng.random() > 0.004 else None,
        "PatientZip":                          zip_ if rng.random() > 0.005 else None,
        "PatientDOB":                          dob,
        "PatientPhone":                        phone,
        "PatientGender":                       gender,
        "ResponsiblePartyFirstName":           first,
        "ResponsiblePartyMiddleName":          middle,
        "ResponsiblePartyLastName":            last,
        "ResponsiblePartyAddress":             rp_addr if rng.random() > 0.004 else None,
        "ResponsiblePartyAddress2":            rp_addr2,
        "ResponsiblePartyCity":                rp_city if rng.random() > 0.004 else None,
        "ResponsiblePartyState":               state if rng.random() > 0.004 else None,
        "ResponsiblePartyZip":                 rp_zip if rng.random() > 0.005 else None,
        "ResponsiblePartyPhone":               rp_phone if rng.random() > 0.056 else None,
        "PatientSSN":                          ssn,
        "PatientSuffix":                       None,     # 100% null in prod
        "PatientCellPhone":                    cell,
        "ResponsiblePartySSN":                 rp_ssn,
        "ResponsiblePartySuffix":              None,     # 100% null in prod
        "ResponsiblePartyCellPhone":           rp_cell if rng.random() > 0.647 else None,
        "ResponsiblePartyDOB":                 rp_dob if rng.random() > 0.041 else None,
        "Imagine_Propensity_to_Pay_Grade":     grade,
        "Imagine_Propensity_to_Pay_Description": desc,
        "Bad_Address_Indicator":               bad_address_indicator(),
        "PatientRace":                         race,
        "PatientEthnicity":                    ethnicity,
        "patEmail":                            email,
        "tbl_Refresh_Date":                    REFRESH_DATE,
        "PatientPhone_norm":                   phone_norm,
        "ResponsiblePartyPhone_norm":          rp_phone_norm,
        "PatientCellPhone_norm":               cell_norm,
        "ResponsiblePartyCellPhone_norm":      rp_cell_norm,
        "_phone_institutional_flag":           institutional,
        "_pk_composite":                       composite_key(src, patient_id),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Financial rollup builder (for nav map)
# ─────────────────────────────────────────────────────────────────────────────

def _build_financial_rollup(patient_id: int, src: str, location_df: pd.DataFrame) -> dict:
    """Simulate per-patient financial summary for the nav map."""
    # Visit summary
    visit_count  = int(rng.integers(1, 12))
    first_visit  = pick_date_historical() if rng.random() > 0.2 else pick_date_in_window()
    last_visit   = pick_date_in_window()
    if last_visit < first_visit:
        first_visit, last_visit = last_visit, first_visit

    in_window = last_visit >= WINDOW_START

    # Location assignment
    loc_rows = location_df[location_df["Source_Database_Code"] == src]
    if len(loc_rows) == 0:
        loc_rows = location_df
    loc_row = loc_rows.iloc[int(rng.integers(0, len(loc_rows)))]
    last_location_id = loc_row["LocationID"]

    # Financial
    charge_count     = visit_count * int(rng.integers(1, 4))
    total_charged    = round(float(sum(float(rng.lognormal(6.2, 0.8)) for _ in range(charge_count))), 2)
    iw_charged       = round(total_charged * float(rng.uniform(0.4, 1.0)), 2) if in_window else 0.0
    total_paid       = round(total_charged * float(rng.uniform(0.05, 0.7)), 2)
    in_window_paid   = round(total_paid * float(rng.uniform(0.3, 1.0)), 2) if in_window else 0.0

    adj_contractual  = round(total_charged * float(rng.uniform(0.2, 0.6)), 2) if rng.random() > 0.243 else None
    adj_bad_debt     = round(total_charged * float(rng.uniform(0.0, 0.4)), 2) if rng.random() > 0.299 else None
    adj_collection   = round(float(rng.uniform(0, 1000)), 2) if rng.random() > 0.934 else None
    adj_charity      = None   # 100% null in prod
    adj_refund       = round(float(rng.uniform(-500, 0)), 2) if rng.random() > 0.997 else None
    adj_payment_plan = 0.0
    adj_other        = round(float(rng.uniform(0, 500)), 2) if rng.random() > 0.387 else None

    total_adjusted = sum(x for x in [adj_contractual, adj_bad_debt, adj_collection,
                                      adj_other, adj_payment_plan] if x is not None)
    outstanding    = max(0.0, round(total_charged - total_paid - total_adjusted, 2))

    # Statements
    stmt_count   = int(rng.integers(0, 8))
    first_stmt   = pick_date_in_window() if stmt_count > 0 else None
    last_stmt    = pick_date_in_window() if stmt_count > 0 else None
    if first_stmt and last_stmt and last_stmt < first_stmt:
        first_stmt, last_stmt = last_stmt, first_stmt
    iw_stmt_count = float(rng.integers(0, stmt_count + 1)) if stmt_count > 0 and rng.random() > 0.144 else None

    first_email  = pick_date_in_window() if rng.random() > 0.854 else None
    first_text   = pick_date_in_window() if rng.random() > 0.443 else None

    # Calls
    rv_in   = float(rng.choice([0,1,2,3,4,5], p=[0.70,0.15,0.07,0.04,0.02,0.02]))
    rv_out  = float(rng.choice([0,1,2,3,4,5], p=[0.72,0.14,0.06,0.04,0.02,0.02]))
    rc_call = float(rng.choice([0,1,2,3,4,5,6,7], p=[0.65,0.15,0.08,0.05,0.03,0.02,0.01,0.01]))
    total_calls = rv_in + rv_out + rc_call

    # Insurance (38.8% have insurance on record)
    has_ins = rng.random() < 0.388
    if has_ins:
        carrier = str(rng.choice(CARRIERS, p=CARRIER_WEIGHTS))
        ptype   = str(rng.choice(PLAN_TYPES))
        plan_name = f"{carrier} {ptype}"
    else:
        carrier, ptype, plan_name = None, None, None

    # Cohort flags
    payor_cohort, is_sp, is_bai, is_fc = assign_payor_cohort(total_calls, has_ins)
    call_tier   = assign_call_tier(total_calls)
    cohort      = assign_cohort(call_tier)
    is_cat      = cohort == "catastrophe"
    is_friction = cohort == "friction"
    is_clean    = cohort == "clean"

    # Geographic flags
    state = PRACTICE_STATES.get(src, "TN")
    is_tn   = state == "TN"
    is_sapa = src == "SAPA"
    is_nraa = src == "NRAA"
    is_atl  = src in ("GRH", "RADI", "RADK", "RADP") and state == "GA"

    # Birdeye at primary location
    has_birdeye = loc_row["birdeye_review_count"] is not None and not pd.isna(loc_row.get("birdeye_review_count", None))

    # RC / campaign attribution
    rc_attr = float(rc_call) if rng.random() > 0.065 else None
    has_campaign = rng.random() < 0.068
    campaign_name = str(rng.choice(list({
        "SAPA":"SAPA","PMR":"PMR","ACRB":"ACRB","NRAA":"NRAA","GSIA":"GSIA"
    }.get(src, "NRA")))) if has_campaign else None
    campaign_count = 1.0 if has_campaign else None

    return {
        "visit_count":                     float(visit_count),
        "first_visit_date":                first_visit,
        "last_visit_date":                 last_visit,
        "last_location_id":                str(last_location_id),
        "charge_count":                    charge_count,
        "total_charged":                   total_charged,
        "in_window_charge_count":          float(charge_count) if in_window and rng.random() > 0.233 else None,
        "in_window_charged":               iw_charged,
        "total_paid":                      total_paid,
        "transaction_count":               float(charge_count + int(rng.integers(0, 3))) if rng.random() > 0.003 else None,
        "in_window_paid":                  in_window_paid,
        "adj_contractual":                 adj_contractual,
        "adj_bad_debt":                    adj_bad_debt,
        "adj_collection_agency":           adj_collection,
        "adj_charity_care":                adj_charity,
        "adj_refund_reversal":             adj_refund,
        "adj_payment_plan":                adj_payment_plan,
        "adj_other":                       adj_other,
        "statement_count":                 stmt_count,
        "first_statement_date":            first_stmt if first_stmt else REFRESH_DATE,
        "last_statement_date":             last_stmt if last_stmt else REFRESH_DATE,
        "first_email_sent":                first_email,
        "first_text_sent":                 first_text,
        "in_window_statement_count":       iw_stmt_count,
        "rv_in_calls_window":              rv_in,
        "rv_out_calls_window":             rv_out,
        "rc_calls_window":                 rc_call,
        "PlanName":                        plan_name,
        "Carrier_Name":                    carrier,
        "PlanType":                        ptype,
        "total_calls_window":              total_calls,
        "total_adjusted":                  total_adjusted,
        "outstanding_balance":             outstanding,
        "_active_window":                  in_window,
        "_cohort":                         cohort,
        "payor_cohort":                    payor_cohort,
        "is_self_pay":                     is_sp,
        "is_bai":                          is_bai,
        "is_fully_covered":                is_fc,
        "call_tier":                       call_tier,
        "is_catastrophe":                  is_cat,
        "is_friction":                     is_friction,
        "is_clean":                        is_clean,
        "is_sapa":                         is_sapa,
        "is_nraa":                         is_nraa,
        "is_tennessee":                    is_tn,
        "is_atlanta_404":                  is_atl,
        "_identity_key":                   str(patient_id),
        "practice_count":                  None,   # filled in post-processing
        "multi_practice_flag":             None,   # filled in post-processing
        "has_visits":                      visit_count > 0,
        "has_charges":                     charge_count > 0,
        "has_transactions":                total_paid > 0,
        "has_statements":                  stmt_count > 0,
        "has_inbound_calls":               rv_in > 0,
        "has_outbound_calls":              rv_out > 0,
        "has_ringcentral":                 rc_call > 0,
        "has_any_calls":                   total_calls > 0,
        "has_insurance":                   has_ins,
        "has_email":                       False,   # filled from patient row
        "has_phone":                       False,   # filled from patient row
        "has_in_window_activity":          in_window,
        "hashed_patient_id":               hashed_patient_id(src, patient_id),
        # Location enrichment cols
        "primary_location_name":           str(loc_row.get("LocationName", "")),
        "primary_location_abbr":           str(loc_row.get("LocationAbbreviation", "")),
        "primary_location_city":           str(loc_row.get("LocationCity", "")) if pd.notna(loc_row.get("LocationCity")) else None,
        "primary_location_state":          str(loc_row.get("LocationState", "")) if pd.notna(loc_row.get("LocationState")) else None,
        "primary_location_zip":            str(loc_row.get("LocationZip", "")) if pd.notna(loc_row.get("LocationZip")) else None,
        "primary_location_npi":            str(loc_row.get("LocationNPINumber", "")) if pd.notna(loc_row.get("LocationNPINumber")) else None,
        "primary_location_type":           str(loc_row.get("LocationType", "")),
        "has_birdeye_at_primary_location": has_birdeye,
        "birdeye_top_source":              str(rng.choice(["Google","Yelp"], p=[0.85,0.15])) if has_birdeye else None,
        # Birdeye rollup cols (from location)
        "primary_location_birdeye_reviews":             loc_row.get("birdeye_review_count"),
        "primary_location_birdeye_rating":              loc_row.get("birdeye_avg_rating"),
        "primary_location_birdeye_median_rating":       loc_row.get("birdeye_median_rating"),
        "primary_location_birdeye_one_star":            loc_row.get("birdeye_one_star_count"),
        "primary_location_birdeye_two_star":            loc_row.get("birdeye_two_star_count"),
        "primary_location_birdeye_three_star":          loc_row.get("birdeye_three_star_count"),
        "primary_location_birdeye_four_star":           loc_row.get("birdeye_four_star_count"),
        "primary_location_birdeye_five_star":           loc_row.get("birdeye_five_star_count"),
        "primary_location_birdeye_last_review":         loc_row.get("birdeye_last_review_date"),
        "primary_location_birdeye_first_review":        loc_row.get("birdeye_first_review_date"),
        "primary_location_birdeye_reviews_with_comment":loc_row.get("birdeye_review_with_comment_count"),
        "primary_location_birdeye_phi_reviews":         loc_row.get("birdeye_phi_review_count"),
        "primary_location_birdeye_phi_phones":          loc_row.get("birdeye_phi_phone_total"),
        "primary_location_birdeye_phi_emails":          loc_row.get("birdeye_phi_email_total"),
        "primary_location_birdeye_phi_ssns":            loc_row.get("birdeye_phi_ssn_total"),
        "primary_location_birdeye_one_star_pct":        loc_row.get("birdeye_one_star_pct"),
        "primary_location_birdeye_one_or_two_star_pct": loc_row.get("birdeye_one_or_two_star_pct"),
        # Birdeye summary cols on nav map
        "birdeye_review_with_comment_count_loc":        loc_row.get("birdeye_review_with_comment_count"),
        "birdeye_one_or_two_star_pct_loc":              loc_row.get("birdeye_one_or_two_star_pct"),
        "birdeye_phi_review_count_loc":                 loc_row.get("birdeye_phi_review_count"),
        "birdeye_phi_phone_total_loc":                  loc_row.get("birdeye_phi_phone_total"),
        "birdeye_phi_email_total_loc":                  loc_row.get("birdeye_phi_email_total"),
        "birdeye_phi_ssn_total_loc":                    loc_row.get("birdeye_phi_ssn_total"),
        "birdeye_top_source_loc":                       str(rng.choice(["Google","Yelp"])) if has_birdeye else None,
        "birdeye_name":                                 str(loc_row.get("LocationName","")) if has_birdeye else None,
        # RC / campaign
        "rc_attributed_calls":             rc_attr,
        "campaign_count_via_phone":        campaign_count,
        "campaigns_contacted_via_phone":   campaign_name if has_campaign else "",
        "primary_campaign_via_phone":      campaign_name,
        "has_campaign_assignment":         has_campaign,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_patients(n: int = N_PATIENTS) -> pd.DataFrame:
    """Generate patient.parquet — 42 columns, one row per patient-practice."""
    rows = []
    patient_id = 1000000
    for _ in range(n):
        src = str(rng.choice(PRACTICE_CODES))
        rows.append(_build_patient_row(patient_id, src))
        patient_id += int(rng.integers(1, 5))

    df = pd.DataFrame(rows)
    df["PatientID"]      = df["PatientID"].astype("Int64")
    df["PatientDOB"]     = pd.to_datetime(df["PatientDOB"])
    df["ResponsiblePartyDOB"] = pd.to_datetime(df["ResponsiblePartyDOB"])
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    df["_phone_institutional_flag"] = df["_phone_institutional_flag"].astype(bool)
    return df


def generate_nav_map(patient_df: pd.DataFrame, location_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate patient_navigation_map.parquet (144 columns).
    Builds from patient_df + financial rollup + multi-practice assignment.
    """
    records = []

    for _, pat in patient_df.iterrows():
        pid = int(pat["PatientID"])
        src = str(pat["Source_Database_Code"])
        rollup = _build_financial_rollup(pid, src, location_df)

        # Merge demographics + rollup
        row = {**pat.to_dict(), **rollup}
        row["has_email"] = pat["patEmail"] is not None
        row["has_phone"] = pat["PatientPhone"] is not None
        row["tbl_Refresh_Date"] = REFRESH_DATE
        records.append(row)

    df = pd.DataFrame(records)

    # Multi-practice: assign practice_count
    pid_counts = df.groupby("PatientID")["Source_Database_Code"].transform("count")
    df["practice_count"]    = pid_counts.astype("Int64")
    df["multi_practice_flag"] = (pid_counts > 1).astype("boolean")

    # Simulate ~15.8% multi-practice by duplicating some rows with a second practice
    single_patients = df[df["practice_count"] == 1]
    n_multi = int(len(single_patients) * MULTI_PRACTICE_RATE)
    extra_rows = single_patients.sample(n=min(n_multi, len(single_patients)),
                                        random_state=SEED).copy()
    # Assign second practice code (different from original)
    def pick_alt_practice(orig):
        alts = [p for p in PRACTICE_CODES if p != orig]
        return str(rng.choice(alts))

    extra_rows["Source_Database_Code"] = extra_rows["Source_Database_Code"].apply(pick_alt_practice)
    extra_rows["_pk_composite"] = extra_rows.apply(
        lambda r: composite_key(r["Source_Database_Code"], r["PatientID"]), axis=1)
    extra_rows["hashed_patient_id"] = extra_rows.apply(
        lambda r: hashed_patient_id(r["Source_Database_Code"], r["PatientID"]), axis=1)

    df = pd.concat([df, extra_rows], ignore_index=True)

    # Recompute practice_count after adding extras
    pid_counts2 = df.groupby("PatientID")["Source_Database_Code"].transform("count")
    df["practice_count"]    = pid_counts2.astype("Int64")
    df["multi_practice_flag"] = (pid_counts2 > 1).astype("boolean")

    # Cast key types
    df["PatientID"]        = df["PatientID"].astype("Int64")
    df["PatientDOB"]       = pd.to_datetime(df["PatientDOB"])
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    df["_active_window"]   = df["_active_window"].astype(bool)
    df["is_catastrophe"]   = df["is_catastrophe"].astype(bool)
    df["is_friction"]      = df["is_friction"].astype(bool)
    df["is_clean"]         = df["is_clean"].astype(bool)
    df["has_visits"]       = df["has_visits"].astype(bool)
    df["has_charges"]      = df["has_charges"].astype(bool)
    df["has_transactions"] = df["has_transactions"].astype(bool)
    df["has_statements"]   = df["has_statements"].astype(bool)
    df["has_inbound_calls"]  = df["has_inbound_calls"].astype(bool)
    df["has_outbound_calls"] = df["has_outbound_calls"].astype(bool)
    df["has_ringcentral"]    = df["has_ringcentral"].astype(bool)
    df["has_any_calls"]      = df["has_any_calls"].astype(bool)
    df["has_insurance"]      = df["has_insurance"].astype(bool)
    df["has_email"]          = df["has_email"].astype(bool)
    df["has_phone"]          = df["has_phone"].astype(bool)
    df["has_in_window_activity"] = df["has_in_window_activity"].astype(bool)
    df["has_birdeye_at_primary_location"] = df["has_birdeye_at_primary_location"].astype(bool)
    df["has_campaign_assignment"] = df["has_campaign_assignment"].astype(bool)
    df["_identity_key"]      = df["_identity_key"].astype("string")
    df["last_location_id"]   = df["last_location_id"].astype("string")
    df["call_tier"]          = df["call_tier"].astype("string")
    df["primary_campaign_via_phone"] = df["primary_campaign_via_phone"].astype("string")
    df["is_sapa"]   = df["is_sapa"].astype("boolean")
    df["is_nraa"]   = df["is_nraa"].astype("boolean")
    df["is_tennessee"] = df["is_tennessee"].astype("boolean")
    df["is_atlanta_404"] = df["is_atlanta_404"].astype("boolean")
    df["is_self_pay"]    = df["is_self_pay"].astype("boolean")
    df["is_bai"]         = df["is_bai"].astype("boolean")
    df["is_fully_covered"] = df["is_fully_covered"].astype("boolean")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path, n: int = N_PATIENTS, location_df: pd.DataFrame = None) -> dict:
    if location_df is None:
        loc_path = out_dir / "02_dims" / "location.parquet"
        if loc_path.exists():
            location_df = pd.read_parquet(loc_path)
        else:
            from gen_dims import generate_location
            location_df = generate_location()

    print(f"Generating patient.parquet ({n:,} patients) ...")
    patient_df = generate_patients(n)
    path = out_dir / "02_dims" / "patient.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    patient_df.to_parquet(path, index=False)
    print(f"  ✓ {len(patient_df):,} rows → {path}")

    print("Generating patient_navigation_map.parquet ...")
    nav_df = generate_nav_map(patient_df, location_df)
    path = out_dir / "00_navigation" / "patient_navigation_map.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    nav_df.to_parquet(path, index=False)
    print(f"  ✓ {len(nav_df):,} rows → {path}")

    return {"patient": patient_df, "nav_map": nav_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output")
    parser.add_argument("--n",   default=N_PATIENTS, type=int)
    args = parser.parse_args()
    run(Path(args.out), args.n)
