"""
gen_facts.py
────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates the four core clinical/financial fact tables:

  • 01_facts/visits.parquet          (one row per visit line-item)
  • 01_facts/charges.parquet         (one row per charge line)
  • 01_facts/transactions.parquet    (one row per payment/adjustment)
  • 01_facts/statements.parquet      (one row per statement)

Relationship chain (enforced):
  visits.PatientID       → patient.PatientID        (FK)
  visits.Source_Database_Code → patient.Source_Database_Code
  visits.LocationID      → location.LocationID       (FK)
  visits.PrimaryInsurancePlanNum → insurance.PlanNumber (FK, partial)
  charges.PatientID      → patient.PatientID
  charges.VisitID        → visits.VisitID
  charges.LocationID     → location.LocationID
  transactions.PatientID → patient.PatientID
  transactions.ChargeID  → charges.ChargeID
  statements.PatientID   → patient.PatientID

Run standalone:
    python gen_facts.py --out ./output
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from constants import (
    SEED, REFRESH_DATE, TXN_REFRESH_DATE, WINDOW_START, WINDOW_END,
    PROCEDURE_CODES, PROCEDURE_DESCRIPTIONS, ICD10_CODES,
    ADJUSTMENT_BUCKETS, ADJUSTMENT_WEIGHTS, ADJUSTMENT_TYPES,
    PROCESSING_TYPES, STATEMENT_LEVELS,
    AVG_VISITS_PER_PAT, AVG_CHARGES_PER_VISIT, AVG_TXNS_PER_CHARGE,
    AVG_STATEMENTS_PER_PAT,
)
from helpers import (
    rng,
    pick_date_in_window, pick_date_historical, pick_service_date, pick_post_date,
    pick_charge_amount, pick_payment_amount, pick_adjustment_amount,
    pick_outstanding_balance, aging_bucket,
    gen_order_number, gen_batch_number, gen_visit_number, gen_history_number,
    gen_statement_id, gen_icn_number, composite_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# VISITS
# ─────────────────────────────────────────────────────────────────────────────

def generate_visits(patient_df: pd.DataFrame,
                    location_df: pd.DataFrame,
                    insurance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grain: visit line-item.
    Visits table is at LINE-ITEM grain (avg 2.41 rows per VisitID) per the audit.
    Multiple rows per VisitID = chart history lines.
    """
    rows = []
    visit_id_counter = 100000

    # Build location and insurance lookup indexed by Source_Database_Code
    loc_by_src = {src: grp for src, grp in location_df.groupby("Source_Database_Code")}
    ins_by_src = {src: grp for src, grp in insurance_df.groupby("Source_Database_Code")}

    for _, pat in patient_df.iterrows():
        pid = int(pat["PatientID"])
        src = str(pat["Source_Database_Code"])
        n_visits = max(1, int(rng.integers(1, 8)))

        # Location pool for this practice
        loc_pool = loc_by_src.get(src, location_df)
        ins_pool = ins_by_src.get(src, insurance_df)

        for _ in range(n_visits):
            visit_id  = str(visit_id_counter)
            visit_id_counter += 1

            admit_date  = pick_date_in_window() if rng.random() > 0.3 else pick_date_historical()
            discharge   = admit_date + timedelta(hours=int(rng.integers(0, 48)))
            loc_row     = loc_pool.iloc[int(rng.integers(0, len(loc_pool)))]
            loc_id      = str(loc_row["LocationID"])

            # Insurance (12.7% null primary, 84.2% null secondary, 99% null tertiary)
            pri_plan = None
            if len(ins_pool) > 0 and rng.random() > 0.127:
                pri_row  = ins_pool.iloc[int(rng.integers(0, len(ins_pool)))]
                pri_plan = str(pri_row["PlanNumber"])
                pri_pol  = f"{int(rng.integers(10000000, 99999999))}"
                pri_grp  = f"{int(rng.integers(10000000, 99999999))}"
            else:
                pri_pol, pri_grp = None, None

            sec_plan, sec_pol, sec_grp = None, None, None
            if len(ins_pool) > 0 and rng.random() > 0.842:
                sec_row  = ins_pool.iloc[int(rng.integers(0, len(ins_pool)))]
                sec_plan = str(sec_row["PlanNumber"])
                sec_pol  = f"{int(rng.integers(100000, 9999999))}"
                sec_grp  = f"{int(rng.integers(10000000, 99999999))}"

            ter_plan = None
            if rng.random() > 0.990:
                ter_plan = str(rng.integers(100000, 999999))

            # Avg 2.41 line-items per visit
            n_lines = max(1, int(rng.choice([1,2,3,4,5], p=[0.42,0.30,0.15,0.08,0.05])))
            for _ in range(n_lines):
                rows.append({
                    "Source_Database_Code":          src,
                    "VisitID":                        visit_id,
                    "PatientID":                      pid,
                    "PrimaryInsurancePlanNum":        pri_plan,
                    "PrimaryInsurancePolicyNumber":   pri_pol,
                    "SecondaryInsurancePlanNum":       sec_plan,
                    "SecondaryInsurancePolicyNumber":  sec_pol,
                    "TertiaryInsurancePlanNum":        ter_plan,
                    "TertiaryInsurancePolicyNumber":   None,
                    "VisitNumber":                     gen_visit_number(src),
                    "HistoryNumber":                   gen_history_number(src) if rng.random() > 0.007 else None,
                    "LocationID":                      loc_id,
                    "PrimaryAuthorizationNumber":      f"{int(rng.integers(100000000000, 999999999999))}" if rng.random() > 0.825 else None,
                    "SecondaryAuthorizationNumber":    f"{int(rng.integers(100000000, 999999999))}" if rng.random() > 0.990 else None,
                    "TertiaryAuthorizationNumber":     None,
                    "AdmitDate":                       admit_date if rng.random() > 0.005 else None,
                    "DischargeDate":                   discharge,
                    "PrimaryInsuranceGroup":           pri_grp,
                    "SecondaryInsuranceGroup":         sec_grp,
                    "TertiaryInsuranceGroup":          None,
                    "tbl_Refresh_Date":                REFRESH_DATE,
                    "_pk_composite":                   composite_key(src, pid),
                })

    df = pd.DataFrame(rows)
    df["PatientID"]     = df["PatientID"].astype("Int64")
    df["VisitID"]       = df["VisitID"].astype("string")
    df["LocationID"]    = df["LocationID"].astype("string")
    df["PrimaryInsurancePlanNum"]   = df["PrimaryInsurancePlanNum"].astype("string")
    df["SecondaryInsurancePlanNum"] = df["SecondaryInsurancePlanNum"].astype("string")
    df["TertiaryInsurancePlanNum"]  = df["TertiaryInsurancePlanNum"].astype("string")
    df["AdmitDate"]         = pd.to_datetime(df["AdmitDate"])
    df["DischargeDate"]     = pd.to_datetime(df["DischargeDate"])
    df["tbl_Refresh_Date"]  = pd.to_datetime(df["tbl_Refresh_Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHARGES
# ─────────────────────────────────────────────────────────────────────────────

def generate_charges(patient_df: pd.DataFrame,
                     visit_df: pd.DataFrame,
                     location_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grain: one row per ChargeID.
    Linked to visits via VisitID (FK). Multiple charges per visit.
    """
    rows = []
    charge_id_counter = 10000000
    loc_by_src = {src: grp for src, grp in location_df.groupby("Source_Database_Code")}

    # Build visit lookup: (src, patient_id) → list of VisitIDs
    visits_lookup = {}
    for _, v in visit_df[["Source_Database_Code","PatientID","VisitID"]].drop_duplicates(
            subset=["Source_Database_Code","PatientID","VisitID"]).iterrows():
        key = (str(v["Source_Database_Code"]), int(v["PatientID"]))
        visits_lookup.setdefault(key, []).append(str(v["VisitID"]))

    for _, pat in patient_df.iterrows():
        pid = int(pat["PatientID"])
        src = str(pat["Source_Database_Code"])
        key = (src, pid)
        visit_ids = visits_lookup.get(key, [str(10000000 + pid)])

        loc_pool = loc_by_src.get(src, location_df)
        n_charges = max(1, int(len(visit_ids) * float(rng.uniform(1.5, 3.5))))

        for i in range(n_charges):
            cid        = str(charge_id_counter)
            charge_id_counter += 1
            visit_id   = str(rng.choice(visit_ids))
            loc_row    = loc_pool.iloc[int(rng.integers(0, len(loc_pool)))]
            loc_id     = str(loc_row["LocationID"])

            proc_code  = str(rng.choice(PROCEDURE_CODES))
            proc_desc  = PROCEDURE_DESCRIPTIONS.get(proc_code,
                         f"RADIOLOGY PROCEDURE {proc_code}")

            svc_date   = pick_date_in_window() if rng.random() > 0.3 else pick_date_historical()
            post_date  = pick_post_date(svc_date)
            amount     = pick_charge_amount()
            balance    = round(amount * float(rng.uniform(0, 1)), 2)

            # ICD-10 diagnoses (1 required, up to 10, increasing null rate)
            icd = lambda null_pct: str(rng.choice(ICD10_CODES)) if rng.random() > null_pct else None

            is_voided   = 1 if rng.random() < 0.019 else 0
            voided_date = pick_date_in_window() if is_voided else None

            rows.append({
                "Source_Database_Code":       src,
                "ChargeID":                   cid,
                "ChargeCount":                float(rng.choice([0,1,2,3], p=[0.7,0.2,0.07,0.03])),
                "PatientID":                  pid,
                "ChargeAmount":               amount,
                "ProcedureModality":          str(rng.choice(["MRI","CT","XR","US","NM","PQRS","DXA"])),
                "PostDate":                   post_date,
                "ServiceDate":                svc_date,
                "LocationID":                 loc_id,
                "CurrentResponsibleLevel":    str(rng.choice(["Primary","Secondary","Tertiary","Patient"],
                                                             p=[0.55,0.15,0.05,0.25])),
                "ReferringDoctorID":          str(int(rng.integers(1000, 9999))),
                "ProcedureCode":              proc_code,
                "ProcedureDescription":       proc_desc,
                "DoctorID":                   str(float(int(rng.integers(1, 200)))),
                "Balance":                    balance,
                "UserName":                   str(rng.choice(["Appliance","System","Import","BatchProc"])),
                "PlaceOfService":             str(float(int(rng.choice([11,21,22,23,24,31,32,49,71,72])))),
                "Modifier":                   str(rng.choice(["26","TC","LT","RT","GG",""])) if rng.random() > 0.282 else None,
                "OrderNumber":                gen_order_number(src),
                "ChargeCreateDate":           post_date + timedelta(hours=int(rng.integers(0, 24)),
                                                                     minutes=int(rng.integers(0, 60))),
                "ICD10Diagnosis1":            icd(0.020),
                "ICD10Diagnosis2":            icd(0.562),
                "ICD10Diagnosis3":            icd(0.772),
                "ICD10Diagnosis4":            icd(0.876),
                "ICD10Diagnosis5":            icd(0.998),
                "ICD10Diagnosis6":            None,
                "ICD10Diagnosis7":            None,
                "ICD10Diagnosis8":            None,
                "ICD10Diagnosis9":            None,
                "ICD10Diagnosis10":           None,
                "isVoided":                   is_voided,
                "VoidedDate":                 voided_date,
                "isHold":                     1 if rng.random() < 0.03 else 0,
                "TransferFlag":               int(rng.choice([0,1], p=[0.3,0.7])),
                "VisitID":                    visit_id,
                "ActionDescription":          None,
                "ChargeUnit":                 float(rng.choice([1,2,3], p=[0.90,0.07,0.03])),
                "InterpretationLocationID":   str(float(int(rng.integers(100000, 200000)))) if rng.random() > 0.264 else None,
                "Batchnumber":                gen_batch_number(src),
                "DepartmentCode":             None,
                "Doctor2ID":                  "0",
                "PrimaryAuthorizationNumber": f"C{int(rng.integers(100000000, 999999999))}" if rng.random() > 0.996 else None,
                "SecondaryAuthorizationNumber": None,
                "TertiaryAuthorizationNumber": None,
                "OtherDoctorID":              "0",
                "NextFollowUpDate":           pick_date_in_window() if rng.random() > 0.882 else None,
                "ChargeStatusBillStage":      int(rng.choice([0,1,2,3], p=[0.05,0.65,0.20,0.10])),
                "ChargeStatusReleased":       int(rng.choice([0,1], p=[0.6,0.4])),
                "SpecialUpdateFlag":          None,
                "DOS_AgingBucket":            aging_bucket(svc_date),
                "Payment_Plan_Present":       None,
                "LineStatus":                 str(rng.choice(["PT","INS","VOID","HOLD","CLEAN"],
                                                             p=[0.30,0.40,0.10,0.05,0.15])) if rng.random() > 0.46 else None,
                "tbl_Refresh_Date":           REFRESH_DATE,
                "_pk_composite":              composite_key(src, pid),
            })

    df = pd.DataFrame(rows)
    df["PatientID"]   = df["PatientID"].astype("Int64")
    df["ChargeID"]    = df["ChargeID"].astype("string")
    df["VisitID"]     = df["VisitID"].astype("string")
    df["LocationID"]  = df["LocationID"].astype("string")
    df["ReferringDoctorID"]     = df["ReferringDoctorID"].astype("string")
    df["Doctor2ID"]   = df["Doctor2ID"].astype("string")
    df["OtherDoctorID"]         = df["OtherDoctorID"].astype("string")
    df["PlaceOfService"]        = df["PlaceOfService"].astype("string")
    df["InterpretationLocationID"] = df["InterpretationLocationID"].astype("string")
    df["SecondaryAuthorizationNumber"] = df["SecondaryAuthorizationNumber"].astype("string")
    df["TertiaryAuthorizationNumber"]  = df["TertiaryAuthorizationNumber"].astype("string")
    df["PostDate"]         = pd.to_datetime(df["PostDate"])
    df["ServiceDate"]      = pd.to_datetime(df["ServiceDate"])
    df["ChargeCreateDate"] = pd.to_datetime(df["ChargeCreateDate"])
    df["VoidedDate"]       = pd.to_datetime(df["VoidedDate"])
    df["NextFollowUpDate"] = pd.to_datetime(df["NextFollowUpDate"])
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_transactions(patient_df: pd.DataFrame,
                           charge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grain: one row per PaymentID.
    Linked to charges via ChargeID (FK).
    Includes payments and adjustments with 4-bucket taxonomy.
    """
    rows = []
    payment_id_counter = 100000000

    # Charge lookup: (src, patient_id) → list of ChargeIDs
    charge_lookup = {}
    for _, c in charge_df[["Source_Database_Code","PatientID","ChargeID","ChargeAmount"]].iterrows():
        key = (str(c["Source_Database_Code"]), int(c["PatientID"]))
        charge_lookup.setdefault(key, []).append((str(c["ChargeID"]), float(c["ChargeAmount"])))

    for _, pat in patient_df.iterrows():
        pid   = int(pat["PatientID"])
        src   = str(pat["Source_Database_Code"])
        key   = (src, pid)
        chgs  = charge_lookup.get(key, [])
        if not chgs:
            continue

        n_txns = max(1, int(len(chgs) * float(rng.uniform(0.8, 2.5))))

        for _ in range(n_txns):
            charge_id, charge_amt = chgs[int(rng.integers(0, len(chgs)))]
            pid_  = payment_id_counter
            payment_id_counter += 1

            post_date  = pick_date_in_window() if rng.random() > 0.4 else pick_date_historical()
            adj_bucket = str(rng.choice(ADJUSTMENT_BUCKETS, p=ADJUSTMENT_WEIGHTS))
            adj_type   = str(rng.choice(ADJUSTMENT_TYPES[adj_bucket]))
            proc_type  = str(rng.choice(PROCESSING_TYPES,
                             p=[0.69,0.14,0.06,0.04,0.02,0.01,0.01,0.01,0.005,0.005,0.003,0.002]))

            pay_amount  = pick_payment_amount(charge_amt) if "Pay" in proc_type or proc_type == "Payment" else 0.0
            adj_amount  = pick_adjustment_amount(charge_amt, pay_amount) if adj_bucket != "payment_plan" else 0.0
            balance_aft = max(0.0, round(charge_amt - pay_amount - abs(adj_amount), 2))
            bad_debt    = adj_amount if adj_bucket == "bad_debt" else 0.0

            rows.append({
                "Source_Database_Code":   src,
                "PatientID":              pid,
                "ChargeID":               charge_id,
                "SourceTableName":        "Payment",
                "PaymentID":              str(pid_),
                "PaymentAmount":          pay_amount,
                "AdjustmentAmount":       adj_amount,
                "PostDate":               post_date,
                "InsurancePlan":          str(int(rng.integers(1000, 9999))) if rng.random() > 0.376 else None,
                "AdjustmentType":         adj_type if rng.random() > 0.240 else None,
                "ProcessingType":         proc_type,
                "Paysource":              str(rng.choice(["Patient","Insurance","Agency",""],
                                                        p=[0.35,0.35,0.05,0.25])) if rng.random() > 0.272 else None,
                "BatchNumber":            str(rng.choice(["COL","PAY","ADJ","BATCH001"])),
                "ICNNumber":              gen_icn_number(),
                "isERSPayment":           0,
                "BalanceAfterPost":       balance_aft,
                "CheckNumber":            str(int(rng.integers(1000, 9999))) if rng.random() > 0.367 else None,
                "PostDateWTime":          post_date + timedelta(hours=int(rng.integers(0,12)),
                                                                minutes=int(rng.integers(0,60))),
                "SystemDateWithTime":     post_date + timedelta(hours=int(rng.integers(0,12)),
                                                                minutes=int(rng.integers(0,60)),
                                                                seconds=int(rng.integers(0,60))),
                "TransactionType":        str(rng.choice(["Standard","Reversal","Transfer"], p=[0.90,0.05,0.05])),
                "TransferFlag":           int(rng.choice([0,1], p=[0.3,0.7])),
                "AllowedAmount":          round(charge_amt * float(rng.uniform(0.3, 0.9)), 2),
                "PaymentType":            str(int(rng.integers(1000,9999))) if rng.random() > 0.367 else None,
                "PaymentModule":          str(rng.choice(["Manual","Auto","EDI","Portal"])),
                "CoInsuranceAmount":      round(float(rng.uniform(0, 100)), 2) if rng.random() > 0.015 else None,
                "DeductibleAmount":       round(float(rng.uniform(0, 500)), 2) if rng.random() > 0.015 else None,
                "CoPayAmount":            round(float(rng.uniform(0, 50)), 2) if rng.random() > 0.015 else None,
                "SpecialUpdateFlag":      None,
                "RefundAmounts":          0.0,
                "DenialCode":             0,
                "DenialNote":             str(rng.choice([
                                            "CO144 - Incentive adjustment, eg, preferred product/service.",
                                            "CO45 - Charges exceed your contracted/legislated fee arrangement.",
                                            "CO29 - The time limit for filing has expired.",
                                            "PR1 - Deductible Amount",
                                          ])) if rng.random() > 0.629 else None,
                "BadDebtAdjustments":     bad_debt,
                "Days_to_Agency_Placement": float(int(rng.integers(30, 365))) if rng.random() > 0.791 else None,
                "PaymentMethod":          str(rng.choice(["Manual Payments","EDI","Portal","IVR","Check"])) if rng.random() > 0.503 else None,
                "tbl_Refresh_Date":       TXN_REFRESH_DATE,
                "AdjustmentBucket":       adj_bucket,
                "_pk_composite":          composite_key(src, pid),
            })

    df = pd.DataFrame(rows)
    df["PatientID"]    = df["PatientID"].astype("Int64")
    df["ChargeID"]     = df["ChargeID"].astype("string")
    df["PaymentID"]    = df["PaymentID"].astype("string")
    df["PostDate"]     = pd.to_datetime(df["PostDate"])
    df["PostDateWTime"]     = pd.to_datetime(df["PostDateWTime"])
    df["SystemDateWithTime"] = pd.to_datetime(df["SystemDateWithTime"])
    df["tbl_Refresh_Date"]   = pd.to_datetime(df["tbl_Refresh_Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATEMENTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_statements(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grain: one row per StatementID.
    Linked to patient via (Source_Database_Code, PatientID).
    """
    rows = []

    for _, pat in patient_df.iterrows():
        pid = int(pat["PatientID"])
        src = str(pat["Source_Database_Code"])
        n_stmts = int(rng.choice([0,1,2,3,4,5,6,7,8],
                                  p=[0.05,0.20,0.25,0.20,0.12,0.08,0.05,0.03,0.02]))
        for _ in range(n_stmts):
            created   = pick_date_in_window()
            released  = created + timedelta(minutes=int(rng.integers(5, 60)))
            stmt_id   = gen_statement_id(src, pid, created)
            pat_bal   = round(float(rng.uniform(0, 2000)), 2)
            tot_bal   = round(pat_bal + float(rng.uniform(0, 5000)), 2)
            level     = str(rng.choice(STATEMENT_LEVELS))

            email_ok  = str(rng.choice(["Yes","No"])) if rng.random() > 0.829 else None
            text_ok   = str(rng.choice(["Yes","No"])) if rng.random() > 0.416 else None
            stmt_ok   = str(rng.choice(["Yes","No"])) if rng.random() > 0.341 else None

            email_dt  = created if email_ok == "Yes" else None
            text_dt   = created if text_ok  == "Yes" else None

            rows.append({
                "Source_Database_Code":  src,
                "PatientID":             pid,
                "StatementID":           stmt_id,
                "PatientBalance":        pat_bal,
                "TotalBalance":          tot_bal,
                "StatementLevel":        level,
                "CreatedDate":           created,
                "ReleasedDate":          released if rng.random() > 0.001 else None,
                "IsReleased":            1 if rng.random() > 0.1 else 0,
                "IsOnHold":              1 if rng.random() < 0.05 else 0,
                "HoldNote":              None,
                "Email_Successful":      email_ok,
                "Text_Successful":       text_ok,
                "Statement_Successful":  stmt_ok,
                "first_email_sent_date": email_dt,
                "first_text_sent_date":  text_dt,
                "tbl_Refresh_Date":      REFRESH_DATE,
                "_pk_composite":         composite_key(src, pid),
            })

    df = pd.DataFrame(rows)
    df["PatientID"]    = df["PatientID"].astype("Int64")
    df["CreatedDate"]  = pd.to_datetime(df["CreatedDate"])
    df["ReleasedDate"] = pd.to_datetime(df["ReleasedDate"])
    df["first_email_sent_date"] = pd.to_datetime(df["first_email_sent_date"])
    df["first_text_sent_date"]  = pd.to_datetime(df["first_text_sent_date"])
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path,
        patient_df: pd.DataFrame = None,
        location_df: pd.DataFrame = None,
        insurance_df: pd.DataFrame = None) -> dict:

    def _load(path, gen_fn, *args):
        if path.exists():
            return pd.read_parquet(path)
        return gen_fn(*args)

    dims = out_dir / "02_dims"
    if patient_df is None:
        patient_df = pd.read_parquet(dims / "patient.parquet")
    if location_df is None:
        location_df = pd.read_parquet(dims / "location.parquet")
    if insurance_df is None:
        insurance_df = pd.read_parquet(dims / "insurance.parquet")

    facts = out_dir / "01_facts"
    facts.mkdir(parents=True, exist_ok=True)

    print("Generating visits.parquet ...")
    visit_df = generate_visits(patient_df, location_df, insurance_df)
    visit_df.to_parquet(facts / "visits.parquet", index=False)
    print(f"  ✓ {len(visit_df):,} rows → {facts / 'visits.parquet'}")

    print("Generating charges.parquet ...")
    charge_df = generate_charges(patient_df, visit_df, location_df)
    charge_df.to_parquet(facts / "charges.parquet", index=False)
    print(f"  ✓ {len(charge_df):,} rows → {facts / 'charges.parquet'}")

    print("Generating transactions.parquet ...")
    txn_df = generate_transactions(patient_df, charge_df)
    txn_df.to_parquet(facts / "transactions.parquet", index=False)
    print(f"  ✓ {len(txn_df):,} rows → {facts / 'transactions.parquet'}")

    print("Generating statements.parquet ...")
    stmt_df = generate_statements(patient_df)
    stmt_df.to_parquet(facts / "statements.parquet", index=False)
    print(f"  ✓ {len(stmt_df):,} rows → {facts / 'statements.parquet'}")

    return {"visits": visit_df, "charges": charge_df,
            "transactions": txn_df, "statements": stmt_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output")
    args = parser.parse_args()
    run(Path(args.out))
