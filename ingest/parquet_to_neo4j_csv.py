"""
parquet_to_neo4j_csv.py
───────────────────────
SGS — RP Knowledge Graph
Converts all RP parquet files to neo4j-admin import-compatible CSV files.

Output structure (all files written to --out directory):
  NODES (17 files):
    nodes_patient.csv          nodes_practice.csv
    nodes_location.csv         nodes_insurance.csv
    nodes_visit.csv            nodes_charge.csv
    nodes_transaction.csv      nodes_statement.csv
    nodes_rccall.csv           nodes_ivrinbound.csv
    nodes_diallercall.csv      nodes_phonebridge.csv
    nodes_campaign.csv         nodes_birdeye.csv
    nodes_diagnosiscode.csv    nodes_procedurecode.csv

  RELATIONSHIPS (22 files):
    rel_patient_practice.csv   rel_patient_visit.csv
    rel_patient_charge.csv     rel_patient_transaction.csv
    rel_patient_statement.csv  rel_patient_ivr.csv
    rel_patient_dialler.csv    rel_patient_phonebridge.csv
    rel_visit_location.csv     rel_visit_insurance.csv
    rel_charge_visit.csv       rel_charge_location.csv
    rel_charge_diagnosis.csv   rel_charge_procedure.csv
    rel_transaction_charge.csv rel_location_practice.csv
    rel_insurance_practice.csv rel_rccall_campaign.csv
    rel_rccall_phonebridge.csv rel_birdeye_location.csv
    rel_campaign_practice.csv  rel_ivr_patient.csv

neo4j-admin import ID spaces used:
  Patient    → "{Source_Database_Code}:{PatientID}"
  Practice   → "{Source_Database_Code}"
  Location   → "{Source_Database_Code}:{LocationID}"
  Visit      → "{Source_Database_Code}:{VisitID}"
  Charge     → "{Source_Database_Code}:{ChargeID}"
  Transaction→ "{Source_Database_Code}:{PaymentID}"
  Statement  → "{StatementID}"
  InsurancePlan → "{Source_Database_Code}:{PlanNumber}"
  RCCall     → "{Contact_ID}"
  IVRInbound → "{ResponseID}"
  DiallerCall→ "{ACCOUNT}"
  PhoneBridge→ "{Source_Database_Code}:{PatientID}:{phone_norm}"
  Campaign   → "{Campaign Name}"
  BirdeyeReview → "{Location}::{Date Posted On}"
  DiagnosisCode → "{icd_code}"
  ProcedureCode → "{ProcedureCode}"

Usage:
    python parquet_to_neo4j_csv.py --data ./rp_dataset_synthetic --out ./neo4j_import
    python parquet_to_neo4j_csv.py --data /home/siva/work/codebase/RP/generator/rp_synthetic_output --out ./neo4j_import
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/NaT/None with empty string for CSV output."""
    return df.where(pd.notnull(df), "")


def fmt_dt(series: pd.Series) -> pd.Series:
    """Format datetime columns as ISO strings neo4j-admin understands."""
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")


def pid(src: pd.Series, patient_id: pd.Series) -> pd.Series:
    """Composite patient ID: 'SAPA:1000000'"""
    return src.astype(str) + ":" + patient_id.astype(str)


def lid(src: pd.Series, location_id: pd.Series) -> pd.Series:
    """Composite location ID: 'SAPA:1'"""
    return src.astype(str) + ":" + location_id.astype(str)


def vid(src: pd.Series, visit_id: pd.Series) -> pd.Series:
    """Composite visit ID: 'SAPA:100000'"""
    return src.astype(str) + ":" + visit_id.astype(str)


def cid(src: pd.Series, charge_id: pd.Series) -> pd.Series:
    """Composite charge ID: 'SAPA:10000000'"""
    return src.astype(str) + ":" + charge_id.astype(str)


def tid(src: pd.Series, payment_id: pd.Series) -> pd.Series:
    """Composite transaction ID: 'SAPA:100000000'"""
    return src.astype(str) + ":" + payment_id.astype(str)


def iid(src: pd.Series, plan_num: pd.Series) -> pd.Series:
    """Composite insurance ID: 'SAPA:100000'"""
    return src.astype(str) + ":" + plan_num.astype(str)


def write_csv(df: pd.DataFrame, path: Path, label: str, rows: int):
    df.to_csv(path, index=False)
    logger.info(f"  ✓ {path.name:<45} {rows:>10,} rows   [{label}]")


# ════════════════════════════════════════════════════════════════════
# NODE EXPORTERS
# ════════════════════════════════════════════════════════════════════

def export_patients(data_dir: Path, out_dir: Path):
    logger.info("Exporting Patient nodes ...")

    # Load patient dims + nav map for financial enrichment
    pat = pd.read_parquet(data_dir / "02_dims/patient.parquet")

    nav_cols = [
        "Source_Database_Code", "PatientID",
        "total_charged", "total_paid", "outstanding_balance", "total_adjusted",
        "adj_contractual", "adj_bad_debt", "adj_collection_agency",
        "adj_refund_reversal", "adj_other",
        "payor_cohort", "call_tier", "is_catastrophe", "is_friction", "is_clean",
        "is_self_pay", "is_bai", "is_fully_covered",
        "is_sapa", "is_nraa", "is_tennessee", "is_atlanta_404",
        "multi_practice_flag", "practice_count",
        "visit_count", "charge_count", "statement_count", "total_calls_window",
        "has_any_calls", "has_insurance", "_active_window",
        "PlanName", "Carrier_Name", "PlanType",
    ]
    nav = pd.read_parquet(
        data_dir / "00_navigation/patient_navigation_map.parquet",
        columns=nav_cols
    ).drop_duplicates(subset=["Source_Database_Code", "PatientID"], keep="last")

    df = pat.merge(nav, on=["Source_Database_Code", "PatientID"], how="left")

    out = pd.DataFrame()
    out["patientId:ID(Patient)"]               = pid(df["Source_Database_Code"], df["PatientID"])
    out["source_db"]                           = df["Source_Database_Code"]
    out["patient_id:long"]                     = df["PatientID"]
    out["first_name"]                          = df["PatientFirstName"].fillna("")
    out["middle_name"]                         = df["PatientMiddleName"].fillna("")
    out["last_name"]                           = df["PatientLastName"].fillna("")
    out["dob:datetime"]                        = fmt_dt(df["PatientDOB"])
    out["gender"]                              = df["PatientGender"].fillna("")
    out["race"]                                = df["PatientRace"].fillna("")
    out["ethnicity"]                           = df["PatientEthnicity"].fillna("")
    out["city"]                                = df["PatientCity"].fillna("")
    out["state"]                               = df["PatientState"].fillna("")
    out["zip"]                                 = df["PatientZip"].fillna("")
    out["phone_norm"]                          = df["PatientPhone_norm"].fillna("")
    out["cell_norm"]                           = df["PatientCellPhone_norm"].fillna("")
    out["email"]                               = df["patEmail"].fillna("")
    out["propensity_grade"]                    = df["Imagine_Propensity_to_Pay_Grade"].fillna("")
    out["propensity_desc"]                     = df["Imagine_Propensity_to_Pay_Description"].fillna("")
    out["bad_address_indicator:float"]         = df["Bad_Address_Indicator"].fillna("")
    # Financial rollups from nav map
    out["total_charged:float"]                 = df["total_charged"].fillna("")
    out["total_paid:float"]                    = df["total_paid"].fillna("")
    out["outstanding_balance:float"]           = df["outstanding_balance"].fillna("")
    out["total_adjusted:float"]                = df["total_adjusted"].fillna("")
    out["adj_contractual:float"]               = df["adj_contractual"].fillna("")
    out["adj_bad_debt:float"]                  = df["adj_bad_debt"].fillna("")
    out["adj_collection_agency:float"]         = df["adj_collection_agency"].fillna("")
    out["adj_refund_reversal:float"]           = df["adj_refund_reversal"].fillna("")
    out["adj_other:float"]                     = df["adj_other"].fillna("")
    # Cohort flags
    out["payor_cohort"]                        = df["payor_cohort"].fillna("")
    out["call_tier"]                           = df["call_tier"].fillna("")
    out["is_catastrophe:boolean"]              = df["is_catastrophe"].fillna(False)
    out["is_friction:boolean"]                 = df["is_friction"].fillna(False)
    out["is_clean:boolean"]                    = df["is_clean"].fillna(False)
    out["is_self_pay:boolean"]                 = df["is_self_pay"].fillna(False)
    out["is_bai:boolean"]                      = df["is_bai"].fillna(False)
    out["is_fully_covered:boolean"]            = df["is_fully_covered"].fillna(False)
    out["is_sapa:boolean"]                     = df["is_sapa"].fillna(False)
    out["is_nraa:boolean"]                     = df["is_nraa"].fillna(False)
    out["is_tennessee:boolean"]                = df["is_tennessee"].fillna(False)
    out["is_atlanta_404:boolean"]              = df["is_atlanta_404"].fillna(False)
    out["multi_practice_flag:boolean"]         = df["multi_practice_flag"].fillna(False)
    out["practice_count:int"]                  = df["practice_count"].fillna(0)
    out["visit_count:float"]                   = df["visit_count"].fillna(0)
    out["charge_count:int"]                    = df["charge_count"].fillna(0)
    out["statement_count:int"]                 = df["statement_count"].fillna(0)
    out["total_calls_window:float"]            = df["total_calls_window"].fillna(0)
    out["has_any_calls:boolean"]               = df["has_any_calls"].fillna(False)
    out["has_insurance:boolean"]               = df["has_insurance"].fillna(False)
    out["active_window:boolean"]               = df["_active_window"].fillna(False)
    out["carrier_name"]                        = df["Carrier_Name"].fillna("")
    out["plan_name"]                           = df["PlanName"].fillna("")
    out["plan_type"]                           = df["PlanType"].fillna("")
    out[":LABEL"]                              = "Patient"

    write_csv(out, out_dir / "nodes_patient.csv", "Patient", len(out))


def export_practices(data_dir: Path, out_dir: Path):
    logger.info("Exporting Practice nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/patient.parquet",
                         columns=["Source_Database_Code"])
    codes = df["Source_Database_Code"].dropna().unique()

    out = pd.DataFrame()
    out["practiceId:ID(Practice)"] = codes
    out["code"]                    = codes
    out[":LABEL"]                  = "Practice"

    write_csv(out, out_dir / "nodes_practice.csv", "Practice", len(out))


def export_locations(data_dir: Path, out_dir: Path):
    logger.info("Exporting Location nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/location.parquet")

    out = pd.DataFrame()
    out["locationId:ID(Location)"]      = lid(df["Source_Database_Code"], df["LocationID"])
    out["source_db"]                    = df["Source_Database_Code"]
    out["location_id"]                  = df["LocationID"].fillna("")
    out["name"]                         = df["LocationName"].fillna("")
    out["abbreviation"]                 = df["LocationAbbreviation"].fillna("")
    out["npi"]                          = df["LocationNPINumber"].fillna("")
    out["address"]                      = df["LocationAddress"].fillna("")
    out["city"]                         = df["LocationCity"].fillna("")
    out["state"]                        = df["LocationState"].fillna("")
    out["zip"]                          = df["LocationZip"].fillna("")
    out["phone_norm"]                   = df["LocationPhone_norm"].fillna("")
    out["fax_norm"]                     = df["LocationFax_norm"].fillna("")
    out["location_type"]                = df["LocationType"].fillna("")
    out["fda_number"]                   = df["LocationFDANumber"].fillna("")
    out["birdeye_review_count:float"]   = df["birdeye_review_count"].fillna("")
    out["birdeye_avg_rating:float"]     = df["birdeye_avg_rating"].fillna("")
    out["birdeye_phi_review_count:float"]= df["birdeye_phi_review_count"].fillna("")
    out["birdeye_one_star_pct:float"]   = df["birdeye_one_star_pct"].fillna("")
    out[":LABEL"]                       = "Location"

    write_csv(out, out_dir / "nodes_location.csv", "Location", len(out))


def export_insurance(data_dir: Path, out_dir: Path):
    logger.info("Exporting InsurancePlan nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/insurance.parquet")

    out = pd.DataFrame()
    out["insuranceId:ID(InsurancePlan)"] = iid(df["Source_Database_Code"], df["PlanNumber"])
    out["source_db"]                     = df["Source_Database_Code"]
    out["plan_number"]                   = df["PlanNumber"].fillna("")
    out["plan_name"]                     = df["PlanName"].fillna("")
    out["plan_type"]                     = df["PlanType"].fillna("")
    out["carrier_name"]                  = df["Carrier_Name"].fillna("")
    out[":LABEL"]                        = "InsurancePlan"

    write_csv(out, out_dir / "nodes_insurance.csv", "InsurancePlan", len(out))


def export_campaigns(data_dir: Path, out_dir: Path):
    logger.info("Exporting Campaign nodes ...")
    df = pd.read_parquet(data_dir / "03_supplementary/campaign_map.parquet")

    out = pd.DataFrame()
    out["campaignId:ID(Campaign)"] = df["Campaign Name"]
    out["name"]                    = df["Campaign Name"]
    out["source_db"]               = df["Source Database Code"].fillna("")
    out["notes"]                   = df["Notes"].fillna("")
    out[":LABEL"]                  = "Campaign"

    write_csv(out, out_dir / "nodes_campaign.csv", "Campaign", len(out))


def export_birdeye(data_dir: Path, out_dir: Path):
    logger.info("Exporting BirdeyeReview nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/birdeye.parquet")
    df = df.drop_duplicates(subset=["Location", "Date Posted On"], keep="last")

    # Composite ID: location::date (:: avoids collision with : in location names)
    bid = df["Location"].astype(str) + "::" + df["Date Posted On"].astype(str)

    out = pd.DataFrame()
    out["birdeyeId:ID(BirdeyeReview)"] = bid
    out["location"]                    = df["Location"]
    out["date_posted"]                 = df["Date Posted On"]
    out["source"]                      = df["Review Source"]
    out["rating:int"]                  = df["Review Rating"]
    out["comment"]                     = df["Review Comment"].fillna("")
    out["phi_phone:int"]               = df["phi_phone_count"]
    out["phi_email:int"]               = df["phi_email_count"]
    out["phi_flagged:boolean"]         = df["phi_flagged"]
    out[":LABEL"]                      = "BirdeyeReview"

    write_csv(out, out_dir / "nodes_birdeye.csv", "BirdeyeReview", len(out))


def export_visits(data_dir: Path, out_dir: Path):
    logger.info("Exporting Visit nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/visits.parquet")
    df = df.drop_duplicates(subset=["Source_Database_Code", "VisitID"], keep="last")

    out = pd.DataFrame()
    out["visitId:ID(Visit)"]              = vid(df["Source_Database_Code"], df["VisitID"])
    out["source_db"]                      = df["Source_Database_Code"]
    out["visit_id"]                       = df["VisitID"].fillna("")
    out["visit_number"]                   = df["VisitNumber"].fillna("")
    out["history_number"]                 = df["HistoryNumber"].fillna("")
    out["admit_date:datetime"]            = fmt_dt(df["AdmitDate"])
    out["discharge_date:datetime"]        = fmt_dt(df["DischargeDate"])
    out["location_id"]                    = df["LocationID"].fillna("")
    out["primary_insurance_plan"]         = df["PrimaryInsurancePlanNum"].fillna("")
    out["primary_policy_number"]          = df["PrimaryInsurancePolicyNumber"].fillna("")
    out["secondary_insurance_plan"]       = df["SecondaryInsurancePlanNum"].fillna("")
    out["primary_auth_number"]            = df["PrimaryAuthorizationNumber"].fillna("")
    out["primary_insurance_group"]        = df["PrimaryInsuranceGroup"].fillna("")
    out[":LABEL"]                         = "Visit"

    write_csv(out, out_dir / "nodes_visit.csv", "Visit", len(out))


def export_charges(data_dir: Path, out_dir: Path):
    logger.info("Exporting Charge nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet")

    out = pd.DataFrame()
    out["chargeId:ID(Charge)"]             = cid(df["Source_Database_Code"], df["ChargeID"])
    out["source_db"]                       = df["Source_Database_Code"]
    out["charge_id"]                       = df["ChargeID"].fillna("")
    out["charge_amount:float"]             = df["ChargeAmount"].fillna("")
    out["procedure_code"]                  = df["ProcedureCode"].fillna("")
    out["procedure_description"]           = df["ProcedureDescription"].fillna("")
    out["procedure_modality"]              = df["ProcedureModality"].fillna("")
    out["service_date:datetime"]           = fmt_dt(df["ServiceDate"])
    out["post_date:datetime"]              = fmt_dt(df["PostDate"])
    out["balance:float"]                   = df["Balance"].fillna("")
    out["current_responsible_level"]       = df["CurrentResponsibleLevel"].fillna("")
    out["place_of_service"]                = df["PlaceOfService"].fillna("")
    out["modifier"]                        = df["Modifier"].fillna("")
    out["dos_aging_bucket"]                = df["DOS_AgingBucket"].fillna("")
    out["line_status"]                     = df["LineStatus"].fillna("")
    out["is_voided:int"]                   = df["isVoided"].fillna(0)
    out["is_hold:int"]                     = df["isHold"].fillna(0)
    out["transfer_flag:int"]               = df["TransferFlag"].fillna(0)
    out["charge_unit:float"]               = df["ChargeUnit"].fillna("")
    out["payment_plan_present:boolean"]    = df["Payment_Plan_Present"].fillna(False)
    out["icd10_1"]                         = df["ICD10Diagnosis1"].fillna("")
    out["icd10_2"]                         = df["ICD10Diagnosis2"].fillna("")
    out["icd10_3"]                         = df["ICD10Diagnosis3"].fillna("")
    out["icd10_4"]                         = df["ICD10Diagnosis4"].fillna("")
    out["icd10_5"]                         = df["ICD10Diagnosis5"].fillna("")
    out["patient_id:long"]                 = df["PatientID"].fillna("")
    out["visit_id"]                        = df["VisitID"].fillna("")
    out[":LABEL"]                          = "Charge"

    write_csv(out, out_dir / "nodes_charge.csv", "Charge", len(out))


def export_transactions(data_dir: Path, out_dir: Path):
    logger.info("Exporting Transaction nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/transactions.parquet")

    out = pd.DataFrame()
    out["transactionId:ID(Transaction)"]  = tid(df["Source_Database_Code"], df["PaymentID"])
    out["source_db"]                      = df["Source_Database_Code"]
    out["payment_id"]                     = df["PaymentID"].fillna("")
    out["payment_amount:float"]           = df["PaymentAmount"].fillna("")
    out["adjustment_amount:float"]        = df["AdjustmentAmount"].fillna("")
    out["adjustment_bucket"]              = df["AdjustmentBucket"].fillna("")
    out["adjustment_type"]                = df["AdjustmentType"].fillna("")
    out["processing_type"]                = df["ProcessingType"].fillna("")
    out["post_date:datetime"]             = fmt_dt(df["PostDate"])
    out["balance_after_post:float"]       = df["BalanceAfterPost"].fillna("")
    out["allowed_amount:float"]           = df["AllowedAmount"].fillna("")
    out["bad_debt_adjustments:float"]     = df["BadDebtAdjustments"].fillna("")
    out["co_insurance_amount:float"]      = df["CoInsuranceAmount"].fillna("")
    out["deductible_amount:float"]        = df["DeductibleAmount"].fillna("")
    out["co_pay_amount:float"]            = df["CoPayAmount"].fillna("")
    out["denial_code:int"]                = df["DenialCode"].fillna(0)
    out["denial_note"]                    = df["DenialNote"].fillna("")
    out["paysource"]                      = df["Paysource"].fillna("")
    out["payment_method"]                 = df["PaymentMethod"].fillna("")
    out["transaction_type"]               = df["TransactionType"].fillna("")
    out["transfer_flag:int"]              = df["TransferFlag"].fillna(0)
    out["days_to_agency:float"]           = df["Days_to_Agency_Placement"].fillna("")
    out["charge_id"]                      = df["ChargeID"].fillna("")
    out[":LABEL"]                         = "Transaction"

    write_csv(out, out_dir / "nodes_transaction.csv", "Transaction", len(out))


def export_statements(data_dir: Path, out_dir: Path):
    logger.info("Exporting Statement nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/statements.parquet")

    out = pd.DataFrame()
    out["statementId:ID(Statement)"] = df["StatementID"]
    out["source_db"]                 = df["Source_Database_Code"]
    out["patient_balance:float"]     = df["PatientBalance"].fillna("")
    out["total_balance:float"]       = df["TotalBalance"].fillna("")
    out["statement_level"]           = df["StatementLevel"].fillna("")
    out["created_date:datetime"]     = fmt_dt(df["CreatedDate"])
    out["released_date:datetime"]    = fmt_dt(df["ReleasedDate"])
    out["is_released:int"]           = df["IsReleased"].fillna(0)
    out["is_on_hold:int"]            = df["IsOnHold"].fillna(0)
    out["email_successful"]          = df["Email_Successful"].fillna("")
    out["text_successful"]           = df["Text_Successful"].fillna("")
    out["patient_id:long"]           = df["PatientID"].fillna("")
    out[":LABEL"]                    = "Statement"

    write_csv(out, out_dir / "nodes_statement.csv", "Statement", len(out))


def export_rccalls(data_dir: Path, out_dir: Path):
    logger.info("Exporting RCCall nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/ringcentral.parquet")

    out = pd.DataFrame()
    out["rccallId:ID(RCCall)"] = df["Contact_ID"]
    out["campaign_name"]       = df["Campaign_Name"].fillna("")
    out["skill_name"]          = df["Skill_Name"].fillna("")
    out["agent_name"]          = df["Agent_Name"].fillna("")
    out["team_name"]           = df["Team_Name"].fillna("")
    out["start_date"]          = df["Start_Date"].fillna("")
    out["pre_queue:int"]       = df["PreQueue"].fillna(0)
    out["in_queue:int"]        = df["InQueue"].fillna(0)
    out["agent_time:int"]      = df["Agent_Time"].fillna(0)
    out["acw_time:int"]        = df["ACW_Time"].fillna(0)
    out["total_time:int"]      = df["Total_Time_Plus_Disposition"].fillna(0)
    out["abandon_time:int"]    = df["Abandon_Time"].fillna(0)
    out["abandon"]             = df["Abandon"].fillna("")
    out["sla:int"]             = df["SLA"].fillna(0)
    out["disp_name"]           = df["Disp_Name"].fillna("")
    out["hold_time:int"]       = df["Hold_Time"].fillna(0)
    out["ani_norm"]            = df["ANI_DIALNUM_norm"].fillna("")
    out["rc_attributable:boolean"] = df["_rc_attributable"].fillna(False)
    out["tags"]                = df["Tags"].fillna("")
    out[":LABEL"]              = "RCCall"

    write_csv(out, out_dir / "nodes_rccall.csv", "RCCall", len(out))


def export_ivr_inbound(data_dir: Path, out_dir: Path):
    logger.info("Exporting IVRInbound nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/rv_inbound.parquet")

    out = pd.DataFrame()
    out["ivrId:ID(IVRInbound)"]   = df["ResponseID"]
    out["response_id"]            = df["ResponseID"]
    out["account_id"]             = df["AccountID"].fillna("")
    out["balance:float"]          = df["Balance"].fillna("")
    out["amount_paid:float"]      = df["AmountPaid"].fillna("")
    out["ivr_type"]               = df["IVR"].fillna("")
    out["call_datetime:datetime"] = fmt_dt(df["CallDateTime"])
    out["call_duration:float"]    = df["CallDuration"].fillna("")
    out["auth_success:boolean"]   = df["AuthenticationSuccess"].fillna(False)
    out["result_desc"]            = df["ResultDesc"].fillna("")
    out["facility_code"]          = df["FacilityCode"].fillna("")
    out[":LABEL"]                 = "IVRInbound"

    write_csv(out, out_dir / "nodes_ivrinbound.csv", "IVRInbound", len(out))


def export_dialler(data_dir: Path, out_dir: Path):
    logger.info("Exporting DiallerCall nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/rv_outbound.parquet")

    out = pd.DataFrame()
    out["diallerId:ID(DiallerCall)"] = df["ACCOUNT"]
    out["account"]                   = df["ACCOUNT"]
    out["account_id"]                = df["ACCOUNTID"].fillna("")
    out["patient_balance:float"]     = df["PATIENTBALANCE"].fillna("")
    out["call_datetime:datetime"]    = fmt_dt(df["CALLDATETIME"])
    out["result_desc"]               = df["RESULTSDESC"].fillna("")
    out["service_loc"]               = df["SERVICELOC"].fillna("")
    out["phone_norm"]                = df["PATIENTPHONE_norm"].fillna("")
    out[":LABEL"]                    = "DiallerCall"

    write_csv(out, out_dir / "nodes_diallercall.csv", "DiallerCall", len(out))


def export_phone_bridge(data_dir: Path, out_dir: Path):
    logger.info("Exporting PhoneBridge nodes ...")
    df = pd.read_parquet(data_dir / "03_supplementary/phone_bridge.parquet")
    df = df.drop_duplicates(subset=["Source_Database_Code", "PatientID", "phone_norm"], keep="last")

    # Composite ID: src:patient_id:phone_norm
    pbid = (df["Source_Database_Code"].astype(str) + ":" +
            df["PatientID"].astype(str) + ":" +
            df["phone_norm"].astype(str))

    out = pd.DataFrame()
    out["phonebridgeId:ID(PhoneBridge)"] = pbid
    out["source_db"]                     = df["Source_Database_Code"]
    out["patient_id:long"]               = df["PatientID"]
    out["phone_norm"]                    = df["phone_norm"].fillna("")
    out["phone_type"]                    = df["phone_type"].fillna("")
    out["rc_call_count:float"]           = df["rc_call_count"].fillna(0)
    out["campaign_count:float"]          = df["campaign_count"].fillna(0)
    out["campaigns_contacted"]           = df["campaigns_contacted"].fillna("")
    out["primary_campaign"]              = df["primary_campaign"].fillna("")
    out[":LABEL"]                        = "PhoneBridge"

    write_csv(out, out_dir / "nodes_phonebridge.csv", "PhoneBridge", len(out))


def export_diagnosis_codes(data_dir: Path, out_dir: Path):
    logger.info("Exporting DiagnosisCode nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["ICD10Diagnosis1","ICD10Diagnosis2","ICD10Diagnosis3",
                                  "ICD10Diagnosis4","ICD10Diagnosis5"])

    codes = pd.concat([
        df["ICD10Diagnosis1"], df["ICD10Diagnosis2"], df["ICD10Diagnosis3"],
        df["ICD10Diagnosis4"], df["ICD10Diagnosis5"]
    ]).dropna().unique()

    out = pd.DataFrame()
    out["diagnosisId:ID(DiagnosisCode)"] = codes
    out["code"]                          = codes
    out[":LABEL"]                        = "DiagnosisCode"

    write_csv(out, out_dir / "nodes_diagnosiscode.csv", "DiagnosisCode", len(out))


def export_procedure_codes(data_dir: Path, out_dir: Path):
    logger.info("Exporting ProcedureCode nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["ProcedureCode","ProcedureDescription","ProcedureModality"])
    df = df.dropna(subset=["ProcedureCode"]).drop_duplicates(subset=["ProcedureCode"])

    out = pd.DataFrame()
    out["procedureId:ID(ProcedureCode)"] = df["ProcedureCode"]
    out["code"]                          = df["ProcedureCode"]
    out["description"]                   = df["ProcedureDescription"].fillna("")
    out["modality"]                      = df["ProcedureModality"].fillna("")
    out[":LABEL"]                        = "ProcedureCode"

    write_csv(out, out_dir / "nodes_procedurecode.csv", "ProcedureCode", len(out))


# ════════════════════════════════════════════════════════════════════
# RELATIONSHIP EXPORTERS
# ════════════════════════════════════════════════════════════════════

def export_rel_patient_practice(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "02_dims/patient.parquet",
                         columns=["Source_Database_Code","PatientID"])
    out = pd.DataFrame()
    out[":START_ID(Patient)"]  = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(Practice)"]   = df["Source_Database_Code"]
    out[":TYPE"]               = "REGISTERED_AT"
    write_csv(out, out_dir / "rel_patient_practice.csv", "REGISTERED_AT", len(out))


def export_rel_location_practice(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "02_dims/location.parquet",
                         columns=["Source_Database_Code","LocationID"])
    out = pd.DataFrame()
    out[":START_ID(Location)"] = lid(df["Source_Database_Code"], df["LocationID"])
    out[":END_ID(Practice)"]   = df["Source_Database_Code"]
    out[":TYPE"]               = "BELONGS_TO_PRACTICE"
    write_csv(out, out_dir / "rel_location_practice.csv", "BELONGS_TO_PRACTICE", len(out))


def export_rel_insurance_practice(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "02_dims/insurance.parquet",
                         columns=["Source_Database_Code","PlanNumber"])
    out = pd.DataFrame()
    out[":START_ID(InsurancePlan)"] = iid(df["Source_Database_Code"], df["PlanNumber"])
    out[":END_ID(Practice)"]        = df["Source_Database_Code"]
    out[":TYPE"]                    = "ISSUED_BY_PRACTICE"
    write_csv(out, out_dir / "rel_insurance_practice.csv", "ISSUED_BY_PRACTICE", len(out))


def export_rel_campaign_practice(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "03_supplementary/campaign_map.parquet")
    df = df.dropna(subset=["Source Database Code"])
    out = pd.DataFrame()
    out[":START_ID(Campaign)"]  = df["Campaign Name"]
    out[":END_ID(Practice)"]    = df["Source Database Code"]
    out[":TYPE"]                = "RUN_BY"
    write_csv(out, out_dir / "rel_campaign_practice.csv", "RUN_BY", len(out))


def export_rel_birdeye_location(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "02_dims/birdeye.parquet")
    df = df.drop_duplicates(subset=["Location", "Date Posted On"])

    bid = df["Location"].astype(str) + "::" + df["Date Posted On"].astype(str)

    # Match birdeye location name to location.name
    loc = pd.read_parquet(data_dir / "02_dims/location.parquet",
                          columns=["Source_Database_Code","LocationID","LocationName"])
    loc["location_node_id"] = lid(loc["Source_Database_Code"], loc["LocationID"])

    merged = df.merge(loc, left_on="Location", right_on="LocationName", how="inner")

    out = pd.DataFrame()
    bid_matched = merged["Location"].astype(str) + "::" + merged["Date Posted On"].astype(str)
    out[":START_ID(BirdeyeReview)"] = bid_matched
    out[":END_ID(Location)"]        = merged["location_node_id"]
    out[":TYPE"]                    = "REVIEWS"
    write_csv(out, out_dir / "rel_birdeye_location.csv", "REVIEWS", len(out))


def export_rel_patient_visit(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/visits.parquet",
                         columns=["Source_Database_Code","PatientID","VisitID"])
    df = df.drop_duplicates(subset=["Source_Database_Code","PatientID","VisitID"])
    out = pd.DataFrame()
    out[":START_ID(Patient)"] = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(Visit)"]     = vid(df["Source_Database_Code"], df["VisitID"])
    out[":TYPE"]              = "HAD_VISIT"
    write_csv(out, out_dir / "rel_patient_visit.csv", "HAD_VISIT", len(out))


def export_rel_visit_location(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/visits.parquet",
                         columns=["Source_Database_Code","VisitID","LocationID"])
    df = df.drop_duplicates(subset=["Source_Database_Code","VisitID"])
    out = pd.DataFrame()
    out[":START_ID(Visit)"]    = vid(df["Source_Database_Code"], df["VisitID"])
    out[":END_ID(Location)"]   = lid(df["Source_Database_Code"], df["LocationID"])
    out[":TYPE"]               = "PERFORMED_AT"
    write_csv(out, out_dir / "rel_visit_location.csv", "PERFORMED_AT", len(out))


def export_rel_visit_insurance(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/visits.parquet",
                         columns=["Source_Database_Code","VisitID","PrimaryInsurancePlanNum"])
    df = df.drop_duplicates(subset=["Source_Database_Code","VisitID"])
    df = df.dropna(subset=["PrimaryInsurancePlanNum"])
    df = df[df["PrimaryInsurancePlanNum"].astype(str).str.strip() != ""]
    out = pd.DataFrame()
    out[":START_ID(Visit)"]        = vid(df["Source_Database_Code"], df["VisitID"])
    out[":END_ID(InsurancePlan)"]  = iid(df["Source_Database_Code"], df["PrimaryInsurancePlanNum"])
    out["plan_type"]               = "primary"
    out[":TYPE"]                   = "UNDER_PLAN"
    write_csv(out, out_dir / "rel_visit_insurance.csv", "UNDER_PLAN", len(out))


def export_rel_patient_charge(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["Source_Database_Code","PatientID","ChargeID",
                                  "ChargeAmount","ServiceDate","ProcedureModality"])
    out = pd.DataFrame()
    out[":START_ID(Patient)"] = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(Charge)"]    = cid(df["Source_Database_Code"], df["ChargeID"])
    out["charge_amount:float"]= df["ChargeAmount"].fillna("")
    out["service_date:datetime"] = fmt_dt(df["ServiceDate"])
    out["modality"]           = df["ProcedureModality"].fillna("")
    out[":TYPE"]              = "HAS_CHARGE"
    write_csv(out, out_dir / "rel_patient_charge.csv", "HAS_CHARGE", len(out))


def export_rel_charge_visit(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["Source_Database_Code","ChargeID","VisitID"])
    df = df.dropna(subset=["VisitID"])
    df = df[df["VisitID"].astype(str).str.strip() != ""]
    out = pd.DataFrame()
    out[":START_ID(Charge)"]  = cid(df["Source_Database_Code"], df["ChargeID"])
    out[":END_ID(Visit)"]     = vid(df["Source_Database_Code"], df["VisitID"])
    out[":TYPE"]              = "PART_OF_VISIT"
    write_csv(out, out_dir / "rel_charge_visit.csv", "PART_OF_VISIT", len(out))


def export_rel_charge_location(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["Source_Database_Code","ChargeID","LocationID"])
    out = pd.DataFrame()
    out[":START_ID(Charge)"]  = cid(df["Source_Database_Code"], df["ChargeID"])
    out[":END_ID(Location)"]  = lid(df["Source_Database_Code"], df["LocationID"])
    out[":TYPE"]              = "AT_LOCATION"
    write_csv(out, out_dir / "rel_charge_location.csv", "AT_LOCATION", len(out))


def export_rel_charge_diagnosis(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["Source_Database_Code","ChargeID",
                                  "ICD10Diagnosis1","ICD10Diagnosis2","ICD10Diagnosis3",
                                  "ICD10Diagnosis4","ICD10Diagnosis5"])
    rows = []
    for col in ["ICD10Diagnosis1","ICD10Diagnosis2","ICD10Diagnosis3",
                "ICD10Diagnosis4","ICD10Diagnosis5"]:
        sub = df[["Source_Database_Code","ChargeID",col]].dropna(subset=[col])
        sub = sub[sub[col].astype(str).str.strip() != ""]
        rows.append(pd.DataFrame({
            "start": cid(sub["Source_Database_Code"], sub["ChargeID"]),
            "end":   sub[col].astype(str),
        }))
    merged = pd.concat(rows, ignore_index=True).drop_duplicates()

    out = pd.DataFrame()
    out[":START_ID(Charge)"]         = merged["start"]
    out[":END_ID(DiagnosisCode)"]    = merged["end"]
    out[":TYPE"]                     = "DIAGNOSED_WITH"
    write_csv(out, out_dir / "rel_charge_diagnosis.csv", "DIAGNOSED_WITH", len(out))


def export_rel_charge_procedure(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet",
                         columns=["Source_Database_Code","ChargeID","ProcedureCode"])
    df = df.dropna(subset=["ProcedureCode"])
    df = df[df["ProcedureCode"].astype(str).str.strip() != ""]
    out = pd.DataFrame()
    out[":START_ID(Charge)"]         = cid(df["Source_Database_Code"], df["ChargeID"])
    out[":END_ID(ProcedureCode)"]    = df["ProcedureCode"].astype(str)
    out[":TYPE"]                     = "USES_PROCEDURE"
    write_csv(out, out_dir / "rel_charge_procedure.csv", "USES_PROCEDURE", len(out))


def export_rel_transaction_charge(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/transactions.parquet",
                         columns=["Source_Database_Code","PaymentID","ChargeID",
                                  "PaymentAmount","AdjustmentAmount","AdjustmentBucket",
                                  "BadDebtAdjustments","BalanceAfterPost"])
    out = pd.DataFrame()
    out[":START_ID(Transaction)"]    = tid(df["Source_Database_Code"], df["PaymentID"])
    out[":END_ID(Charge)"]           = cid(df["Source_Database_Code"], df["ChargeID"])
    out["payment_amount:float"]      = df["PaymentAmount"].fillna("")
    out["adjustment_amount:float"]   = df["AdjustmentAmount"].fillna("")
    out["adjustment_bucket"]         = df["AdjustmentBucket"].fillna("")
    out["bad_debt:float"]            = df["BadDebtAdjustments"].fillna("")
    out["balance_after_post:float"]  = df["BalanceAfterPost"].fillna("")
    out[":TYPE"]                     = "SETTLES"
    write_csv(out, out_dir / "rel_transaction_charge.csv", "SETTLES", len(out))


def export_rel_patient_transaction(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/transactions.parquet",
                         columns=["Source_Database_Code","PatientID","PaymentID",
                                  "PaymentAmount","AdjustmentAmount","AdjustmentBucket"])
    out = pd.DataFrame()
    out[":START_ID(Patient)"]        = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(Transaction)"]      = tid(df["Source_Database_Code"], df["PaymentID"])
    out["payment_amount:float"]      = df["PaymentAmount"].fillna("")
    out["adjustment_amount:float"]   = df["AdjustmentAmount"].fillna("")
    out["bucket"]                    = df["AdjustmentBucket"].fillna("")
    out[":TYPE"]                     = "HAS_TRANSACTION"
    write_csv(out, out_dir / "rel_patient_transaction.csv", "HAS_TRANSACTION", len(out))


def export_rel_patient_statement(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/statements.parquet",
                         columns=["Source_Database_Code","PatientID","StatementID",
                                  "PatientBalance","TotalBalance","StatementLevel"])
    out = pd.DataFrame()
    out[":START_ID(Patient)"]    = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(Statement)"]    = df["StatementID"]
    out["patient_balance:float"] = df["PatientBalance"].fillna("")
    out["total_balance:float"]   = df["TotalBalance"].fillna("")
    out["level"]                 = df["StatementLevel"].fillna("")
    out[":TYPE"]                 = "RECEIVED_STATEMENT"
    write_csv(out, out_dir / "rel_patient_statement.csv", "RECEIVED_STATEMENT", len(out))


def export_rel_rccall_campaign(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/ringcentral.parquet",
                         columns=["Contact_ID","Campaign_Name"])
    df = df.dropna(subset=["Campaign_Name"])
    out = pd.DataFrame()
    out[":START_ID(RCCall)"]    = df["Contact_ID"]
    out[":END_ID(Campaign)"]    = df["Campaign_Name"]
    out[":TYPE"]                = "PART_OF_CAMPAIGN"
    write_csv(out, out_dir / "rel_rccall_campaign.csv", "PART_OF_CAMPAIGN", len(out))


def export_rel_rccall_phonebridge(data_dir: Path, out_dir: Path):
    rc = pd.read_parquet(data_dir / "01_facts/ringcentral.parquet",
                         columns=["Contact_ID","ANI_DIALNUM_norm","_rc_attributable"])
    rc = rc[rc["_rc_attributable"] == True]
    rc = rc.dropna(subset=["ANI_DIALNUM_norm"])

    pb = pd.read_parquet(data_dir / "03_supplementary/phone_bridge.parquet",
                         columns=["Source_Database_Code","PatientID","phone_norm"])
    pb = pb.drop_duplicates(subset=["Source_Database_Code","PatientID","phone_norm"])
    pbid = (pb["Source_Database_Code"].astype(str) + ":" +
            pb["PatientID"].astype(str) + ":" +
            pb["phone_norm"].astype(str))
    pb["pb_id"] = pbid

    merged = rc.merge(pb, left_on="ANI_DIALNUM_norm", right_on="phone_norm", how="inner")

    out = pd.DataFrame()
    out[":START_ID(RCCall)"]         = merged["Contact_ID"]
    out[":END_ID(PhoneBridge)"]      = merged["pb_id"]
    out[":TYPE"]                     = "ATTRIBUTED_TO_PHONE"
    write_csv(out, out_dir / "rel_rccall_phonebridge.csv", "ATTRIBUTED_TO_PHONE", len(out))


def export_rel_patient_phonebridge(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "03_supplementary/phone_bridge.parquet",
                         columns=["Source_Database_Code","PatientID","phone_norm"])
    df = df.drop_duplicates(subset=["Source_Database_Code","PatientID","phone_norm"])
    pbid = (df["Source_Database_Code"].astype(str) + ":" +
            df["PatientID"].astype(str) + ":" +
            df["phone_norm"].astype(str))
    out = pd.DataFrame()
    out[":START_ID(Patient)"]    = pid(df["Source_Database_Code"], df["PatientID"])
    out[":END_ID(PhoneBridge)"]  = pbid
    out[":TYPE"]                 = "IDENTIFIED_BY_PHONE"
    write_csv(out, out_dir / "rel_patient_phonebridge.csv", "IDENTIFIED_BY_PHONE", len(out))


def export_rel_patient_ivr(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/rv_inbound.parquet",
                         columns=["ResponseID","AccountID","Balance","AmountPaid","CallDateTime"])
    df = df.dropna(subset=["AccountID"])

    # AccountID = PatientID per DQ-003 — need to find which source_db
    pat = pd.read_parquet(data_dir / "02_dims/patient.parquet",
                          columns=["Source_Database_Code","PatientID"])
    pat["PatientID_str"] = pat["PatientID"].astype(str)

    df["AccountID_str"] = df["AccountID"].astype(str).str.replace(r"\.0$","",regex=True)
    merged = df.merge(pat, left_on="AccountID_str", right_on="PatientID_str", how="inner")

    out = pd.DataFrame()
    out[":START_ID(Patient)"]    = pid(merged["Source_Database_Code"], merged["PatientID"])
    out[":END_ID(IVRInbound)"]   = merged["ResponseID"]
    out["amount_paid:float"]     = merged["AmountPaid"].fillna("")
    out["balance:float"]         = merged["Balance"].fillna("")
    out["call_date:datetime"]    = fmt_dt(merged["CallDateTime"])
    out[":TYPE"]                 = "CALLED_IVR"
    write_csv(out, out_dir / "rel_patient_ivr.csv", "CALLED_IVR", len(out))


def export_rel_patient_dialler(data_dir: Path, out_dir: Path):
    df = pd.read_parquet(data_dir / "01_facts/rv_outbound.parquet",
                         columns=["ACCOUNT","ACCOUNTID","PATIENTBALANCE","CALLDATETIME","RESULTSDESC"])
    df = df.dropna(subset=["ACCOUNTID"])
    df["ACCOUNTID_str"] = df["ACCOUNTID"].astype(str).str.replace(r"\.0$","",regex=True)

    pat = pd.read_parquet(data_dir / "02_dims/patient.parquet",
                          columns=["Source_Database_Code","PatientID"])
    pat["PatientID_str"] = pat["PatientID"].astype(str)
    merged = df.merge(pat, left_on="ACCOUNTID_str", right_on="PatientID_str", how="inner")

    out = pd.DataFrame()
    out[":START_ID(Patient)"]       = pid(merged["Source_Database_Code"], merged["PatientID"])
    out[":END_ID(DiallerCall)"]     = merged["ACCOUNT"]
    out["patient_balance:float"]    = merged["PATIENTBALANCE"].fillna("")
    out["call_date:datetime"]       = fmt_dt(merged["CALLDATETIME"])
    out["result"]                   = merged["RESULTSDESC"].fillna("")
    out[":TYPE"]                    = "CONTACTED_BY_DIALLER"
    write_csv(out, out_dir / "rel_patient_dialler.csv", "CONTACTED_BY_DIALLER", len(out))


# ════════════════════════════════════════════════════════════════════
# IMPORT COMMAND GENERATOR
# ════════════════════════════════════════════════════════════════════

def write_import_command(out_dir: Path):
    """Write the full neo4j-admin import command to a shell script."""
    script = """#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SGS — RP Knowledge Graph
# neo4j-admin database import command
# Generated by parquet_to_neo4j_csv.py
#
# BEFORE RUNNING:
#   1. docker stop neo4j_rp
#   2. docker volume rm neo4j_data  (or clear the data folder)
#   3. Run this script
#   4. docker start neo4j_rp
#   5. python schema.py  (creates indexes — constraints already embedded)
# ──────────────────────────────────────────────────────────────────────────────

IMPORT_DIR="/var/lib/neo4j/import"   # path inside container
CONTAINER="neo4j_rp"

docker exec $CONTAINER neo4j-admin database import full neo4j \\
  --nodes=Patient=$IMPORT_DIR/nodes_patient.csv \\
  --nodes=Practice=$IMPORT_DIR/nodes_practice.csv \\
  --nodes=Location=$IMPORT_DIR/nodes_location.csv \\
  --nodes=InsurancePlan=$IMPORT_DIR/nodes_insurance.csv \\
  --nodes=Campaign=$IMPORT_DIR/nodes_campaign.csv \\
  --nodes=BirdeyeReview=$IMPORT_DIR/nodes_birdeye.csv \\
  --nodes=Visit=$IMPORT_DIR/nodes_visit.csv \\
  --nodes=Charge=$IMPORT_DIR/nodes_charge.csv \\
  --nodes=Transaction=$IMPORT_DIR/nodes_transaction.csv \\
  --nodes=Statement=$IMPORT_DIR/nodes_statement.csv \\
  --nodes=RCCall=$IMPORT_DIR/nodes_rccall.csv \\
  --nodes=IVRInbound=$IMPORT_DIR/nodes_ivrinbound.csv \\
  --nodes=DiallerCall=$IMPORT_DIR/nodes_diallercall.csv \\
  --nodes=PhoneBridge=$IMPORT_DIR/nodes_phonebridge.csv \\
  --nodes=DiagnosisCode=$IMPORT_DIR/nodes_diagnosiscode.csv \\
  --nodes=ProcedureCode=$IMPORT_DIR/nodes_procedurecode.csv \\
  --relationships=REGISTERED_AT=$IMPORT_DIR/rel_patient_practice.csv \\
  --relationships=BELONGS_TO_PRACTICE=$IMPORT_DIR/rel_location_practice.csv \\
  --relationships=ISSUED_BY_PRACTICE=$IMPORT_DIR/rel_insurance_practice.csv \\
  --relationships=RUN_BY=$IMPORT_DIR/rel_campaign_practice.csv \\
  --relationships=REVIEWS=$IMPORT_DIR/rel_birdeye_location.csv \\
  --relationships=HAD_VISIT=$IMPORT_DIR/rel_patient_visit.csv \\
  --relationships=PERFORMED_AT=$IMPORT_DIR/rel_visit_location.csv \\
  --relationships=UNDER_PLAN=$IMPORT_DIR/rel_visit_insurance.csv \\
  --relationships=HAS_CHARGE=$IMPORT_DIR/rel_patient_charge.csv \\
  --relationships=PART_OF_VISIT=$IMPORT_DIR/rel_charge_visit.csv \\
  --relationships=AT_LOCATION=$IMPORT_DIR/rel_charge_location.csv \\
  --relationships=DIAGNOSED_WITH=$IMPORT_DIR/rel_charge_diagnosis.csv \\
  --relationships=USES_PROCEDURE=$IMPORT_DIR/rel_charge_procedure.csv \\
  --relationships=SETTLES=$IMPORT_DIR/rel_transaction_charge.csv \\
  --relationships=HAS_TRANSACTION=$IMPORT_DIR/rel_patient_transaction.csv \\
  --relationships=RECEIVED_STATEMENT=$IMPORT_DIR/rel_patient_statement.csv \\
  --relationships=PART_OF_CAMPAIGN=$IMPORT_DIR/rel_rccall_campaign.csv \\
  --relationships=ATTRIBUTED_TO_PHONE=$IMPORT_DIR/rel_rccall_phonebridge.csv \\
  --relationships=IDENTIFIED_BY_PHONE=$IMPORT_DIR/rel_patient_phonebridge.csv \\
  --relationships=CALLED_IVR=$IMPORT_DIR/rel_patient_ivr.csv \\
  --relationships=CONTACTED_BY_DIALLER=$IMPORT_DIR/rel_patient_dialler.csv \\
  --skip-bad-relationships=true \\
  --skip-duplicate-nodes=true \\
  --high-io=true \\
  --verbose

echo "Import complete. Starting Neo4j..."
docker start $CONTAINER
echo "Done. Run: python schema.py to create indexes."
"""
    script_path = out_dir / "run_import.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)
    logger.info(f"\n  ✓ Import script written to: {script_path}")


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

def run(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    logger.info("=" * 62)
    logger.info("  SGS — RP Parquet → Neo4j CSV")
    logger.info(f"  Source: {data_dir}")
    logger.info(f"  Output: {out_dir}")
    logger.info("=" * 62)

    # ── Node CSVs ──────────────────────────────────────────────────
    logger.info("\n[1/3] Exporting node CSVs ...")
    export_patients(data_dir, out_dir)
    export_practices(data_dir, out_dir)
    export_locations(data_dir, out_dir)
    export_insurance(data_dir, out_dir)
    export_campaigns(data_dir, out_dir)
    export_birdeye(data_dir, out_dir)
    export_visits(data_dir, out_dir)
    export_charges(data_dir, out_dir)
    export_transactions(data_dir, out_dir)
    export_statements(data_dir, out_dir)
    export_rccalls(data_dir, out_dir)
    export_ivr_inbound(data_dir, out_dir)
    export_dialler(data_dir, out_dir)
    export_phone_bridge(data_dir, out_dir)
    export_diagnosis_codes(data_dir, out_dir)
    export_procedure_codes(data_dir, out_dir)

    # ── Relationship CSVs ──────────────────────────────────────────
    logger.info("\n[2/3] Exporting relationship CSVs ...")
    export_rel_patient_practice(data_dir, out_dir)
    export_rel_location_practice(data_dir, out_dir)
    export_rel_insurance_practice(data_dir, out_dir)
    export_rel_campaign_practice(data_dir, out_dir)
    export_rel_birdeye_location(data_dir, out_dir)
    export_rel_patient_visit(data_dir, out_dir)
    export_rel_visit_location(data_dir, out_dir)
    export_rel_visit_insurance(data_dir, out_dir)
    export_rel_patient_charge(data_dir, out_dir)
    export_rel_charge_visit(data_dir, out_dir)
    export_rel_charge_location(data_dir, out_dir)
    export_rel_charge_diagnosis(data_dir, out_dir)
    export_rel_charge_procedure(data_dir, out_dir)
    export_rel_transaction_charge(data_dir, out_dir)
    export_rel_patient_transaction(data_dir, out_dir)
    export_rel_patient_statement(data_dir, out_dir)
    export_rel_rccall_campaign(data_dir, out_dir)
    export_rel_rccall_phonebridge(data_dir, out_dir)
    export_rel_patient_phonebridge(data_dir, out_dir)
    export_rel_patient_ivr(data_dir, out_dir)
    export_rel_patient_dialler(data_dir, out_dir)

    # ── Summary + import script ────────────────────────────────────
    logger.info("\n[3/3] Writing import script ...")
    write_import_command(out_dir)

    elapsed = time.time() - t0
    csvs = list(out_dir.glob("*.csv"))
    total_size = sum(p.stat().st_size for p in csvs) / 1024 / 1024

    logger.info(f"\n{'='*62}")
    logger.info(f"  Done in {elapsed:.1f}s")
    logger.info(f"  Files:  {len(csvs)} CSV files")
    logger.info(f"  Size:   {total_size:.1f} MB")
    logger.info(f"\n  Next steps:")
    logger.info(f"  1. Copy {out_dir} to your Neo4j import folder:")
    logger.info(f"     cp {out_dir}/*.csv ~/work/codebase/RP/synthea-neo4j/dockers/import/")
    logger.info(f"  2. Stop Neo4j:  docker stop neo4j_rp")
    logger.info(f"  3. Clear data:  docker volume rm neo4j_data && docker volume create neo4j_data")
    logger.info(f"  4. Run import:  bash {out_dir}/run_import.sh")
    logger.info(f"{'='*62}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Convert RP parquets to neo4j-admin import CSVs"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to rp_dataset root (contains 00_navigation/, 01_facts/, 02_dims/)"
    )
    parser.add_argument(
        "--out", default="./neo4j_import",
        help="Output directory for CSV files (default: ./neo4j_import)"
    )
    args = parser.parse_args()
    run(Path(args.data), Path(args.out))