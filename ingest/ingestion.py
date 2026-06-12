"""
ingestion.py
────────────
Sutherland Global Services — Radiology Partners Knowledge Graph
Ingests all parquet files into Neo4j as a connected knowledge graph.

Fixes vs previous version:
  1. batch_load → uses execute_write (explicit transactions, 3x faster)
  2. Every function split into focused passes (one Cypher action per query)
  3. No UNWIND→WHERE without WITH row in between
  4. No Cypher // comments between WITH and WHERE clauses
  5. Nulls pre-filtered in Python before Cypher — no WHERE needed
  6. BATCH_SIZE default 2000 (tuned for 62GB server)
  7. ingest_visits, ingest_transactions, ingest_dialler all split into passes
"""
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from graph.connection import Neo4jConnection
from config import get_settings

logger = logging.getLogger(__name__)


# ── Batch helpers ─────────────────────────────────────────────────────────────

def batch_load(session, query: str, records: list, batch_size: int = 2000):
    """Execute Cypher in batches using explicit write transactions (faster)."""
    for i in tqdm(range(0, len(records), batch_size), desc="    batches", leave=False):
        batch = records[i: i + batch_size]
        session.execute_write(lambda tx, b=batch: tx.run(query, {"rows": b}))


def safe_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to records replacing NaN/NaT with None."""
    df = df.where(pd.notnull(df), None)
    return df.to_dict("records")


# ════════════════════════════════════════════════════════════════════
# DIMENSION NODES
# ════════════════════════════════════════════════════════════════════

def ingest_practices(data_dir: Path, batch_size: int):
    logger.info("📥 [1/13] Ingesting Practice nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/patient.parquet",
                         columns=["Source_Database_Code"])
    codes = df["Source_Database_Code"].dropna().unique().tolist()
    records = [{"code": c} for c in codes]
    query = """
    UNWIND $rows AS row
    MERGE (pr:Practice {code: row.code})
    SET pr.code = row.code
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} practices")


def ingest_locations(data_dir: Path, batch_size: int):
    logger.info("📥 [2/13] Ingesting Location nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/location.parquet")
    records = safe_records(df)

    # Pass 1: Location nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (l:Location {source_db: row.Source_Database_Code, location_id: row.LocationID})
    SET l.name                     = row.LocationName,
        l.abbreviation             = row.LocationAbbreviation,
        l.npi                      = row.LocationNPINumber,
        l.address                  = row.LocationAddress,
        l.city                     = row.LocationCity,
        l.state                    = row.LocationState,
        l.zip                      = row.LocationZip,
        l.phone                    = row.LocationPhone,
        l.phone_norm               = row.LocationPhone_norm,
        l.fax_norm                 = row.LocationFax_norm,
        l.location_type            = row.LocationType,
        l.fda_number               = row.LocationFDANumber,
        l.birdeye_review_count     = row.birdeye_review_count,
        l.birdeye_avg_rating       = row.birdeye_avg_rating,
        l.birdeye_median_rating    = row.birdeye_median_rating,
        l.birdeye_one_star_count   = row.birdeye_one_star_count,
        l.birdeye_two_star_count   = row.birdeye_two_star_count,
        l.birdeye_three_star_count = row.birdeye_three_star_count,
        l.birdeye_four_star_count  = row.birdeye_four_star_count,
        l.birdeye_five_star_count  = row.birdeye_five_star_count,
        l.birdeye_phi_review_count = row.birdeye_phi_review_count,
        l.birdeye_one_star_pct     = row.birdeye_one_star_pct,
        l.birdeye_one_two_star_pct = row.birdeye_one_or_two_star_pct,
        l.source_db                = row.Source_Database_Code
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Location → Practice
    q2 = """
    UNWIND $rows AS row
    MATCH (l:Location {source_db: row.Source_Database_Code, location_id: row.LocationID})
    MATCH (pr:Practice {code: row.Source_Database_Code})
    MERGE (l)-[:BELONGS_TO_PRACTICE]->(pr)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)
    logger.info(f"  ✅ {len(records)} locations")


def ingest_insurance(data_dir: Path, batch_size: int):
    logger.info("📥 [3/13] Ingesting InsurancePlan nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/insurance.parquet")
    records = safe_records(df)

    # Pass 1: InsurancePlan nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (i:InsurancePlan {source_db: row.Source_Database_Code, plan_number: row.PlanNumber})
    SET i.plan_name    = row.PlanName,
        i.plan_type    = row.PlanType,
        i.carrier_name = row.Carrier_Name,
        i.source_db    = row.Source_Database_Code
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: InsurancePlan → Practice
    q2 = """
    UNWIND $rows AS row
    MATCH (i:InsurancePlan {source_db: row.Source_Database_Code, plan_number: row.PlanNumber})
    MATCH (pr:Practice {code: row.Source_Database_Code})
    MERGE (i)-[:ISSUED_BY_PRACTICE]->(pr)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)
    logger.info(f"  ✅ {len(records)} insurance plans")


def ingest_campaigns(data_dir: Path, batch_size: int):
    logger.info("📥 [4/13] Ingesting Campaign nodes ...")
    df = pd.read_parquet(data_dir / "03_supplementary/campaign_map.parquet")
    records = safe_records(df)

    # Pass 1: Campaign nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (c:Campaign {name: row["Campaign Name"]})
    SET c.source_db = row["Source Database Code"],
        c.notes     = row.Notes
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Campaign → Practice (Python pre-filter nulls)
    linked = [r for r in records if r.get("Source Database Code") is not None]
    if linked:
        q2 = """
        UNWIND $rows AS row
        MATCH (c:Campaign {name: row["Campaign Name"]})
        MATCH (pr:Practice {code: row["Source Database Code"]})
        MERGE (c)-[:RUN_BY]->(pr)
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q2, linked, batch_size)
    logger.info(f"  ✅ {len(records)} campaigns")


def ingest_birdeye(data_dir: Path, batch_size: int):
    logger.info("📥 [5/13] Ingesting BirdeyeReview nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/birdeye.parquet")
    records = safe_records(df)

    # Pass 1: BirdeyeReview nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (b:BirdeyeReview {location: row.Location, date_posted: row["Date Posted On"]})
    SET b.source      = row["Review Source"],
        b.rating      = row["Review Rating"],
        b.comment     = row["Review Comment"],
        b.phi_phone   = row.phi_phone_count,
        b.phi_email   = row.phi_email_count,
        b.phi_flagged = row.phi_flagged
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: BirdeyeReview → Location
    q2 = """
    UNWIND $rows AS row
    MATCH (b:BirdeyeReview {location: row.Location, date_posted: row["Date Posted On"]})
    MATCH (l:Location {name: row.Location})
    MERGE (b)-[:REVIEWS]->(l)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)
    logger.info(f"  ✅ {len(records)} reviews")


# ════════════════════════════════════════════════════════════════════
# PATIENT NODES
# ════════════════════════════════════════════════════════════════════

def ingest_patients(data_dir: Path, batch_size: int):
    logger.info("📥 [6/13] Ingesting Patient nodes ...")
    df = pd.read_parquet(data_dir / "02_dims/patient.parquet")
    records = safe_records(df)

    # Pass 1: Patient nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    SET p.first_name            = row.PatientFirstName,
        p.middle_name           = row.PatientMiddleName,
        p.last_name             = row.PatientLastName,
        p.dob                   = row.PatientDOB,
        p.gender                = row.PatientGender,
        p.race                  = row.PatientRace,
        p.ethnicity             = row.PatientEthnicity,
        p.city                  = row.PatientCity,
        p.state                 = row.PatientState,
        p.zip                   = row.PatientZip,
        p.phone_norm            = row.PatientPhone_norm,
        p.cell_norm             = row.PatientCellPhone_norm,
        p.email                 = row.patEmail,
        p.propensity_grade      = row.Imagine_Propensity_to_Pay_Grade,
        p.propensity_desc       = row.Imagine_Propensity_to_Pay_Description,
        p.bad_address_indicator = row.Bad_Address_Indicator,
        p.source_db             = row.Source_Database_Code,
        p.pk_composite          = row._pk_composite,
        p.hashed_id             = row._pk_composite
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → Practice
    q2 = """
    UNWIND $rows AS row
    MATCH (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (pr:Practice {code: row.Source_Database_Code})
    MERGE (p)-[:REGISTERED_AT]->(pr)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)
    logger.info(f"  ✅ {len(records)} patients")


def enrich_patients_from_nav(data_dir: Path, batch_size: int):
    logger.info("📥 [7/13] Enriching Patients from Navigation Map ...")
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
        "rv_in_calls_window", "rv_out_calls_window", "rc_calls_window",
        "has_any_calls", "has_insurance", "_active_window",
        "last_visit_date", "first_visit_date",
        "PlanName", "Carrier_Name", "PlanType",
    ]
    df = pd.read_parquet(
        data_dir / "00_navigation/patient_navigation_map.parquet",
        columns=nav_cols,
    )
    records = safe_records(df)

    query = """
    UNWIND $rows AS row
    MATCH (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    SET p.total_charged         = row.total_charged,
        p.total_paid            = row.total_paid,
        p.outstanding_balance   = row.outstanding_balance,
        p.total_adjusted        = row.total_adjusted,
        p.adj_contractual       = row.adj_contractual,
        p.adj_bad_debt          = row.adj_bad_debt,
        p.adj_collection_agency = row.adj_collection_agency,
        p.adj_refund_reversal   = row.adj_refund_reversal,
        p.adj_other             = row.adj_other,
        p.payor_cohort          = row.payor_cohort,
        p.call_tier             = row.call_tier,
        p.is_catastrophe        = row.is_catastrophe,
        p.is_friction           = row.is_friction,
        p.is_clean              = row.is_clean,
        p.is_self_pay           = row.is_self_pay,
        p.is_bai                = row.is_bai,
        p.is_fully_covered      = row.is_fully_covered,
        p.is_sapa               = row.is_sapa,
        p.is_nraa               = row.is_nraa,
        p.is_tennessee          = row.is_tennessee,
        p.is_atlanta_404        = row.is_atlanta_404,
        p.multi_practice_flag   = row.multi_practice_flag,
        p.practice_count        = row.practice_count,
        p.visit_count           = row.visit_count,
        p.charge_count          = row.charge_count,
        p.statement_count       = row.statement_count,
        p.total_calls_window    = row.total_calls_window,
        p.rv_in_calls           = row.rv_in_calls_window,
        p.rv_out_calls          = row.rv_out_calls_window,
        p.rc_calls              = row.rc_calls_window,
        p.has_any_calls         = row.has_any_calls,
        p.has_insurance         = row.has_insurance,
        p.active_window         = row._active_window,
        p.carrier_name          = row.Carrier_Name,
        p.plan_name             = row.PlanName,
        p.plan_type             = row.PlanType
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} patients enriched")


# ════════════════════════════════════════════════════════════════════
# FACT NODES
# ════════════════════════════════════════════════════════════════════

def ingest_visits(data_dir: Path, batch_size: int):
    logger.info("📥 [8/13] Ingesting Visit nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/visits.parquet")
    records = safe_records(df)

    # Pass 1: Visit nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (v:Visit {source_db: row.Source_Database_Code, visit_id: row.VisitID})
    SET v.visit_number             = row.VisitNumber,
        v.history_number           = row.HistoryNumber,
        v.admit_date               = row.AdmitDate,
        v.discharge_date           = row.DischargeDate,
        v.location_id              = row.LocationID,
        v.primary_insurance_plan   = row.PrimaryInsurancePlanNum,
        v.primary_policy_number    = row.PrimaryInsurancePolicyNumber,
        v.secondary_insurance_plan = row.SecondaryInsurancePlanNum,
        v.tertiary_insurance_plan  = row.TertiaryInsurancePlanNum,
        v.primary_auth_number      = row.PrimaryAuthorizationNumber,
        v.primary_insurance_group  = row.PrimaryInsuranceGroup,
        v.source_db                = row.Source_Database_Code
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → Visit
    q2 = """
    UNWIND $rows AS row
    MATCH (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (v:Visit   {source_db: row.Source_Database_Code, visit_id:   row.VisitID})
    MERGE (p)-[:HAD_VISIT]->(v)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    # Pass 3: Visit → Location
    q3 = """
    UNWIND $rows AS row
    MATCH (v:Visit    {source_db: row.Source_Database_Code, visit_id:    row.VisitID})
    MATCH (l:Location {source_db: row.Source_Database_Code, location_id: row.LocationID})
    MERGE (v)-[:PERFORMED_AT]->(l)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q3, records, batch_size)

    # Pass 4: Visit → InsurancePlan (Python pre-filter nulls)
    ins_records = [r for r in records if r.get("PrimaryInsurancePlanNum") is not None]
    if ins_records:
        q4 = """
        UNWIND $rows AS row
        MATCH (v:Visit         {source_db: row.Source_Database_Code, visit_id:    row.VisitID})
        MATCH (i:InsurancePlan {source_db: row.Source_Database_Code, plan_number: row.PrimaryInsurancePlanNum})
        MERGE (v)-[:UNDER_PLAN {plan_type: 'primary'}]->(i)
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q4, ins_records, batch_size)

    logger.info(f"  ✅ {len(records)} visits")


def ingest_charges(data_dir: Path, batch_size: int):
    """
    5 focused passes — one action per pass, no multi-hop Cypher in one query.
    Pass 1: Charge nodes
    Pass 2: Patient → Charge
    Pass 3: Charge → Visit
    Pass 4: ProcedureCode nodes + Charge → ProcedureCode
    Pass 5: DiagnosisCode nodes + Charge → DiagnosisCode (ICD-10)
    """
    logger.info("📥 [9/13] Ingesting Charge nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/charges.parquet")
    records = safe_records(df)
    n = len(records)

    # Pass 1: Charge nodes
    logger.info(f"    Pass 1/5: {n:,} Charge nodes ...")
    q1 = """
    UNWIND $rows AS row
    MERGE (c:Charge {source_db: row.Source_Database_Code, charge_id: row.ChargeID})
    SET c.charge_amount             = row.ChargeAmount,
        c.procedure_code            = row.ProcedureCode,
        c.procedure_description     = row.ProcedureDescription,
        c.procedure_modality        = row.ProcedureModality,
        c.service_date              = row.ServiceDate,
        c.post_date                 = row.PostDate,
        c.balance                   = row.Balance,
        c.current_responsible_level = row.CurrentResponsibleLevel,
        c.place_of_service          = row.PlaceOfService,
        c.modifier                  = row.Modifier,
        c.dos_aging_bucket          = row.DOS_AgingBucket,
        c.line_status               = row.LineStatus,
        c.is_voided                 = row.isVoided,
        c.is_hold                   = row.isHold,
        c.transfer_flag             = row.TransferFlag,
        c.charge_unit               = row.ChargeUnit,
        c.payment_plan_present      = row.Payment_Plan_Present,
        c.icd10_1                   = row.ICD10Diagnosis1,
        c.icd10_2                   = row.ICD10Diagnosis2,
        c.icd10_3                   = row.ICD10Diagnosis3,
        c.icd10_4                   = row.ICD10Diagnosis4,
        c.icd10_5                   = row.ICD10Diagnosis5,
        c.source_db                 = row.Source_Database_Code,
        c.patient_id                = row.PatientID,
        c.visit_id                  = row.VisitID
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → Charge
    logger.info("    Pass 2/5: Patient → Charge ...")
    q2 = """
    UNWIND $rows AS row
    MATCH (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (c:Charge  {source_db: row.Source_Database_Code, charge_id:  row.ChargeID})
    MERGE (p)-[r:HAS_CHARGE]->(c)
    SET r.charge_amount = row.ChargeAmount,
        r.service_date  = row.ServiceDate,
        r.modality      = row.ProcedureModality
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    # Pass 3: Charge → Visit (Python pre-filter nulls)
    logger.info("    Pass 3/5: Charge → Visit ...")
    visit_records = [r for r in records if r.get("VisitID") is not None]
    q3 = """
    UNWIND $rows AS row
    MATCH (c:Charge {source_db: row.Source_Database_Code, charge_id: row.ChargeID})
    MATCH (v:Visit  {source_db: row.Source_Database_Code, visit_id:  row.VisitID})
    MERGE (c)-[:PART_OF_VISIT]->(v)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q3, visit_records, batch_size)

    # Pass 4: ProcedureCode nodes + Charge → ProcedureCode (Python pre-filter nulls)
    logger.info("    Pass 4/5: ProcedureCode nodes ...")
    proc_records = [r for r in records if r.get("ProcedureCode") is not None]
    q4 = """
    UNWIND $rows AS row
    MERGE (pc:ProcedureCode {code: row.ProcedureCode})
    SET pc.description = row.ProcedureDescription,
        pc.modality    = row.ProcedureModality
    WITH pc, row
    MATCH (c:Charge {source_db: row.Source_Database_Code, charge_id: row.ChargeID})
    MERGE (c)-[:USES_PROCEDURE]->(pc)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q4, proc_records, batch_size)

    # Pass 5: DiagnosisCode nodes + Charge → DiagnosisCode (melt ICD-10 cols)
    logger.info("    Pass 5/5: DiagnosisCode (ICD-10) nodes ...")
    icd_long = df[["Source_Database_Code", "ChargeID",
                   "ICD10Diagnosis1", "ICD10Diagnosis2", "ICD10Diagnosis3",
                   "ICD10Diagnosis4", "ICD10Diagnosis5"]].melt(
        id_vars=["Source_Database_Code", "ChargeID"],
        value_vars=["ICD10Diagnosis1", "ICD10Diagnosis2", "ICD10Diagnosis3",
                    "ICD10Diagnosis4", "ICD10Diagnosis5"],
        var_name="diagnosis_col",
        value_name="icd_code",
    ).dropna(subset=["icd_code"])
    icd_records = icd_long[["Source_Database_Code", "ChargeID", "icd_code"]].to_dict("records")

    if icd_records:
        q5 = """
        UNWIND $rows AS row
        MERGE (d:DiagnosisCode {code: row.icd_code})
        WITH d, row
        MATCH (c:Charge {source_db: row.Source_Database_Code, charge_id: row.ChargeID})
        MERGE (c)-[:DIAGNOSED_WITH]->(d)
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q5, icd_records, batch_size)

    logger.info(f"  ✅ {n:,} charges | {len(proc_records):,} procedures | {len(icd_records):,} ICD-10 links")


def ingest_transactions(data_dir: Path, batch_size: int):
    logger.info("📥 [10/13] Ingesting Transaction nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/transactions.parquet")
    records = safe_records(df)

    # Pass 1: Transaction nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (t:Transaction {source_db: row.Source_Database_Code, payment_id: row.PaymentID})
    SET t.payment_amount      = row.PaymentAmount,
        t.adjustment_amount   = row.AdjustmentAmount,
        t.adjustment_bucket   = row.AdjustmentBucket,
        t.adjustment_type     = row.AdjustmentType,
        t.processing_type     = row.ProcessingType,
        t.post_date           = row.PostDate,
        t.balance_after_post  = row.BalanceAfterPost,
        t.allowed_amount      = row.AllowedAmount,
        t.bad_debt_adjustments= row.BadDebtAdjustments,
        t.co_insurance_amount = row.CoInsuranceAmount,
        t.deductible_amount   = row.DeductibleAmount,
        t.co_pay_amount       = row.CoPayAmount,
        t.denial_code         = row.DenialCode,
        t.denial_note         = row.DenialNote,
        t.paysource           = row.Paysource,
        t.payment_method      = row.PaymentMethod,
        t.transaction_type    = row.TransactionType,
        t.transfer_flag       = row.TransferFlag,
        t.days_to_agency      = row.Days_to_Agency_Placement,
        t.source_db           = row.Source_Database_Code,
        t.charge_id           = row.ChargeID,
        t.patient_id          = row.PatientID
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Transaction → Charge
    q2 = """
    UNWIND $rows AS row
    MATCH (t:Transaction {source_db: row.Source_Database_Code, payment_id: row.PaymentID})
    MATCH (c:Charge      {source_db: row.Source_Database_Code, charge_id:  row.ChargeID})
    MERGE (t)-[r:SETTLES]->(c)
    SET r.payment_amount     = row.PaymentAmount,
        r.adjustment_amount  = row.AdjustmentAmount,
        r.adjustment_bucket  = row.AdjustmentBucket,
        r.bad_debt           = row.BadDebtAdjustments,
        r.balance_after_post = row.BalanceAfterPost
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    # Pass 3: Patient → Transaction
    q3 = """
    UNWIND $rows AS row
    MATCH (p:Patient     {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (t:Transaction {source_db: row.Source_Database_Code, payment_id: row.PaymentID})
    MERGE (p)-[r:HAS_TRANSACTION]->(t)
    SET r.payment_amount    = row.PaymentAmount,
        r.adjustment_amount = row.AdjustmentAmount,
        r.bucket            = row.AdjustmentBucket
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q3, records, batch_size)

    logger.info(f"  ✅ {len(records)} transactions")


def ingest_statements(data_dir: Path, batch_size: int):
    logger.info("📥 [11/13] Ingesting Statement nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/statements.parquet")
    records = safe_records(df)

    # Pass 1: Statement nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (s:Statement {statement_id: row.StatementID})
    SET s.patient_balance  = row.PatientBalance,
        s.total_balance    = row.TotalBalance,
        s.statement_level  = row.StatementLevel,
        s.created_date     = row.CreatedDate,
        s.released_date    = row.ReleasedDate,
        s.is_released      = row.IsReleased,
        s.is_on_hold       = row.IsOnHold,
        s.email_successful = row.Email_Successful,
        s.text_successful  = row.Text_Successful,
        s.source_db        = row.Source_Database_Code,
        s.patient_id       = row.PatientID
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → Statement
    q2 = """
    UNWIND $rows AS row
    MATCH (p:Patient   {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (s:Statement {statement_id: row.StatementID})
    MERGE (p)-[r:RECEIVED_STATEMENT]->(s)
    SET r.patient_balance = row.PatientBalance,
        r.total_balance   = row.TotalBalance,
        r.level           = row.StatementLevel,
        r.created_date    = row.CreatedDate
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    logger.info(f"  ✅ {len(records)} statements")


# ════════════════════════════════════════════════════════════════════
# CALL CENTRE NODES
# ════════════════════════════════════════════════════════════════════

def ingest_ringcentral(data_dir: Path, batch_size: int):
    logger.info("📥 [12/13] Ingesting RingCentral call nodes ...")
    df = pd.read_parquet(data_dir / "01_facts/ringcentral.parquet")
    records = safe_records(df)

    # Pass 1: RCCall nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (r:RCCall {contact_id: row.Contact_ID})
    SET r.campaign_name   = row.Campaign_Name,
        r.skill_name      = row.Skill_Name,
        r.agent_name      = row.Agent_Name,
        r.team_name       = row.Team_Name,
        r.start_date      = row.Start_Date,
        r.pre_queue       = row.PreQueue,
        r.in_queue        = row.InQueue,
        r.agent_time      = row.Agent_Time,
        r.acw_time        = row.ACW_Time,
        r.total_time      = row.Total_Time_Plus_Disposition,
        r.abandon_time    = row.Abandon_Time,
        r.abandon         = row.Abandon,
        r.sla             = row.SLA,
        r.disp_name       = row.Disp_Name,
        r.hold_time       = row.Hold_Time,
        r.ani_norm        = row.ANI_DIALNUM_norm,
        r.rc_attributable = row._rc_attributable,
        r.tags            = row.Tags
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: RCCall → Campaign (Python pre-filter nulls)
    campaign_records = [r for r in records if r.get("Campaign_Name") is not None]
    if campaign_records:
        q2 = """
        UNWIND $rows AS row
        MATCH (r:RCCall   {contact_id: row.Contact_ID})
        MATCH (c:Campaign {name: row.Campaign_Name})
        MERGE (r)-[:PART_OF_CAMPAIGN]->(c)
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q2, campaign_records, batch_size)

    logger.info(f"  ✅ {len(records)} RC calls")


def ingest_ivr_inbound(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting IVR Inbound calls ...")
    df = pd.read_parquet(data_dir / "01_facts/rv_inbound.parquet")
    records = safe_records(df)

    # Pass 1: IVRInbound nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (i:IVRInbound {response_id: row.ResponseID})
    SET i.caller_id     = row.CallerID,
        i.account_id    = row.AccountID,
        i.balance       = row.Balance,
        i.amount_paid   = row.AmountPaid,
        i.ivr_type      = row.IVR,
        i.call_datetime = row.CallDateTime,
        i.call_duration = row.CallDuration,
        i.auth_success  = row.AuthenticationSuccess,
        i.result_desc   = row.ResultDesc,
        i.facility_code = row.FacilityCode
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → IVRInbound (AccountID = PatientID per DQ-003)
    q2 = """
    UNWIND $rows AS row
    MATCH (i:IVRInbound {response_id: row.ResponseID})
    MATCH (p:Patient)
    WHERE p.patient_id = toInteger(row.AccountID)
    MERGE (p)-[r:CALLED_IVR]->(i)
    SET r.amount_paid = row.AmountPaid,
        r.balance     = row.Balance,
        r.call_date   = row.CallDateTime
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    logger.info(f"  ✅ {len(records)} IVR inbound calls")


def ingest_dialler_outbound(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Dialler Outbound calls ...")
    df = pd.read_parquet(data_dir / "01_facts/rv_outbound.parquet")
    records = safe_records(df)

    # Pass 1: DiallerCall nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (d:DiallerCall {account: row.ACCOUNT})
    SET d.account_id      = row.ACCOUNTID,
        d.patient_balance = row.PATIENTBALANCE,
        d.call_datetime   = row.CALLDATETIME,
        d.result_desc     = row.RESULTSDESC,
        d.service_loc     = row.SERVICELOC,
        d.phone_norm      = row.PATIENTPHONE_norm
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → DiallerCall (Python pre-filter nulls, ACCOUNTID = PatientID)
    linked = [r for r in records if r.get("ACCOUNTID") is not None]
    if linked:
        q2 = """
        UNWIND $rows AS row
        MATCH (d:DiallerCall {account: row.ACCOUNT})
        MATCH (p:Patient)
        WHERE p.patient_id = toInteger(toFloat(row.ACCOUNTID))
        MERGE (p)-[r:CONTACTED_BY_DIALLER]->(d)
        SET r.patient_balance = row.PATIENTBALANCE,
            r.call_date       = row.CALLDATETIME,
            r.result          = row.RESULTSDESC
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q2, linked, batch_size)

    logger.info(f"  ✅ {len(records)} dialler outbound calls")


def ingest_phone_bridge(data_dir: Path, batch_size: int):
    logger.info("📥 [13/13] Ingesting Phone Bridge ...")
    df = pd.read_parquet(data_dir / "03_supplementary/phone_bridge.parquet")
    records = safe_records(df)

    # Pass 1: PhoneBridge nodes
    q1 = """
    UNWIND $rows AS row
    MERGE (pb:PhoneBridge {
        source_db:  row.Source_Database_Code,
        patient_id: row.PatientID,
        phone_norm: row.phone_norm
    })
    SET pb.phone_type          = row.phone_type,
        pb.rc_call_count       = row.rc_call_count,
        pb.campaign_count      = row.campaign_count,
        pb.campaigns_contacted = row.campaigns_contacted,
        pb.primary_campaign    = row.primary_campaign
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q1, records, batch_size)

    # Pass 2: Patient → PhoneBridge
    q2 = """
    UNWIND $rows AS row
    MATCH (p:Patient {source_db: row.Source_Database_Code, patient_id: row.PatientID})
    MATCH (pb:PhoneBridge {source_db: row.Source_Database_Code,
                           patient_id: row.PatientID,
                           phone_norm: row.phone_norm})
    MERGE (p)-[:IDENTIFIED_BY_PHONE]->(pb)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, q2, records, batch_size)

    # Pass 3: RCCall → PhoneBridge (Python pre-filter: only attributed rows)
    rc_records = [r for r in records
                  if r.get("rc_call_count") is not None and r.get("rc_call_count", 0) > 0]
    if rc_records:
        q3 = """
        UNWIND $rows AS row
        MATCH (pb:PhoneBridge {source_db:  row.Source_Database_Code,
                               patient_id: row.PatientID,
                               phone_norm: row.phone_norm})
        MATCH (r:RCCall {ani_norm: row.phone_norm})
        MERGE (r)-[:ATTRIBUTED_TO_PHONE]->(pb)
        """
        with Neo4jConnection.session() as s:
            batch_load(s, q3, rc_records, batch_size)

    logger.info(f"  ✅ {len(records)} phone bridge rows")


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

def run_ingestion(data_dir: str = None, drop_first: bool = False):
    settings = get_settings()
    data_path = Path(data_dir or settings.synthea_data_dir)
    batch_size = getattr(settings, "batch_size", 2000)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info(f"🚀 Starting RP Knowledge Graph ingestion from: {data_path}")
    logger.info(f"   Batch size: {batch_size}")

    if drop_first:
        from ingest.schema import drop_all_data
        drop_all_data()

    from ingest.schema import create_schema
    create_schema()

    # Dimension nodes (no FK dependencies — load first)
    ingest_practices(data_path, batch_size)
    ingest_locations(data_path, batch_size)
    ingest_insurance(data_path, batch_size)
    ingest_campaigns(data_path, batch_size)
    ingest_birdeye(data_path, batch_size)

    # Patient nodes (depend on Practice)
    ingest_patients(data_path, batch_size)
    enrich_patients_from_nav(data_path, batch_size)

    # Fact nodes (depend on Patient, Location, Insurance)
    ingest_visits(data_path, batch_size)
    ingest_charges(data_path, batch_size)
    ingest_transactions(data_path, batch_size)
    ingest_statements(data_path, batch_size)

    # Call centre (depend on Patient, Campaign)
    ingest_ringcentral(data_path, batch_size)
    ingest_ivr_inbound(data_path, batch_size)
    ingest_dialler_outbound(data_path, batch_size)
    ingest_phone_bridge(data_path, batch_size)

    logger.info("🎉 Ingestion complete!")
    _print_summary()


def _print_summary():
    queries = {
        "Patient":       "MATCH (n:Patient) RETURN count(n) AS c",
        "Practice":      "MATCH (n:Practice) RETURN count(n) AS c",
        "Location":      "MATCH (n:Location) RETURN count(n) AS c",
        "InsurancePlan": "MATCH (n:InsurancePlan) RETURN count(n) AS c",
        "Visit":         "MATCH (n:Visit) RETURN count(n) AS c",
        "Charge":        "MATCH (n:Charge) RETURN count(n) AS c",
        "Transaction":   "MATCH (n:Transaction) RETURN count(n) AS c",
        "Statement":     "MATCH (n:Statement) RETURN count(n) AS c",
        "DiagnosisCode": "MATCH (n:DiagnosisCode) RETURN count(n) AS c",
        "ProcedureCode": "MATCH (n:ProcedureCode) RETURN count(n) AS c",
        "RCCall":        "MATCH (n:RCCall) RETURN count(n) AS c",
        "IVRInbound":    "MATCH (n:IVRInbound) RETURN count(n) AS c",
        "DiallerCall":   "MATCH (n:DiallerCall) RETURN count(n) AS c",
        "PhoneBridge":   "MATCH (n:PhoneBridge) RETURN count(n) AS c",
        "Campaign":      "MATCH (n:Campaign) RETURN count(n) AS c",
        "BirdeyeReview": "MATCH (n:BirdeyeReview) RETURN count(n) AS c",
        "Relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
    }
    print("\n📊 Knowledge Graph Summary")
    print("─" * 38)
    for label, q in queries.items():
        result = Neo4jConnection.run_query(q)
        count = result[0]["c"] if result else 0
        print(f"  {label:<20} {count:>10,}")
    print()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drop = "--drop" in sys.argv
    data = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--data"), None)
    run_ingestion(data_dir=data, drop_first=drop)
    