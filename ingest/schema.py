"""
schema.py
─────────
Sutherland Global Services — Radiology Partners Knowledge Graph
Creates all Neo4j constraints and indexes for the RP parquet dataset.

Node labels:
  Patient       — from patient.parquet / patient_navigation_map.parquet
  Practice      — derived from Source_Database_Code (appears in every table)
  Location      — from location.parquet
  Visit         — from visits.parquet
  Charge        — from charges.parquet
  Transaction   — from transactions.parquet
  Statement     — from statements.parquet
  InsurancePlan — from insurance.parquet
  RCCall        — from ringcentral.parquet
  IVRInbound    — from rv_inbound.parquet
  DiallerCall   — from rv_outbound.parquet
  PhoneBridge   — from phone_bridge.parquet
  Campaign      — from campaign_map.parquet
  BirdeyeReview — from birdeye.parquet
  DiagnosisCode — derived from ICD10Diagnosis1-10 on charges
  ProcedureCode — derived from ProcedureCode on charges

Run:
  python schema.py
"""
import logging
from graph.connection import Neo4jConnection

logger = logging.getLogger(__name__)

# ─── Uniqueness constraints ────────────────────────────────────────────────────
# These also create backing indexes automatically
CONSTRAINTS = [
    # Core patient entities
    "CREATE CONSTRAINT patient_pk IF NOT EXISTS FOR (p:Patient) REQUIRE (p.source_db, p.patient_id) IS NODE KEY",
    "CREATE CONSTRAINT practice_pk IF NOT EXISTS FOR (pr:Practice) REQUIRE pr.code IS UNIQUE",
    "CREATE CONSTRAINT location_pk IF NOT EXISTS FOR (l:Location) REQUIRE (l.source_db, l.location_id) IS NODE KEY",

    # Clinical / financial facts
    "CREATE CONSTRAINT visit_pk IF NOT EXISTS FOR (v:Visit) REQUIRE (v.source_db, v.visit_id) IS NODE KEY",
    "CREATE CONSTRAINT charge_pk IF NOT EXISTS FOR (c:Charge) REQUIRE (c.source_db, c.charge_id) IS NODE KEY",
    "CREATE CONSTRAINT transaction_pk IF NOT EXISTS FOR (t:Transaction) REQUIRE (t.source_db, t.payment_id) IS NODE KEY",
    "CREATE CONSTRAINT statement_pk IF NOT EXISTS FOR (s:Statement) REQUIRE s.statement_id IS UNIQUE",

    # Insurance
    "CREATE CONSTRAINT insurance_pk IF NOT EXISTS FOR (i:InsurancePlan) REQUIRE (i.source_db, i.plan_number) IS NODE KEY",

    # Call centre
    "CREATE CONSTRAINT rccall_pk IF NOT EXISTS FOR (r:RCCall) REQUIRE r.contact_id IS UNIQUE",
    "CREATE CONSTRAINT ivr_pk IF NOT EXISTS FOR (i:IVRInbound) REQUIRE i.response_id IS UNIQUE",
    "CREATE CONSTRAINT dialler_pk IF NOT EXISTS FOR (d:DiallerCall) REQUIRE d.account IS UNIQUE",

    # Supplementary
    "CREATE CONSTRAINT phonebridge_pk IF NOT EXISTS FOR (pb:PhoneBridge) REQUIRE (pb.source_db, pb.patient_id, pb.phone_norm) IS NODE KEY",
    "CREATE CONSTRAINT campaign_pk IF NOT EXISTS FOR (c:Campaign) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT diagnosis_pk IF NOT EXISTS FOR (d:DiagnosisCode) REQUIRE d.code IS UNIQUE",
    "CREATE CONSTRAINT procedure_code_pk IF NOT EXISTS FOR (p:ProcedureCode) REQUIRE p.code IS UNIQUE",
    "CREATE CONSTRAINT birdeye_pk IF NOT EXISTS FOR (b:BirdeyeReview) REQUIRE (b.location, b.date_posted) IS NODE KEY",
]

# ─── Additional indexes for query performance ──────────────────────────────────
INDEXES = [
    # Patient lookups
    "CREATE INDEX patient_gender  IF NOT EXISTS FOR (p:Patient) ON (p.gender)",
    "CREATE INDEX patient_state   IF NOT EXISTS FOR (p:Patient) ON (p.state)",
    "CREATE INDEX patient_cohort  IF NOT EXISTS FOR (p:Patient) ON (p.payor_cohort)",
    "CREATE INDEX patient_tier    IF NOT EXISTS FOR (p:Patient) ON (p.call_tier)",
    "CREATE INDEX patient_cat     IF NOT EXISTS FOR (p:Patient) ON (p.is_catastrophe)",
    "CREATE INDEX patient_sp      IF NOT EXISTS FOR (p:Patient) ON (p.is_self_pay)",
    "CREATE INDEX patient_name    IF NOT EXISTS FOR (p:Patient) ON (p.last_name, p.first_name)",

    # Visit lookups
    "CREATE INDEX visit_admit     IF NOT EXISTS FOR (v:Visit) ON (v.admit_date)",
    "CREATE INDEX visit_location  IF NOT EXISTS FOR (v:Visit) ON (v.location_id)",

    # Charge lookups
    "CREATE INDEX charge_service  IF NOT EXISTS FOR (c:Charge) ON (c.service_date)",
    "CREATE INDEX charge_proc     IF NOT EXISTS FOR (c:Charge) ON (c.procedure_code)",
    "CREATE INDEX charge_amount   IF NOT EXISTS FOR (c:Charge) ON (c.charge_amount)",
    "CREATE INDEX charge_aging    IF NOT EXISTS FOR (c:Charge) ON (c.dos_aging_bucket)",
    "CREATE INDEX charge_modality IF NOT EXISTS FOR (c:Charge) ON (c.procedure_modality)",

    # Transaction lookups
    "CREATE INDEX txn_postdate    IF NOT EXISTS FOR (t:Transaction) ON (t.post_date)",
    "CREATE INDEX txn_bucket      IF NOT EXISTS FOR (t:Transaction) ON (t.adjustment_bucket)",
    "CREATE INDEX txn_type        IF NOT EXISTS FOR (t:Transaction) ON (t.processing_type)",

    # Statement lookups
    "CREATE INDEX stmt_level      IF NOT EXISTS FOR (s:Statement) ON (s.statement_level)",
    "CREATE INDEX stmt_date       IF NOT EXISTS FOR (s:Statement) ON (s.created_date)",

    # Location lookups
    "CREATE INDEX loc_state       IF NOT EXISTS FOR (l:Location) ON (l.state)",
    "CREATE INDEX loc_type        IF NOT EXISTS FOR (l:Location) ON (l.location_type)",

    # Insurance lookups
    "CREATE INDEX ins_carrier     IF NOT EXISTS FOR (i:InsurancePlan) ON (i.carrier_name)",
    "CREATE INDEX ins_type        IF NOT EXISTS FOR (i:InsurancePlan) ON (i.plan_type)",

    # Call centre lookups
    "CREATE INDEX rc_date         IF NOT EXISTS FOR (r:RCCall) ON (r.start_date)",
    "CREATE INDEX rc_campaign     IF NOT EXISTS FOR (r:RCCall) ON (r.campaign_name)",
    "CREATE INDEX rc_agent        IF NOT EXISTS FOR (r:RCCall) ON (r.agent_name)",
    "CREATE INDEX rc_ani          IF NOT EXISTS FOR (r:RCCall) ON (r.ani_norm)",
    "CREATE INDEX ivr_datetime    IF NOT EXISTS FOR (i:IVRInbound) ON (i.call_datetime)",
    "CREATE INDEX dialler_dt      IF NOT EXISTS FOR (d:DiallerCall) ON (d.call_datetime)",

    # Phone bridge
    "CREATE INDEX pb_phone        IF NOT EXISTS FOR (pb:PhoneBridge) ON (pb.phone_norm)",

    # Diagnosis / Procedure codes
    "CREATE FULLTEXT INDEX diagnosis_ft IF NOT EXISTS FOR (d:DiagnosisCode) ON EACH [d.code]",
    "CREATE FULLTEXT INDEX procedure_ft IF NOT EXISTS FOR (p:ProcedureCode) ON EACH [p.code]",
]

INDEX_NAMES = [stmt.split()[2] for stmt in INDEXES]


def create_constraints():
    logger.info("Creating RP Knowledge Graph constraints ...")
    for stmt in CONSTRAINTS:
        try:
            Neo4jConnection.run_query(stmt)
            logger.info(f"  ✓ Constraint: {stmt[20:70]}...")
        except Exception as e:
            logger.warning(f"  ⚠ {e}")
    logger.info("✅ Constraints ready")


def create_indexes():
    logger.info("Creating RP Knowledge Graph indexes ...")
    for stmt in INDEXES:
        try:
            Neo4jConnection.run_query(stmt)
            logger.info(f"  ✓ Index: {stmt[13:70]}...")
        except Exception as e:
            logger.warning(f"  ⚠ {e}")
    logger.info("✅ Indexes ready")


def drop_indexes():
    logger.info("Dropping non-constraint indexes before ingestion ...")
    for name in INDEX_NAMES:
        try:
            Neo4jConnection.run_query(f"DROP INDEX {name} IF EXISTS")
            logger.info(f"  ✓ Dropped index: {name}")
        except Exception as e:
            logger.warning(f"  ⚠ Failed to drop index {name}: {e}")


def create_schema():
    create_constraints()
    create_indexes()


def drop_all_data():
    """WARNING: wipes entire graph. Use --drop flag only."""
    logger.warning("⚠️  Dropping ALL data from Neo4j ...")
    Neo4jConnection.run_query("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
    logger.info("✅ All data cleared")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    create_schema()