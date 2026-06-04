"""
Creates Neo4j schema: constraints and indexes for Synthea data.
Run once before ingestion.
"""
import logging
from graph.connection import Neo4jConnection

logger = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT encounter_id IF NOT EXISTS FOR (e:Encounter) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT condition_code IF NOT EXISTS FOR (c:Condition) REQUIRE c.code IS UNIQUE",
    "CREATE CONSTRAINT medication_code IF NOT EXISTS FOR (m:Medication) REQUIRE m.code IS UNIQUE",
    "CREATE CONSTRAINT procedure_code IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.code IS UNIQUE",
    "CREATE CONSTRAINT provider_id IF NOT EXISTS FOR (pv:Provider) REQUIRE pv.id IS UNIQUE",
    "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT observation_code IF NOT EXISTS FOR (ob:Observation) REQUIRE ob.code IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX patient_name IF NOT EXISTS FOR (p:Patient) ON (p.last, p.first)",
    "CREATE INDEX patient_gender IF NOT EXISTS FOR (p:Patient) ON (p.gender)",
    "CREATE INDEX condition_desc IF NOT EXISTS FOR (c:Condition) ON (c.description)",
    "CREATE INDEX medication_desc IF NOT EXISTS FOR (m:Medication) ON (m.description)",
    "CREATE INDEX encounter_date IF NOT EXISTS FOR (e:Encounter) ON (e.start)",
    "CREATE INDEX encounter_class IF NOT EXISTS FOR (e:Encounter) ON (e.encounterclass)",
]


def create_schema():
    logger.info("Creating Neo4j schema (constraints + indexes)...")
    for stmt in CONSTRAINTS:
        try:
            Neo4jConnection.run_query(stmt)
            logger.info(f"  ✓ {stmt[:60]}...")
        except Exception as e:
            logger.warning(f"  ⚠ Constraint may already exist: {e}")

    for stmt in INDEXES:
        try:
            Neo4jConnection.run_query(stmt)
            logger.info(f"  ✓ {stmt[:60]}...")
        except Exception as e:
            logger.warning(f"  ⚠ Index may already exist: {e}")

    logger.info("✅ Schema ready")


def drop_all_data():
    """WARNING: Clears all nodes and relationships. Use for fresh reload."""
    logger.warning("⚠️  Dropping ALL data from Neo4j...")
    Neo4jConnection.run_query("MATCH (n) DETACH DELETE n")
    logger.info("✅ All data cleared")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_schema()
