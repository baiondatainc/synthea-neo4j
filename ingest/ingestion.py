"""
Ingests Synthea CSV files into Neo4j Aura as a knowledge graph.
Nodes: Patient, Encounter, Condition, Medication, Procedure, Provider, Organization, Observation
Relationships: HAS_ENCOUNTER, HAS_CONDITION, PRESCRIBED, HAD_PROCEDURE, PERFORMED_BY,
               BELONGS_TO, HAS_OBSERVATION, DURING_ENCOUNTER
"""
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

from graph.connection import Neo4jConnection
from config import get_settings

logger = logging.getLogger(__name__)


def batch_load(session, query: str, records: list, batch_size: int):
    """Execute a Cypher query in batches."""
    for i in tqdm(range(0, len(records), batch_size), desc="  Batches"):
        batch = records[i: i + batch_size]
        session.run(query, {"rows": batch})


def ingest_patients(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Patients...")
    df = pd.read_csv(data_dir / "patients.csv")
    df = df.fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (p:Patient {id: row.id})
    SET p.first       = row.first,
        p.last        = row.last,
        p.gender      = row.gender,
        p.birthdate   = row.birthdate,
        p.deathdate   = row.deathdate,
        p.race        = row.race,
        p.ethnicity   = row.ethnicity,
        p.city        = row.city,
        p.state       = row.state,
        p.county      = row.county,
        p.zip         = row.zip,
        p.lat         = toFloat(row.lat),
        p.lon         = toFloat(row.lon),
        p.healthcare_expenses = toFloat(coalesce(row.healthcare_expenses, '0')),
        p.healthcare_coverage = toFloat(coalesce(row.healthcare_coverage, '0'))
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} patients")


def ingest_organizations(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Organizations...")
    path = data_dir / "organizations.csv"
    if not path.exists():
        logger.warning("  ⚠ organizations.csv not found, skipping")
        return
    df = pd.read_csv(path).fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (o:Organization {id: row.id})
    SET o.name = row.name, o.city = row.city, o.state = row.state,
        o.zip = row.zip, o.phone = row.phone
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} organizations")


def ingest_providers(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Providers...")
    path = data_dir / "providers.csv"
    if not path.exists():
        logger.warning("  ⚠ providers.csv not found, skipping")
        return
    df = pd.read_csv(path).fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (pv:Provider {id: row.id})
    SET pv.name         = row.name,
        pv.gender       = row.gender,
        pv.speciality   = row.speciality,
        pv.city         = row.city,
        pv.state        = row.state
    WITH pv, row
    WHERE row.organization <> ''
    MATCH (o:Organization {id: row.organization})
    MERGE (pv)-[:BELONGS_TO]->(o)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} providers")


def ingest_encounters(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Encounters...")
    df = pd.read_csv(data_dir / "encounters.csv").fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (e:Encounter {id: row.id})
    SET e.start          = row.start,
        e.stop           = row.stop,
        e.encounterclass = row.encounterclass,
        e.code           = row.code,
        e.description    = row.description,
        e.base_encounter_cost     = toFloat(coalesce(row.base_encounter_cost, '0')),
        e.total_claim_cost        = toFloat(coalesce(row.total_claim_cost, '0')),
        e.payer_coverage          = toFloat(coalesce(row.payer_coverage, '0'))
    WITH e, row
    MATCH (p:Patient {id: row.patient})
    MERGE (p)-[:HAS_ENCOUNTER]->(e)
    WITH e, row
    WHERE row.provider <> ''
    MATCH (pv:Provider {id: row.provider})
    MERGE (e)-[:PERFORMED_BY]->(pv)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} encounters")


def ingest_conditions(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Conditions...")
    df = pd.read_csv(data_dir / "conditions.csv").fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (c:Condition {code: row.code})
    SET c.description = row.description
    WITH c, row
    MATCH (p:Patient {id: row.patient})
    MERGE (p)-[r:HAS_CONDITION]->(c)
    SET r.start = row.start, r.stop = row.stop
    WITH c, r, row
    WHERE row.encounter <> ''
    MATCH (e:Encounter {id: row.encounter})
    MERGE (c)-[:DIAGNOSED_IN]->(e)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} condition records")


def ingest_medications(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Medications...")
    df = pd.read_csv(data_dir / "medications.csv").fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (m:Medication {code: row.code})
    SET m.description = row.description
    WITH m, row
    MATCH (p:Patient {id: row.patient})
    MERGE (p)-[r:PRESCRIBED]->(m)
    SET r.start = row.start,
        r.stop  = row.stop,
        r.base_cost = toFloat(coalesce(row.base_cost, '0')),
        r.dispenses = toInteger(coalesce(row.dispenses, '0'))
    WITH m, r, row
    WHERE row.encounter <> ''
    MATCH (e:Encounter {id: row.encounter})
    MERGE (m)-[:PRESCRIBED_IN]->(e)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} medication records")


def ingest_procedures(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Procedures...")
    path = data_dir / "procedures.csv"
    if not path.exists():
        logger.warning("  ⚠ procedures.csv not found, skipping")
        return
    df = pd.read_csv(path).fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (pr:Procedure {code: row.code})
    SET pr.description = row.description
    WITH pr, row
    MATCH (p:Patient {id: row.patient})
    MERGE (p)-[r:HAD_PROCEDURE]->(pr)
    SET r.start = row.start, r.stop = row.stop,
        r.base_cost = toFloat(coalesce(row.base_cost, '0'))
    WITH pr, r, row
    WHERE row.encounter <> ''
    MATCH (e:Encounter {id: row.encounter})
    MERGE (pr)-[:PERFORMED_IN]->(e)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} procedure records")


def ingest_observations(data_dir: Path, batch_size: int):
    logger.info("📥 Ingesting Observations...")
    path = data_dir / "observations.csv"
    if not path.exists():
        logger.warning("  ⚠ observations.csv not found, skipping")
        return
    df = pd.read_csv(path).fillna("")
    records = df.rename(columns=str.lower).to_dict("records")

    query = """
    UNWIND $rows AS row
    MERGE (ob:Observation {code: row.code})
    SET ob.description = row.description,
        ob.category    = row.category,
        ob.units       = row.units
    WITH ob, row
    MATCH (p:Patient {id: row.patient})
    MERGE (p)-[r:HAS_OBSERVATION]->(ob)
    SET r.date  = row.date,
        r.value = row.value,
        r.type  = row.type
    WITH ob, r, row
    WHERE row.encounter <> ''
    MATCH (e:Encounter {id: row.encounter})
    MERGE (ob)-[:RECORDED_IN]->(e)
    """
    with Neo4jConnection.session() as s:
        batch_load(s, query, records, batch_size)
    logger.info(f"  ✅ {len(records)} observation records")


def run_ingestion(data_dir: str = None, drop_first: bool = False):
    settings = get_settings()
    data_path = Path(data_dir or settings.synthea_data_dir)
    batch_size = settings.batch_size

    if not data_path.exists():
        raise FileNotFoundError(f"Synthea data directory not found: {data_path}")

    logger.info(f"🚀 Starting ingestion from: {data_path}")

    if drop_first:
        from ingest.schema import drop_all_data
        drop_all_data()

    # Order matters — nodes before relationships
    ingest_patients(data_path, batch_size)
    ingest_organizations(data_path, batch_size)
    ingest_providers(data_path, batch_size)
    ingest_encounters(data_path, batch_size)
    ingest_conditions(data_path, batch_size)
    ingest_medications(data_path, batch_size)
    ingest_procedures(data_path, batch_size)
    ingest_observations(data_path, batch_size)

    logger.info("🎉 Ingestion complete!")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drop = "--drop" in sys.argv
    run_ingestion(drop_first=drop)
