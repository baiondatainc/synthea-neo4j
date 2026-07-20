"""
schema.py
─────────
SGS — RP Knowledge Graph
Run AFTER neo4j-admin import to create all indexes and constraints.

Usage:
    python schema.py
    python schema.py --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025
"""

import argparse
from neo4j import GraphDatabase

# ── Index definitions ─────────────────────────────────────────────────────────
# Format: (label, property, index_type)
# index_type: "btree" for range/equality, "text" for CONTAINS/STARTS WITH

INDEXES = [
    # ── Patient — most queried node ───────────────────────────────────────────
    # dob: age filters, date range queries
    ("Patient", "dob",                  "btree"),
    # financial: WHERE outstanding_balance > X, ORDER BY
    ("Patient", "outstanding_balance",  "btree"),
    ("Patient", "total_charged",        "btree"),
    ("Patient", "adj_bad_debt",         "btree"),
    # cohort flags: WHERE is_self_pay = true (bitmap-style)
    ("Patient", "is_self_pay",          "btree"),
    ("Patient", "is_tennessee",         "btree"),
    ("Patient", "is_catastrophe",       "btree"),
    ("Patient", "is_friction",          "btree"),
    ("Patient", "is_clean",             "btree"),
    # geo: WHERE state = 'TX'
    ("Patient", "state",                "btree"),
    # lookup by business key
    ("Patient", "patient_id",           "btree"),
    ("Patient", "source_db",            "btree"),
    # payor segmentation
    ("Patient", "payor_cohort",         "btree"),
    ("Patient", "call_tier",            "btree"),
    # phone lookup for PhoneBridge join
    ("Patient", "phone_norm",           "btree"),

    # ── Charge — 474K nodes, heavily filtered ─────────────────────────────────
    # date range queries
    ("Charge",  "service_date",         "btree"),
    ("Charge",  "post_date",            "btree"),
    # WHERE balance > 0
    ("Charge",  "balance",              "btree"),
    ("Charge",  "charge_amount",        "btree"),
    # WHERE line_status = 'X'
    ("Charge",  "line_status",          "btree"),
    ("Charge",  "dos_aging_bucket",     "btree"),
    ("Charge",  "procedure_modality",   "btree"),

    # ── Transaction — 758K nodes ──────────────────────────────────────────────
    ("Transaction", "post_date",        "btree"),
    ("Transaction", "adjustment_bucket","btree"),
    ("Transaction", "paysource",        "btree"),
    ("Transaction", "days_to_agency",   "btree"),

    # ── Visit ─────────────────────────────────────────────────────────────────
    ("Visit",   "admit_date",           "btree"),
    ("Visit",   "discharge_date",       "btree"),

    # ── Statement ─────────────────────────────────────────────────────────────
    ("Statement", "created_date",       "btree"),
    ("Statement", "statement_level",    "btree"),

    # ── BirdeyeReview — date trend queries ───────────────────────────────────
    ("BirdeyeReview", "date_posted",    "btree"),   # now DATE_TIME after ETL fix
    ("BirdeyeReview", "rating",         "btree"),

    # ── Location ──────────────────────────────────────────────────────────────
    ("Location", "name",                "btree"),
    ("Location", "state",               "btree"),
    ("Location", "source_db",           "btree"),

    # ── IVRInbound / DiallerCall ──────────────────────────────────────────────
    ("IVRInbound",   "call_datetime",   "btree"),
    ("DiallerCall",  "call_datetime",   "btree"),

    # ── PhoneBridge ───────────────────────────────────────────────────────────
    ("PhoneBridge", "phone_norm",       "btree"),

    # ── RCCall ────────────────────────────────────────────────────────────────
    ("RCCall", "rc_attributable",       "btree"),
]

# ── Uniqueness constraints (also creates an index) ────────────────────────────
CONSTRAINTS = [
    ("Patient",       "patientId"),
    ("Practice",      "practiceId"),
    ("Location",      "locationId"),
    ("InsurancePlan", "insuranceId"),
    ("Visit",         "visitId"),
    ("Charge",        "chargeId"),
    ("Transaction",   "transactionId"),
    ("Statement",     "statementId"),
    ("RCCall",        "rccallId"),
    ("IVRInbound",    "ivrId"),
    ("DiallerCall",   "diallerId"),
    ("PhoneBridge",   "phonebridgeId"),
    ("Campaign",      "campaignId"),
    ("BirdeyeReview", "birdeyeId"),
    ("DiagnosisCode", "diagnosisId"),
    ("ProcedureCode", "procedureId"),
]


def run(uri: str, user: str, password: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        print("\n── Constraints (uniqueness + index) ──────────────────────────")
        for label, prop in CONSTRAINTS:
            name = f"constraint_{label.lower()}_{prop.lower()}"
            cypher = (
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
            session.run(cypher)
            print(f"  ✓ UNIQUE  {label}.{prop}")

        print("\n── Range / BTree indexes ─────────────────────────────────────")
        for label, prop, idx_type in INDEXES:
            name = f"idx_{label.lower()}_{prop.lower()}"
            cypher = (
                f"CREATE INDEX {name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{prop})"
            )
            session.run(cypher)
            print(f"  ✓ INDEX   {label}.{prop}")

        print("\n── Waiting for indexes to come online ────────────────────────")
        session.run("CALL db.awaitIndexes(300)")
        print("  ✓ All indexes online")

        print("\n── Index summary ─────────────────────────────────────────────")
        result = session.run(
            "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties "
            "WHERE state = 'ONLINE' RETURN name, type, labelsOrTypes, properties "
            "ORDER BY labelsOrTypes, properties"
        )
        for row in result:
            print(f"  {row['labelsOrTypes']} {row['properties']}  [{row['type']}]")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Neo4j indexes and constraints for RP graph")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default="rp_strong_pass_2025")
    args = parser.parse_args()
    run(args.uri, args.user, args.password)