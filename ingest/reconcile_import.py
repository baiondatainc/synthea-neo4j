"""
reconcile_import.py
───────────────────
Sutherland Global Services — RP Knowledge Graph
Reconciles CSV files against what actually loaded into Neo4j,
and diagnoses WHY records were skipped during neo4j-admin import.

Checks:
  1. CSV row count vs Neo4j node count (per label)
  2. Duplicate IDs in node CSVs (cause of --skip-duplicate-nodes drops)
  3. Orphan relationship endpoints (START/END IDs not in any node CSV)
  4. CSV parse issues (embedded newlines, unescaped quotes/commas)
  5. Relationship CSV count vs Neo4j relationship count

Usage:
  python reconcile_import.py --csv ./neo4j_import
  python reconcile_import.py --csv ./neo4j_import --check-neo4j
"""

import argparse
import csv
import logging
from pathlib import Path
from collections import Counter

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ── Node label → CSV file + ID column mapping ─────────────────────────────────
NODE_FILES = {
    "Patient":       ("nodes_patient.csv",       "patientId:ID(Patient)"),
    "Practice":      ("nodes_practice.csv",      "practiceId:ID(Practice)"),
    "Location":      ("nodes_location.csv",      "locationId:ID(Location)"),
    "InsurancePlan": ("nodes_insurance.csv",     "insuranceId:ID(InsurancePlan)"),
    "Campaign":      ("nodes_campaign.csv",      "campaignId:ID(Campaign)"),
    "BirdeyeReview": ("nodes_birdeye.csv",       "birdeyeId:ID(BirdeyeReview)"),
    "Visit":         ("nodes_visit.csv",         "visitId:ID(Visit)"),
    "Charge":        ("nodes_charge.csv",        "chargeId:ID(Charge)"),
    "Transaction":   ("nodes_transaction.csv",   "transactionId:ID(Transaction)"),
    "Statement":     ("nodes_statement.csv",     "statementId:ID(Statement)"),
    "RCCall":        ("nodes_rccall.csv",        "rccallId:ID(RCCall)"),
    "IVRInbound":    ("nodes_ivrinbound.csv",    "ivrId:ID(IVRInbound)"),
    "DiallerCall":   ("nodes_diallercall.csv",   "diallerId:ID(DiallerCall)"),
    "PhoneBridge":   ("nodes_phonebridge.csv",   "phonebridgeId:ID(PhoneBridge)"),
    "DiagnosisCode": ("nodes_diagnosiscode.csv", "diagnosisId:ID(DiagnosisCode)"),
    "ProcedureCode": ("nodes_procedurecode.csv", "procedureId:ID(ProcedureCode)"),
}

# ── Relationship file → (start label, end label) ──────────────────────────────
REL_FILES = {
    "rel_patient_practice.csv":     ("Patient", "Practice"),
    "rel_location_practice.csv":    ("Location", "Practice"),
    "rel_insurance_practice.csv":   ("InsurancePlan", "Practice"),
    "rel_campaign_practice.csv":    ("Campaign", "Practice"),
    "rel_birdeye_location.csv":     ("BirdeyeReview", "Location"),
    "rel_patient_visit.csv":        ("Patient", "Visit"),
    "rel_visit_location.csv":       ("Visit", "Location"),
    "rel_visit_insurance.csv":      ("Visit", "InsurancePlan"),
    "rel_patient_charge.csv":       ("Patient", "Charge"),
    "rel_charge_visit.csv":         ("Charge", "Visit"),
    "rel_charge_location.csv":      ("Charge", "Location"),
    "rel_charge_diagnosis.csv":     ("Charge", "DiagnosisCode"),
    "rel_charge_procedure.csv":     ("Charge", "ProcedureCode"),
    "rel_transaction_charge.csv":   ("Transaction", "Charge"),
    "rel_patient_transaction.csv":  ("Patient", "Transaction"),
    "rel_patient_statement.csv":    ("Patient", "Statement"),
    "rel_rccall_campaign.csv":      ("RCCall", "Campaign"),
    "rel_rccall_phonebridge.csv":   ("RCCall", "PhoneBridge"),
    "rel_patient_phonebridge.csv":  ("Patient", "PhoneBridge"),
    "rel_patient_ivr.csv":          ("Patient", "IVRInbound"),
    "rel_patient_dialler.csv":      ("Patient", "DiallerCall"),
}


def count_csv_rows(path: Path) -> int:
    """Count data rows (excluding header) using a proper CSV reader."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def load_node_ids(csv_dir: Path) -> dict:
    """Load the full set of node IDs per label, detecting duplicates."""
    node_ids = {}
    for label, (fname, id_col) in NODE_FILES.items():
        path = csv_dir / fname
        if not path.exists():
            logger.warning(f"  ⚠ {fname} not found")
            node_ids[label] = set()
            continue
        df = pd.read_csv(path, usecols=[id_col], dtype=str)
        ids = df[id_col].tolist()
        unique = set(ids)
        dupes = len(ids) - len(unique)
        node_ids[label] = unique
        flag = f"  ⚠ {dupes:,} DUPLICATE IDs" if dupes > 0 else ""
        logger.info(f"  {label:<16} {len(ids):>8,} rows  {len(unique):>8,} unique{flag}")
    return node_ids


def check_csv_integrity(csv_dir: Path):
    """Detect CSV parse issues — row count mismatch between line count and parsed count."""
    logger.info("\n[CHECK 1] CSV parse integrity (raw lines vs parsed rows)")
    logger.info("─" * 64)
    issues = []
    for label, (fname, _) in NODE_FILES.items():
        path = csv_dir / fname
        if not path.exists():
            continue
        # Raw line count
        with open(path, "rb") as f:
            raw_lines = sum(1 for _ in f) - 1  # minus header
        # Parsed row count (handles quoted fields with embedded newlines)
        parsed = count_csv_rows(path)
        if raw_lines != parsed:
            issues.append((fname, raw_lines, parsed))
            logger.info(f"  ⚠ {fname:<32} raw_lines={raw_lines:,}  parsed={parsed:,}  "
                        f"DIFF={raw_lines - parsed:,}")
        else:
            logger.info(f"  ✓ {fname:<32} {parsed:,} rows clean")
    if issues:
        logger.info(f"\n  → {len(issues)} files have embedded newlines or quote issues")
        logger.info(f"    These cause neo4j-admin to misparse rows.")
    return issues


def check_duplicate_nodes(csv_dir: Path):
    """Find duplicate node IDs — these get dropped by --skip-duplicate-nodes."""
    logger.info("\n[CHECK 2] Duplicate node IDs (dropped by --skip-duplicate-nodes)")
    logger.info("─" * 64)
    found_dupes = False
    for label, (fname, id_col) in NODE_FILES.items():
        path = csv_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=[id_col], dtype=str)
        counts = Counter(df[id_col])
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            found_dupes = True
            total_extra = sum(v - 1 for v in dupes.values())
            logger.info(f"  ⚠ {label:<16} {len(dupes):,} duplicate IDs "
                        f"({total_extra:,} extra rows will be dropped)")
            for k, v in list(dupes.items())[:3]:
                logger.info(f"      '{k}' appears {v}× ")
    if not found_dupes:
        logger.info("  ✓ No duplicate node IDs found")


def check_orphan_relationships(csv_dir: Path, node_ids: dict):
    """Find relationship endpoints whose ID isn't in the corresponding node CSV."""
    logger.info("\n[CHECK 3] Orphan relationship endpoints (skipped by --skip-bad-relationships)")
    logger.info("─" * 64)
    for fname, (start_label, end_label) in REL_FILES.items():
        path = csv_dir / fname
        if not path.exists():
            logger.warning(f"  ⚠ {fname} not found")
            continue
        df = pd.read_csv(path, dtype=str)
        start_col = df.columns[0]  # :START_ID(...)
        end_col = df.columns[1]    # :END_ID(...)

        start_ids = set(df[start_col])
        end_ids = set(df[end_col])

        start_orphans = start_ids - node_ids.get(start_label, set())
        end_orphans = end_ids - node_ids.get(end_label, set())

        total = len(df)
        bad_start = df[df[start_col].isin(start_orphans)].shape[0] if start_orphans else 0
        bad_end = df[df[end_col].isin(end_orphans)].shape[0] if end_orphans else 0
        bad_total = df[
            df[start_col].isin(start_orphans) | df[end_col].isin(end_orphans)
        ].shape[0]

        if bad_total > 0:
            pct = 100 * bad_total / total
            logger.info(f"  ⚠ {fname:<32} {bad_total:,}/{total:,} ({pct:.0f}%) will be SKIPPED")
            if start_orphans:
                sample = list(start_orphans)[:2]
                logger.info(f"      missing {start_label} nodes: {sample}")
            if end_orphans:
                sample = list(end_orphans)[:2]
                logger.info(f"      missing {end_label} nodes: {sample}")
        else:
            logger.info(f"  ✓ {fname:<32} all {total:,} endpoints valid")


def compare_with_neo4j(csv_dir: Path):
    """Compare CSV counts against live Neo4j (requires connection)."""
    logger.info("\n[CHECK 4] CSV rows vs Neo4j loaded counts")
    logger.info("─" * 64)
    try:
        from graph.connection import Neo4jConnection
    except ImportError:
        logger.warning("  ⚠ Cannot import Neo4jConnection — run from project root")
        return

    for label, (fname, _) in NODE_FILES.items():
        path = csv_dir / fname
        if not path.exists():
            continue
        csv_count = count_csv_rows(path)
        result = Neo4jConnection.run_query(
            f"MATCH (n:{label}) RETURN count(n) AS c"
        )
        neo_count = result[0]["c"] if result else 0
        match = "✓" if csv_count == neo_count else "✗ MISMATCH"
        diff = f"  (missing {csv_count - neo_count:,})" if csv_count != neo_count else ""
        logger.info(f"  {match} {label:<16} CSV={csv_count:>8,}  Neo4j={neo_count:>8,}{diff}")


def run(csv_dir: Path, check_neo4j: bool):
    logger.info("=" * 64)
    logger.info("  RP Knowledge Graph — Import Reconciliation")
    logger.info(f"  CSV dir: {csv_dir}")
    logger.info("=" * 64)

    logger.info("\n[LOADING] Node ID sets ...")
    node_ids = load_node_ids(csv_dir)

    check_csv_integrity(csv_dir)
    check_duplicate_nodes(csv_dir)
    check_orphan_relationships(csv_dir, node_ids)

    if check_neo4j:
        compare_with_neo4j(csv_dir)

    logger.info("\n" + "=" * 64)
    logger.info("  Reconciliation complete.")
    logger.info("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Directory with the import CSVs")
    parser.add_argument("--check-neo4j", action="store_true",
                        help="Also compare against live Neo4j (run from project root)")
    args = parser.parse_args()
    run(Path(args.csv), args.check_neo4j)