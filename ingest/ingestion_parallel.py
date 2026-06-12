"""
ingestion_parallel.py
─────────────────────
Sutherland Global Services — Radiology Partners Knowledge Graph
Parallel ingestion using ThreadPoolExecutor.

Strategy:
  - Independent tables (no FK deps between them) run in parallel
  - Tables with FK dependencies run sequentially after their parents
  - 4 workers × batch_size=5000 = optimal for 62GB / 32GB Neo4j allocation

Execution order:
  Phase 1 (parallel): Practices, Locations, Insurance, Campaigns, Birdeye
  Phase 2 (parallel): Patients + nav enrichment (after Phase 1)
  Phase 3 (parallel): Visits, Charges, Transactions, Statements (after Phase 2)
  Phase 4 (parallel): RC Calls, IVR, Dialler, PhoneBridge (after Phase 2+3)

Usage:
  python ingestion_parallel.py
  python ingestion_parallel.py --drop
  python ingestion_parallel.py --workers 4 --batch 5000
"""
import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_settings
from graph.connection import Neo4jConnection
from ingest.ingestion import (
    ingest_practices, ingest_locations, ingest_insurance,
    ingest_campaigns, ingest_birdeye,
    ingest_patients, enrich_patients_from_nav,
    ingest_visits, ingest_charges, ingest_transactions, ingest_statements,
    ingest_ringcentral, ingest_ivr_inbound, ingest_dialler_outbound,
    ingest_phone_bridge, _print_summary,
)

logger = logging.getLogger(__name__)


def run_parallel(fns: list, data_path: Path, batch_size: int,
                 workers: int, phase_name: str):
    """
    Run a list of ingest functions in parallel using a thread pool.
    Each function gets its own Neo4j session (connection pool handles it).
    """
    logger.info(f"  ⚡ {phase_name} — running {len(fns)} functions with {workers} workers ...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fn, data_path, batch_size): fn.__name__
            for fn in fns
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"    ❌ {name} failed: {e}", exc_info=True)
                raise

    elapsed = time.time() - t0
    logger.info(f"  ✅ {phase_name} complete in {elapsed:.1f}s")


def run_ingestion_parallel(data_dir: str = None,
                            drop_first: bool = False,
                            workers: int = 4,
                            batch_size: int = 5000):

    settings = get_settings()
    data_path = Path(data_dir or settings.synthea_data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info("=" * 60)
    logger.info("  Sutherland Global Services — RP Knowledge Graph")
    logger.info(f"  Data:       {data_path}")
    logger.info(f"  Batch size: {batch_size:,}")
    logger.info(f"  Workers:    {workers}")
    logger.info("=" * 60)

    if drop_first:
        from ingest.schema import drop_all_data
        drop_all_data()

    from ingest.schema import create_schema
    create_schema()

    t_total = time.time()

    # ── Phase 1: Dimensions (no FK deps — fully parallel) ─────────────────────
    logger.info("\n[Phase 1/4] Dimension nodes ...")
    run_parallel(
        fns=[ingest_practices, ingest_locations,
             ingest_insurance, ingest_campaigns, ingest_birdeye],
        data_path=data_path,
        batch_size=batch_size,
        workers=min(workers, 5),
        phase_name="Phase 1 — Dimensions",
    )

    # ── Phase 2: Patients (depend on Practice — sequential) ───────────────────
    logger.info("\n[Phase 2/4] Patient nodes ...")
    run_parallel(
        fns=[ingest_patients],
        data_path=data_path,
        batch_size=batch_size,
        workers=1,
        phase_name="Phase 2a — Patient nodes",
    )
    # Nav enrichment must run after patient nodes exist
    run_parallel(
        fns=[enrich_patients_from_nav],
        data_path=data_path,
        batch_size=batch_size,
        workers=1,
        phase_name="Phase 2b — Patient enrichment",
    )

    # ── Phase 3: Fact nodes (parallel — all depend on Patient+Location) ───────
    logger.info("\n[Phase 3/4] Fact nodes ...")
    run_parallel(
        fns=[ingest_visits, ingest_charges,
             ingest_transactions, ingest_statements],
        data_path=data_path,
        batch_size=batch_size,
        workers=min(workers, 4),
        phase_name="Phase 3 — Facts",
    )

    # ── Phase 4: Call centre (parallel — depend on Patient+Campaign) ──────────
    logger.info("\n[Phase 4/4] Call centre nodes ...")
    run_parallel(
        fns=[ingest_ringcentral, ingest_ivr_inbound,
             ingest_dialler_outbound, ingest_phone_bridge],
        data_path=data_path,
        batch_size=batch_size,
        workers=min(workers, 4),
        phase_name="Phase 4 — Call centre",
    )

    total_elapsed = time.time() - t_total
    logger.info(f"\n🎉 Ingestion complete in {total_elapsed:.1f}s "
                f"({total_elapsed/60:.1f} minutes)")
    _print_summary()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Parallel RP Knowledge Graph ingestion"
    )
    parser.add_argument("--drop",    action="store_true", help="Drop all data first")
    parser.add_argument("--workers", type=int, default=4,    help="Parallel workers (default: 4)")
    parser.add_argument("--batch",   type=int, default=5000, help="Batch size (default: 5000)")
    parser.add_argument("--data",    type=str, default=None,  help="Data directory path")
    args = parser.parse_args()

    run_ingestion_parallel(
        data_dir=args.data,
        drop_first=args.drop,
        workers=args.workers,
        batch_size=args.batch,
    )