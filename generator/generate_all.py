"""
generate_all.py
───────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Master orchestrator. Runs all generators in correct dependency order.

Dependency graph:
  constants.py ─────────────────────────────────── (shared config)
  helpers.py   ─────────────────────────────────── (shared utilities)
       │
       ▼
  gen_dims.py          → location, insurance, birdeye
       │
       ▼
  gen_patients.py      → patient, patient_navigation_map
    (needs: location)
       │
       ├──────────────────────────────────────────────────────┐
       ▼                                                       ▼
  gen_facts.py                                           gen_calls.py
    → visits (needs: patient, location, insurance)         → ringcentral (needs: patient)
    → charges (needs: patient, visits, location)           → rv_inbound  (needs: patient)
    → transactions (needs: patient, charges)               → rv_outbound (needs: patient)
    → statements (needs: patient)
       │                                                       │
       └──────────────────┬────────────────────────────────────┘
                          ▼
               gen_supplementary.py
                 → phone_bridge (needs: patient, ringcentral)
                 → campaign_map (standalone reference)

Output folder structure (mirrors prod):
  output/
    00_navigation/  patient_navigation_map.parquet
    01_facts/       charges, transactions, visits, statements,
                    ringcentral, rv_inbound, rv_outbound
    02_dims/        patient, insurance, location, birdeye
    03_supplementary/ phone_bridge, campaign_map

Usage:
    python generate_all.py                  # default 5,000 patients
    python generate_all.py --n 1000         # quick smoke test
    python generate_all.py --n 50000        # larger dataset
    python generate_all.py --out ./my_data  # custom output path
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from constants import N_PATIENTS


def run_all(out_dir: Path, n_patients: int = N_PATIENTS) -> None:
    t0 = time.time()
    print("=" * 62)
    print("  Sutherland Global Services")
    print("  Radiology Partners — Synthetic Dataset Generator")
    print(f"  Patients: {n_patients:,} | Output: {out_dir}")
    print("=" * 62)

    # ── Step 1: Dimension tables ──────────────────────────────────
    print("\n[1/5] Generating dimension tables ...")
    from gen_dims import run as run_dims
    dim_results = run_dims(out_dir)
    location_df  = dim_results["location"]
    insurance_df = dim_results["insurance"]

    # ── Step 2: Patient + Navigation Map ─────────────────────────
    print("\n[2/5] Generating patient tables ...")
    from gen_patients import run as run_patients
    pat_results  = run_patients(out_dir, n=n_patients, location_df=location_df)
    patient_df   = pat_results["patient"]
    nav_map_df   = pat_results["nav_map"]

    # ── Step 3: Core financial/clinical facts ────────────────────
    print("\n[3/5] Generating fact tables ...")
    from gen_facts import run as run_facts
    fact_results = run_facts(out_dir,
                             patient_df=patient_df,
                             location_df=location_df,
                             insurance_df=insurance_df)

    # ── Step 4: Call centre facts ────────────────────────────────
    print("\n[4/5] Generating call tables ...")
    from gen_calls import run as run_calls
    call_results = run_calls(out_dir, patient_df=patient_df)
    ringcentral_df = call_results["ringcentral"]

    # ── Step 5: Supplementary / reference tables ─────────────────
    print("\n[5/6] Generating supplementary tables ...")
    from gen_supplementary import run as run_supp
    run_supp(out_dir, patient_df=patient_df, ringcentral_df=ringcentral_df)

    # ── Step 6: Derived analytical tables ────────────────────────
    print("\n[6/6] Generating derived analytical tables ...")
    from gen_derived import run as run_derived
    run_derived(
        out_dir,
        patient_df=patient_df,
        visits_df=fact_results["visits"],
        charges_df=fact_results["charges"],
        transactions_df=fact_results["transactions"],
        statements_df=fact_results["statements"],
        rv_inbound_df=call_results["rv_inbound"],
        rv_outbound_df=call_results["rv_outbound"],
    )

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print("  Generation complete")
    print(f"  Time: {elapsed:.1f}s")
    print()

    all_parquets = list(out_dir.rglob("*.parquet"))
    total_size = sum(p.stat().st_size for p in all_parquets)

    print(f"  {'File':<55} {'Rows':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for p in sorted(all_parquets):
        try:
            df = pd.read_parquet(p)
            rel = p.relative_to(out_dir)
            print(f"  {str(rel):<55} {len(df):>10,}")
        except Exception as e:
            print(f"  {str(p.relative_to(out_dir)):<55}  [error: {e}]")

    print()
    print(f"  Total files: {len(all_parquets)}")
    print(f"  Total size:  {total_size / 1024 / 1024:.1f} MB")
    print("=" * 62)

    # ── Relationship integrity spot-check ─────────────────────────
    print("\nRunning referential integrity spot-checks ...")
    _spot_check(out_dir)


def _spot_check(out_dir: Path) -> None:
    """Quick FK checks to verify relationships are intact."""
    try:
        pat   = pd.read_parquet(out_dir / "02_dims"   / "patient.parquet")
        chg   = pd.read_parquet(out_dir / "01_facts"  / "charges.parquet")
        vis   = pd.read_parquet(out_dir / "01_facts"  / "visits.parquet")
        txn   = pd.read_parquet(out_dir / "01_facts"  / "transactions.parquet")
        nav   = pd.read_parquet(out_dir / "00_navigation" / "patient_navigation_map.parquet")
        pb    = pd.read_parquet(out_dir / "03_supplementary" / "phone_bridge.parquet")

        pat_keys = set(zip(pat["Source_Database_Code"], pat["PatientID"].astype(int)))

        def check_fk(name, df, src_col, pid_col, sample_n=1000):
            sample = df.sample(n=min(sample_n, len(df)), random_state=42)
            sample_keys = set(zip(sample[src_col], sample[pid_col].astype(int)))
            orphans = sample_keys - pat_keys
            pct = (1 - len(orphans) / max(len(sample_keys), 1)) * 100
            status = "✓" if pct == 100.0 else "⚠"
            print(f"  {status} {name:<35} {pct:.1f}% sample joinable to patient")

        check_fk("charges → patient",      chg,  "Source_Database_Code", "PatientID")
        check_fk("visits → patient",       vis,  "Source_Database_Code", "PatientID")
        check_fk("transactions → patient", txn,  "Source_Database_Code", "PatientID")
        check_fk("nav_map → patient",      nav,  "Source_Database_Code", "PatientID")
        check_fk("phone_bridge → patient", pb,   "Source_Database_Code", "PatientID")

        # Visit → Charge FK
        vis_ids  = set(vis["VisitID"].dropna().astype(str))
        chg_vis  = set(chg["VisitID"].dropna().astype(str))
        overlap  = len(chg_vis & vis_ids) / max(len(chg_vis), 1) * 100
        print(f"  {'✓' if overlap > 95 else '⚠'} {'charges.VisitID → visits.VisitID':<35} {overlap:.1f}% matched")

        # Charge → Transaction FK
        chg_ids  = set(chg["ChargeID"].dropna().astype(str))
        txn_chgs = set(txn["ChargeID"].dropna().astype(str))
        overlap2 = len(txn_chgs & chg_ids) / max(len(txn_chgs), 1) * 100
        print(f"  {'✓' if overlap2 > 95 else '⚠'} {'transactions.ChargeID → charges.ChargeID':<35} {overlap2:.1f}% matched")

        print("\n  All checks passed." if True else "")
    except Exception as e:
        print(f"  Spot-check error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sutherland / Radiology Partners — synthetic dataset generator"
    )
    parser.add_argument(
        "--out", default="./rp_synthetic_output",
        help="Output root directory (default: ./rp_synthetic_output)"
    )
    parser.add_argument(
        "--n", default=N_PATIENTS, type=int,
        help=f"Number of patients to generate (default: {N_PATIENTS})"
    )
    args = parser.parse_args()
    run_all(Path(args.out), n_patients=args.n)