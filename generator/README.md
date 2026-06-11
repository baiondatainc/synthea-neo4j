# Radiology Partners — Synthetic Dataset Generator
**Sutherland Global Services | Healthcare Analytics**

Generates a fully relational synthetic parquet dataset that exactly matches
the schema of `rp_dataset_20260520`. Every column name, dtype, null rate,
and FK relationship is preserved. No real patient data is used.

---

## Module map

```
synthetic_data_generator/
  constants.py          Shared vocab, seeds, practice codes, state pools
  helpers.py            Pure generation functions (phone, address, dates, IDs)
  gen_dims.py           location, insurance, birdeye
  gen_patients.py       patient, patient_navigation_map (144-col master)
  gen_facts.py          visits, charges, transactions, statements
  gen_calls.py          ringcentral, rv_inbound, rv_outbound
  gen_supplementary.py  phone_bridge, campaign_map
  generate_all.py       Master orchestrator — run this
```

---

## Relationship diagram

```
campaign_map ──(Source Database Code)──┐
                                       │
location ──(Source_Database_Code,      │
            LocationID)────────────────┤
                                       │
insurance ──(Source_Database_Code,     │
             PlanNumber)───────────────┤
                                       ▼
patient ──(Source_Database_Code, PatientID) ◄─── PRIMARY KEY
    │                                               │
    ├──► visits      (Source_Database_Code,         │
    │                 PatientID, VisitID)            │
    │       │                                       │
    │       ▼                                       │
    ├──► charges     (Source_Database_Code,         │
    │                 PatientID, ChargeID)           │
    │       │                                       │
    │       ▼                                       │
    ├──► transactions(Source_Database_Code,         │
    │                 PatientID, ChargeID)           │
    │                                               │
    ├──► statements  (Source_Database_Code,         │
    │                 PatientID)                    │
    │                                               │
    ├──► rv_inbound  (AccountID = PatientID)        │
    │                                               │
    ├──► rv_outbound (ACCOUNTID = PatientID)        │
    │                                               │
    └──► phone_bridge(Source_Database_Code,         │
                      PatientID, phone_norm)        │
              │                                     │
              └──(phone_norm = ANI_DIALNUM_norm)──► ringcentral
```

---

## Install

```bash
pip install pandas pyarrow numpy faker
```

---

## Run

```bash
# Default: 5,000 patients
python generate_all.py

# Quick smoke test: 500 patients
python generate_all.py --n 500 --out ./test_output

# Larger dataset
python generate_all.py --n 50000 --out ./large_output
```

---

## Run individual modules

Each module can run standalone if upstream parquets already exist:

```bash
python gen_dims.py --out ./output
python gen_patients.py --out ./output --n 5000
python gen_facts.py --out ./output
python gen_calls.py --out ./output
python gen_supplementary.py --out ./output
```

---

## Output structure

```
rp_synthetic_output/
  00_navigation/
    patient_navigation_map.parquet   144 cols, one row per patient-practice
  01_facts/
    charges.parquet
    transactions.parquet
    visits.parquet
    statements.parquet
    ringcentral.parquet
    rv_inbound.parquet
    rv_outbound.parquet
  02_dims/
    patient.parquet
    insurance.parquet
    location.parquet
    birdeye.parquet
  03_supplementary/
    phone_bridge.parquet
    campaign_map.parquet
```

---

## Key design decisions

**Deterministic** — `SEED = 42` in `constants.py`. Same seed = same output every run.

**Relational integrity** — every FK is enforced at generation time, not post-hoc.
`charges.VisitID` always exists in `visits.VisitID`.
`transactions.ChargeID` always exists in `charges.ChargeID`.

**Null rates match prod** — every column's null % is taken directly from `DATA_DICTIONARY.md`.

**DQ issues replicated** — `ringcentral.start_time` is intentionally collapsed to
`2026-05-04` (DQ-002). Analysts must use `Start_Date` instead, exactly as in prod.

**Multi-practice patients** — ~15.8% of patients appear under 2+ practice codes,
matching the prod `multi_practice_flag` rate.

**Catastrophe cohort** — patients with `total_calls_window >= 5` are flagged
`is_catastrophe = True`. ~0.17% rate matches prod (5,454 / 3.29M).

**4-bucket adjustment taxonomy** — `AdjustmentBucket` is assigned from the same
taxonomy as prod: `contractual`, `bad_debt`, `collection_agency`, `charity_care`,
`refund_reversal`, `payment_plan`, `other`. Do not use `BadDebtAdjustments` field
directly — use bucket = 'bad_debt' (DQ-001).

---

## Validation after generation

```python
import pandas as pd
from pathlib import Path

root = Path("./rp_synthetic_output")
nav = pd.read_parquet(root / "00_navigation" / "patient_navigation_map.parquet")

# Verify catastrophe cohort
print(nav["is_catastrophe"].sum(), "catastrophe patients")

# Verify multi-practice
print(nav["multi_practice_flag"].sum(), "multi-practice rows")

# Verify payor split
print(nav["payor_cohort"].value_counts(normalize=True))

# Run JOIN_RECIPES.md recipe 1
print(nav[nav["is_catastrophe"]]["PatientState"].value_counts().head(10))
```

---

*Generated by Sutherland Global Services — Healthcare Analytics Practice*
