"""
gen_dims.py
───────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Generates dimension tables:
  • 02_dims/location.parquet
  • 02_dims/insurance.parquet
  • 02_dims/birdeye.parquet

Run standalone:
    python gen_dims.py --out ./output

Relationships:
  location.Source_Database_Code  → all fact tables Source_Database_Code
  location.LocationID            → charges.LocationID, visits.LocationID
  insurance.Source_Database_Code + PlanNumber → visits.PrimaryInsurancePlanNum
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    SEED, PRACTICE_CODES, PRACTICE_STATES, N_LOCATIONS,
    STATE_CITY_ZIP, DEFAULT_STATE, STATE_AREA_CODES,
    LOCATION_TYPES, CARRIERS, CARRIER_WEIGHTS, PLAN_TYPES,
    BIRDEYE_SOURCES, BIRDEYE_SOURCE_WEIGHTS,
    REFRESH_DATE,
)
from helpers import (
    rng, pick_address, fmt_phone, normalise_phone,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOCATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_location(n: int = N_LOCATIONS) -> pd.DataFrame:
    """
    One row per (Source_Database_Code, LocationID).
    ~300 rows total, ~6-7 per practice.
    66 of 15,012 in prod have Birdeye data (~0.4%) — we scale to ~2%.
    """
    rows = []
    loc_id_counter = {}

    for i in range(n):
        src   = rng.choice(PRACTICE_CODES)
        state = PRACTICE_STATES.get(src, DEFAULT_STATE)

        loc_id_counter[src] = loc_id_counter.get(src, 0) + 1
        loc_id = str(loc_id_counter[src])

        addr, addr2, city, _, zip_ = pick_address(state)

        # NPI: 44.3% null
        npi = f"{int(rng.integers(1000000000, 1999999999))}" if rng.random() > 0.443 else None

        # Phone: 53.5% null
        area = rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES[DEFAULT_STATE]))
        raw_phone = fmt_phone(area, str(int(rng.integers(2000000, 9999999)))) if rng.random() > 0.535 else None
        raw_fax   = fmt_phone(area, str(int(rng.integers(2000000, 9999999)))) if rng.random() > 0.865 else None

        loc_type = rng.choice(LOCATION_TYPES, p=[0.55, 0.30, 0.10, 0.05])

        # Birdeye enrichment: ~2% of locations
        has_birdeye = rng.random() < 0.02
        if has_birdeye:
            total_reviews = int(rng.integers(5, 150))
            five  = int(rng.integers(int(total_reviews * 0.4), int(total_reviews * 0.85)))
            four  = int(rng.integers(0, max(1, total_reviews - five)))
            three = int(rng.integers(0, max(1, total_reviews - five - four)))
            two   = int(rng.integers(0, max(1, total_reviews - five - four - three)))
            one   = max(0, total_reviews - five - four - three - two)
            avg   = round((five*5 + four*4 + three*3 + two*2 + one*1) / max(total_reviews, 1), 2)
            med   = 5.0 if five > (total_reviews / 2) else 4.0
            first_rev = REFRESH_DATE - pd.Timedelta(days=int(rng.integers(30, 365)))
            last_rev  = REFRESH_DATE - pd.Timedelta(days=int(rng.integers(0, 29)))
            with_comment = int(total_reviews * rng.uniform(0.6, 0.95))
            phi_rev   = int(rng.integers(0, 3)) if rng.random() < 0.05 else 0
            phi_phone = phi_rev
            phi_email = int(rng.integers(0, phi_rev + 1)) if phi_rev > 0 else 0
            phi_ssn   = 0
            one_pct   = round(one / max(total_reviews, 1) * 100, 2)
            one_two   = round((one + two) / max(total_reviews, 1) * 100, 2)
        else:
            (total_reviews, five, four, three, two, one,
             avg, med, first_rev, last_rev, with_comment,
             phi_rev, phi_phone, phi_email, phi_ssn,
             one_pct, one_two) = (None,)*17

        abbr = src + loc_id.zfill(2)
        name = f"{src} Radiology Location {loc_id}"

        rows.append({
            "Source_Database_Code":       src,
            "LocationID":                 loc_id,
            "LocationName":               name,
            "LocationAbbreviation":       abbr,
            "LocationNPINumber":          npi,
            "LocationAddress":            addr,
            "LocationAddress2":           addr2,
            "LocationCity":               city,
            "LocationState":              state,
            "LocationZip":                zip_,
            "LocationPhone":              raw_phone,
            "LocationFax":                raw_fax,
            "LocationType":               loc_type,
            "LocationFDANumber":          str(int(rng.integers(100000, 999999))) if rng.random() > 0.880 else None,
            "LocationAlternativeAddress": addr if rng.random() > 0.529 else None,
            "LocationAlternativeAddress2":None,
            "LocationAlternativecity":    city if rng.random() > 0.529 else None,
            "LocationAlternativestate":   state if rng.random() > 0.529 else None,
            "LocationAlternativezip":     zip_ if rng.random() > 0.529 else None,
            "tbl_Refresh_Date":           REFRESH_DATE,
            "LocationPhone_norm":         normalise_phone(raw_phone),
            "LocationFax_norm":           normalise_phone(raw_fax),
            "birdeye_review_count":       float(total_reviews) if has_birdeye else None,
            "birdeye_avg_rating":         float(avg) if has_birdeye else None,
            "birdeye_median_rating":      float(med) if has_birdeye else None,
            "birdeye_one_star_count":     float(one) if has_birdeye else None,
            "birdeye_two_star_count":     float(two) if has_birdeye else None,
            "birdeye_three_star_count":   float(three) if has_birdeye else None,
            "birdeye_four_star_count":    float(four) if has_birdeye else None,
            "birdeye_five_star_count":    float(five) if has_birdeye else None,
            "birdeye_last_review_date":   last_rev if has_birdeye else None,
            "birdeye_first_review_date":  first_rev if has_birdeye else None,
            "birdeye_review_with_comment_count": float(with_comment) if has_birdeye else None,
            "birdeye_phi_review_count":   float(phi_rev) if has_birdeye else None,
            "birdeye_phi_phone_total":    float(phi_phone) if has_birdeye else None,
            "birdeye_phi_email_total":    float(phi_email) if has_birdeye else None,
            "birdeye_phi_ssn_total":      float(phi_ssn) if has_birdeye else None,
            "birdeye_one_star_pct":       float(one_pct) if has_birdeye else None,
            "birdeye_one_or_two_star_pct":float(one_two) if has_birdeye else None,
        })

    df = pd.DataFrame(rows)
    # Cast types to match schema
    for col in ["birdeye_review_count","birdeye_avg_rating","birdeye_median_rating",
                "birdeye_one_star_count","birdeye_two_star_count","birdeye_three_star_count",
                "birdeye_four_star_count","birdeye_five_star_count",
                "birdeye_review_with_comment_count","birdeye_phi_review_count",
                "birdeye_phi_phone_total","birdeye_phi_email_total","birdeye_phi_ssn_total",
                "birdeye_one_star_pct","birdeye_one_or_two_star_pct"]:
        df[col] = df[col].astype("float64")
    df["LocationID"] = df["LocationID"].astype("string")
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    df["birdeye_last_review_date"]  = pd.to_datetime(df["birdeye_last_review_date"])
    df["birdeye_first_review_date"] = pd.to_datetime(df["birdeye_first_review_date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INSURANCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_insurance(n: int = 2000) -> pd.DataFrame:
    """
    Insurance plan dictionary.
    Grain: (Source_Database_Code, PlanNumber)
    ~2000 plans across 44 practices mirrors prod density.
    Joins to visits.PrimaryInsurancePlanNum / SecondaryInsurancePlanNum.
    """
    rows = []
    plan_counter = 100000

    for _ in range(n):
        src      = rng.choice(PRACTICE_CODES)
        carrier  = rng.choice(CARRIERS, p=CARRIER_WEIGHTS)
        ptype    = rng.choice(PLAN_TYPES)
        plan_num = str(plan_counter)
        plan_counter += int(rng.integers(1, 10))
        plan_name = f"{carrier} {ptype}"

        rows.append({
            "Source_Database_Code": src,
            "PlanNumber":           plan_num,
            "PlanName":             plan_name,
            "PlanType":             ptype,
            "Carrier_Name":         carrier,
            "tbl_Refresh_Date":     REFRESH_DATE,
        })

    df = pd.DataFrame(rows)
    df["PlanNumber"]      = df["PlanNumber"].astype("string")
    df["tbl_Refresh_Date"] = pd.to_datetime(df["tbl_Refresh_Date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BIRDEYE (raw reviews dim)
# ─────────────────────────────────────────────────────────────────────────────

def generate_birdeye(location_df: pd.DataFrame, n_reviews: int = 500) -> pd.DataFrame:
    """
    Raw Birdeye review records.
    Only locations with birdeye data get reviews.
    PHI-flagged reviews: ~0.5% contain phone/email tokens.
    """
    birdeye_locs = location_df[location_df["birdeye_review_count"].notna()]["LocationName"].tolist()
    if not birdeye_locs:
        # Fallback: use all locations
        birdeye_locs = location_df["LocationName"].tolist()

    rows = []
    review_comments = [
        "Great staff and fast service.",
        "Billing was confusing but imaging was professional.",
        "Long wait times but quality results.",
        "Excellent radiologist. Very thorough.",
        "Had trouble with insurance but staff was helpful.",
        None,  # ~24.8% null comment
        "The facility was clean and modern.",
        "Report came back quickly. Very impressed.",
        "Scheduling was easy and staff was friendly.",
        "Would not recommend. Poor communication.",
        "Insurance billing nightmare but good images.",
        "Very professional team.",
    ]

    for i in range(n_reviews):
        loc_name = rng.choice(birdeye_locs)
        source   = rng.choice(BIRDEYE_SOURCES, p=BIRDEYE_SOURCE_WEIGHTS)
        rating   = int(rng.choice([1,2,3,4,5], p=[0.08,0.05,0.07,0.15,0.65]))
        comment  = rng.choice(review_comments)

        # PHI flags: ~0.5% have phone, ~0.2% email, ~0% SSN
        phi_phone = 1 if rng.random() < 0.005 else 0
        phi_email = 1 if rng.random() < 0.002 else 0
        phi_ssn   = 0
        phi_flag  = bool(phi_phone or phi_email)

        # Date: within last 12 months
        days_ago  = int(rng.integers(0, 365))
        dt_str    = (REFRESH_DATE - pd.Timedelta(days=days_ago)).strftime("%a, %b %d, %Y %I:%M %p")

        rows.append({
            "Location":         loc_name,
            "Review Source":    source,
            "Date Posted On":   dt_str,
            "Review Rating":    rating,
            "Review Comment":   comment,
            "phi_phone_count":  phi_phone,
            "phi_email_count":  phi_email,
            "phi_ssn_count":    phi_ssn,
            "phi_flagged":      phi_flag,
        })

    df = pd.DataFrame(rows)
    df["Review Rating"]    = df["Review Rating"].astype("int64")
    df["phi_phone_count"]  = df["phi_phone_count"].astype("int64")
    df["phi_email_count"]  = df["phi_email_count"].astype("int64")
    df["phi_ssn_count"]    = df["phi_ssn_count"].astype("int64")
    df["phi_flagged"]      = df["phi_flagged"].astype("bool")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(out_dir: Path) -> dict:
    print("Generating location.parquet ...")
    location_df = generate_location()
    path = out_dir / "02_dims" / "location.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    location_df.to_parquet(path, index=False)
    print(f"  ✓ {len(location_df):,} rows → {path}")

    print("Generating insurance.parquet ...")
    insurance_df = generate_insurance()
    path = out_dir / "02_dims" / "insurance.parquet"
    insurance_df.to_parquet(path, index=False)
    print(f"  ✓ {len(insurance_df):,} rows → {path}")

    print("Generating birdeye.parquet ...")
    birdeye_df = generate_birdeye(location_df)
    path = out_dir / "02_dims" / "birdeye.parquet"
    birdeye_df.to_parquet(path, index=False)
    print(f"  ✓ {len(birdeye_df):,} rows → {path}")

    return {"location": location_df, "insurance": insurance_df, "birdeye": birdeye_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RP dimension parquets")
    parser.add_argument("--out", default="./output", help="Output root directory")
    args = parser.parse_args()
    run(Path(args.out))
