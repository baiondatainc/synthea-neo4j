"""
helpers.py
──────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Low-level generation helpers. Stateless pure functions.
All randomness comes through numpy/random seeded in constants.py.
"""

import re
import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from constants import (
    FIRST_NAMES_M, FIRST_NAMES_F, LAST_NAMES, MIDDLE_INITIALS, SUFFIXES,
    GENDERS, GENDER_WEIGHTS,
    RACES, RACE_WEIGHTS, ETHNICITIES, ETHNICITY_WEIGHTS,
    PROPENSITY_GRADES, PROPENSITY_DESCS,
    STATE_CITY_ZIP, DEFAULT_STATE, STATE_AREA_CODES,
    PRACTICE_STATES, fmt_phone, norm_phone,
    SEED,
)

rng = np.random.default_rng(SEED)


# ─── Identity helpers ─────────────────────────────────────────────────────────

def pick_gender() -> str:
    return rng.choice(GENDERS, p=GENDER_WEIGHTS)

def pick_name(gender: str) -> tuple[str, Optional[str], str]:
    """Returns (first, middle_or_None, last)"""
    pool = FIRST_NAMES_M if gender == "M" else FIRST_NAMES_F
    first  = rng.choice(pool)
    last   = rng.choice(LAST_NAMES)
    middle = rng.choice(MIDDLE_INITIALS) if rng.random() > 0.425 else None
    return first, middle, last

def pick_suffix() -> Optional[str]:
    """~0% populated in prod (100% null) — return None always"""
    return None

def pick_dob(min_age: int = 18, max_age: int = 85) -> datetime:
    days_range = (max_age - min_age) * 365
    offset = int(rng.integers(0, days_range))
    base = datetime(2026, 5, 1) - timedelta(days=max_age * 365)
    return base + timedelta(days=offset)

def pick_race() -> Optional[str]:
    """56% null in prod"""
    if rng.random() < 0.561:
        return None
    return rng.choice(RACES, p=RACE_WEIGHTS)

def pick_ethnicity() -> Optional[str]:
    """88.9% null"""
    if rng.random() < 0.889:
        return None
    return rng.choice(ETHNICITIES, p=ETHNICITY_WEIGHTS)

def pick_propensity() -> tuple[Optional[str], Optional[str]]:
    """46% null grade"""
    if rng.random() < 0.462:
        return None, None
    grade = rng.choice(PROPENSITY_GRADES)
    desc  = PROPENSITY_DESCS[grade]
    if rng.random() < 0.514:
        return grade, None
    return grade, desc

def pick_ssn() -> Optional[str]:
    """55.6% null; use 999-XX-XXXX pseudonymisation format"""
    if rng.random() < 0.556:
        return None
    mid  = int(rng.integers(10, 99))
    last = int(rng.integers(1000, 9999))
    return f"999-{mid:02d}-{last}"


# ─── Address helpers ──────────────────────────────────────────────────────────

STREET_TYPES  = ["DR","ST","AVE","BLVD","LN","RD","WAY","CT","PL","CIR"]
STREET_NAMES  = ["MAIN","OAK","MAPLE","PINE","ELM","CEDAR","LAKE","HILL",
                 "PARK","SUNSET","RIVER","CHURCH","SCHOOL","AMANA","SPRING"]

def pick_address(state: str) -> tuple[str, Optional[str], str, str, str]:
    """Returns (address, address2_or_None, city, state, zip)"""
    num   = int(rng.integers(100, 9999))
    sname = rng.choice(STREET_NAMES)
    stype = rng.choice(STREET_TYPES)
    addr  = f"{num} {sname} {stype}"
    # 89% null address2 in prod
    addr2 = f"APT {int(rng.integers(1, 500))}" if rng.random() > 0.894 else None
    pool  = STATE_CITY_ZIP.get(state, STATE_CITY_ZIP[DEFAULT_STATE])
    city_zip = rng.choice(pool)
    return addr, addr2, city_zip[0], state, city_zip[1]

def bad_address_indicator() -> Optional[float]:
    """23.6% null; otherwise 0.0 or 1.0"""
    if rng.random() < 0.236:
        return None
    return float(rng.choice([0, 1], p=[0.85, 0.15]))


# ─── Phone helpers ────────────────────────────────────────────────────────────

def pick_phone(state: str) -> Optional[str]:
    """8.8% null"""
    if rng.random() < 0.088:
        return None
    area = rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES[DEFAULT_STATE]))
    rest = str(int(rng.integers(2000000, 9999999)))
    return fmt_phone(area, rest)

def pick_cell_phone(state: str) -> Optional[str]:
    """97.8% null in prod"""
    if rng.random() < 0.978:
        return None
    area = rng.choice(STATE_AREA_CODES.get(state, STATE_AREA_CODES[DEFAULT_STATE]))
    rest = str(int(rng.integers(2000000, 9999999)))
    return fmt_phone(area, rest)

def pick_email(first: str, last: str) -> Optional[str]:
    """61.3% null"""
    if rng.random() < 0.613:
        return None
    domains = ["GMAIL.COM","YAHOO.COM","HOTMAIL.COM","OUTLOOK.COM","AOL.COM"]
    domain  = rng.choice(domains)
    suffix  = str(int(rng.integers(10, 999)))
    return f"{first}{last[:3]}{suffix}@{domain}"

def normalise_phone(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits[-10:] if len(digits) >= 10 else None

def is_institutional_phone(phone_norm: Optional[str]) -> bool:
    """Simple heuristic: certain area codes are institutional"""
    if phone_norm is None:
        return False
    institutional_areas = {"8005","8008","8885","8004"}
    return phone_norm[:4] in institutional_areas


# ─── Financial helpers ────────────────────────────────────────────────────────

def pick_charge_amount() -> float:
    """Radiology charges: log-normal around $500"""
    return round(float(rng.lognormal(mean=6.2, sigma=0.8)), 2)

def pick_payment_amount(charged: float) -> float:
    """Payments typically 20-80% of charged"""
    rate = float(rng.uniform(0.0, 0.82))
    return round(charged * rate, 2)

def pick_adjustment_amount(charged: float, paid: float) -> float:
    """Adjustment = charged - paid - remaining_balance"""
    remaining = max(0, charged - paid)
    adj_rate  = float(rng.uniform(0.3, 1.0))
    return round(remaining * adj_rate, 2)

def pick_outstanding_balance(charged: float, paid: float, adjusted: float) -> float:
    return round(max(0.0, charged - paid - adjusted), 2)

def aging_bucket(service_date: datetime) -> str:
    days = (datetime(2026, 5, 1) - service_date).days
    if days <= 30:   return "0-30"
    if days <= 60:   return "31-60"
    if days <= 90:   return "61-90"
    if days <= 180:  return "91-180"
    if days <= 360:  return "181-360"
    return "361+"


# ─── Date helpers ─────────────────────────────────────────────────────────────

def pick_date_in_window() -> datetime:
    """Random datetime in 2025-05-01 to 2026-05-01"""
    start = datetime(2025, 5, 1)
    days  = int(rng.integers(0, 365))
    secs  = int(rng.integers(0, 86400))
    return start + timedelta(days=days, seconds=secs)

def pick_date_historical() -> datetime:
    """Random date 2018-2025 for older charges/transactions"""
    start = datetime(2018, 1, 1)
    days  = int(rng.integers(0, 2556))
    return start + timedelta(days=days)

def pick_service_date(admit: datetime) -> datetime:
    """Service date is admit date ± 2 days"""
    offset = int(rng.integers(-2, 3))
    return admit + timedelta(days=offset)

def pick_post_date(service: datetime) -> datetime:
    """Post date is service date + 0-14 days"""
    return service + timedelta(days=int(rng.integers(0, 15)))


# ─── ID / key helpers ─────────────────────────────────────────────────────────

def composite_key(src: str, patient_id: int) -> str:
    return f"{src}:{patient_id}"

def hashed_patient_id(src: str, patient_id: int) -> str:
    """Salted SHA-256 matching prod format"""
    salt  = "rp_synthetic_salt_v1"
    token = f"{salt}:{src}:{patient_id}"
    return hashlib.sha256(token.encode()).hexdigest()

def gen_order_number(src: str) -> str:
    num = int(rng.integers(10000000, 99999999))
    return f"{num}{src}"

def gen_batch_number(src: str) -> str:
    dt = pick_date_in_window()
    seq = int(rng.integers(1000, 9999))
    return f"{dt.strftime('%m%d%Y')}{seq:06d}_{src}"

def gen_visit_number(src: str) -> str:
    num = int(rng.integers(1000000000, 9999999999))
    return f"V{num:011d}{src[:3]}"

def gen_history_number(src: str) -> str:
    num = int(rng.integers(100000, 999999))
    return f"E{num}{src[:3]}"

def gen_icn_number() -> Optional[str]:
    """39.6% null"""
    if rng.random() < 0.396:
        return None
    return f"IP{int(rng.integers(100000000000, 999999999999))}"

def gen_statement_id(src: str, patient_id: int, dt: datetime) -> str:
    return f"{dt.strftime('%m%d%Y%H%M%S')}_{patient_id}"


# ─── Cohort assignment helpers ────────────────────────────────────────────────

def assign_payor_cohort(total_calls: float, has_insurance: bool) -> tuple[str, bool, bool, bool]:
    """
    Returns (payor_cohort, is_self_pay, is_bai, is_fully_covered)
    Based on prod: ~54.5% self_pay, ~25% bai, ~20.5% fully_covered
    """
    r = rng.random()
    if r < 0.545:
        return "self_pay", True,  False, False
    if r < 0.795:
        return "bai",      False, True,  False
    return "fully_covered", False, False, True

def assign_call_tier(total_calls: float) -> str:
    """
    call_tier drives leakage curve analysis.
    Prod: most patients are 'zero', catastrophe ≥ 5 calls.
    """
    tc = int(total_calls)
    if tc == 0:    return "zero"
    if tc == 1:    return "one"
    if tc <= 4:    return "two_to_four"
    return "five_plus"   # catastrophe territory

def assign_cohort(call_tier: str) -> str:
    if call_tier == "five_plus": return "catastrophe"
    if call_tier in ("two_to_four","one"):
        return "friction" if rng.random() < 0.3 else "clean"
    return "clean"
