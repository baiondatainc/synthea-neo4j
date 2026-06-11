"""
constants.py
────────────
Sutherland Global Services — Radiology Partners Synthetic Dataset
Shared constants, vocabularies, and configuration used by every generator.

All generators import from here so vocabulary is defined exactly once.
Changing SEED here changes every table deterministically and consistently.
"""

import random
import numpy as np
from datetime import datetime, date

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Dataset parameters ───────────────────────────────────────────────────────
N_PATIENTS          = 5_000       # unique human patients (cross-practice ~3.3M in prod, scaled down)
N_PRACTICES         = 44          # matches real RP practice count
N_LOCATIONS         = 300         # locations across practices
MULTI_PRACTICE_RATE = 0.158       # 520k / 3.29M in prod
CATASTROPHE_RATE    = 0.00166     # 5,454 / 3.29M in prod
SELF_PAY_RATE       = 0.545       # ~1.79M / 3.29M
AVG_VISITS_PER_PAT  = 2.41        # from audit: 30.5M visits / 12.7M unique visits
AVG_CHARGES_PER_VISIT = 2.4       # charges > visits in prod
AVG_TXNS_PER_CHARGE = 1.69        # 51.4M txns / 30.5M charges
AVG_STATEMENTS_PER_PAT = 3.4      # 11M statements / 3.29M patients
REFRESH_DATE        = datetime(2026, 5, 1)
TXN_REFRESH_DATE    = datetime(2026, 5, 14)   # transactions has later refresh
WINDOW_START        = datetime(2025, 5, 1)
WINDOW_END          = datetime(2026, 5, 1)

# ─── Practice codes — real RP Source_Database_Code values ─────────────────────
PRACTICE_CODES = [
    "SAPA", "PMR",  "ACRB", "ESR",  "MRB",  "CRC",  "NRAA", "GSIA",
    "CIRPA","IAI",  "GRH",  "NRA",  "ACR",  "RASFA","TRI",  "PXP",
    "SMED", "WRAD", "RADI", "RADS", "RADX", "RADY", "RADZ", "RADA",
    "RADB", "RADC", "RADD", "RADE", "RADF", "RADG", "RADH", "RADI2",
    "RADJ", "RADK", "RADL", "RADM", "RADN", "RADO", "RADP", "RADQ",
    "RADR", "RADS2","RADT", "RADU",
]
assert len(PRACTICE_CODES) == N_PRACTICES

# Practice → state mapping (drives geographic cohort flags)
PRACTICE_STATES = {
    "SAPA": "TX", "PMR": "TN",  "ACRB": "OH", "ESR": "FL",  "MRB": "TN",
    "CRC":  "OH", "NRAA":"TN",  "GSIA": "TX", "CIRPA":"FL", "IAI": "TX",
    "GRH":  "GA", "NRA": "TN",  "ACR":  "OH", "RASFA":"TX", "TRI": "TN",
    "PXP":  "FL", "SMED":"TX",  "WRAD": "NV", "RADI": "GA", "RADS": "TN",
    "RADX": "TX", "RADY": "OH", "RADZ": "FL", "RADA": "TN", "RADB": "TX",
    "RADC": "OH", "RADD": "GA", "RADE": "FL", "RADF": "NV", "RADG": "TN",
    "RADH": "TX", "RADI2":"OH", "RADJ": "FL", "RADK": "GA", "RADL": "TN",
    "RADM": "TX", "RADN": "OH", "RADO": "FL", "RADP": "GA", "RADQ": "NV",
    "RADR": "TN", "RADS2":"TX", "RADT": "OH", "RADU": "FL",
}

# ─── State → city/zip pools ────────────────────────────────────────────────────
STATE_CITY_ZIP = {
    "TN": [("NASHVILLE","37201"),("FRANKLIN","37064"),("MURFREESBORO","37128"),
           ("KNOXVILLE","37902"),("MOUNT JULIET","37122"),("LEBANON","37087")],
    "TX": [("HOUSTON","77001"),("DALLAS","75201"),("AUSTIN","78701"),
           ("SAN ANTONIO","78201"),("FORT WORTH","76101"),("PLANO","75023")],
    "OH": [("COLUMBUS","43201"),("CLEVELAND","44101"),("CINCINNATI","45201"),
           ("TOLEDO","43601"),("AKRON","44301"),("DAYTON","45401")],
    "FL": [("MIAMI","33101"),("ORLANDO","32801"),("TAMPA","33601"),
           ("JACKSONVILLE","32099"),("FORT LAUDERDALE","33301"),("ST PETE","33701")],
    "GA": [("ATLANTA","30301"),("SAVANNAH","31401"),("AUGUSTA","30901"),
           ("MACON","31201"),("ATHENS","30601"),("COLUMBUS","31901")],
    "NV": [("LAS VEGAS","89101"),("HENDERSON","89002"),("RENO","89501"),
           ("SPARKS","89431"),("NORTH LAS VEGAS","89030"),("CARSON CITY","89701")],
}
# Any practice state not in above falls back to TN
DEFAULT_STATE = "TN"

# ─── Patient demographics ─────────────────────────────────────────────────────
FIRST_NAMES_M = ["JAMES","JOHN","ROBERT","MICHAEL","WILLIAM","DAVID","RICHARD",
                 "JOSEPH","THOMAS","CHARLES","CHRISTOPHER","DANIEL","PAUL","MARK",
                 "DONALD","GEORGE","KENNETH","STEVEN","EDWARD","BRIAN","RONALD",
                 "ANTHONY","KEVIN","JASON","MATTHEW","GARY","TIMOTHY","JOSE","LARRY"]
FIRST_NAMES_F = ["MARY","PATRICIA","LINDA","BARBARA","ELIZABETH","JENNIFER","MARIA",
                 "SUSAN","MARGARET","DOROTHY","LISA","NANCY","KAREN","BETTY","HELEN",
                 "SANDRA","DONNA","CAROL","RUTH","SHARON","MICHELLE","LAURA","SARAH",
                 "KIMBERLY","DEBORAH","JESSICA","SHIRLEY","CYNTHIA","ANGELA","MELISSA"]
LAST_NAMES    = ["SMITH","JOHNSON","WILLIAMS","BROWN","JONES","GARCIA","MILLER",
                 "DAVIS","RODRIGUEZ","MARTINEZ","HERNANDEZ","LOPEZ","GONZALEZ",
                 "WILSON","ANDERSON","THOMAS","TAYLOR","MOORE","JACKSON","MARTIN",
                 "LEE","PEREZ","THOMPSON","WHITE","HARRIS","SANCHEZ","CLARK","RAMIREZ",
                 "LEWIS","ROBINSON","WALKER","YOUNG","ALLEN","KING","WRIGHT","SCOTT",
                 "TORRES","NGUYEN","HILL","FLORES","GREEN","ADAMS","NELSON","BAKER",
                 "HALL","RIVERA","CAMPBELL","MITCHELL","CARTER","ROBERTS","RICE"]
MIDDLE_INITIALS = list("ABCDEFGHJKLMNPRSTVWZ")
SUFFIXES = ["JR","SR","II","III"]

GENDERS = ["M","F"]
GENDER_WEIGHTS = [0.44, 0.56]   # matches prod: 44% M, 56% F

RACES = ["White","Black or African American","Hispanic","Asian",
         "American Indian","Other","Unknown"]
RACE_WEIGHTS = [0.60, 0.18, 0.12, 0.06, 0.01, 0.02, 0.01]

ETHNICITIES = ["Not Hispanic","Hispanic or Latino"]
ETHNICITY_WEIGHTS = [0.85, 0.15]

PROPENSITY_GRADES = ["A","B","C","D","E","F"]
PROPENSITY_DESCS  = {
    "A":"Very High","B":"High","C":"Medium-High",
    "D":"Medium","E":"Low","F":"Very Low"
}

# ─── Insurance / carrier pools ────────────────────────────────────────────────
CARRIERS = ["AETNA","BCBS","CIGNA","UNITED","HUMANA","MEDICARE","MEDICAID",
            "COMMERCIAL","TRICARE","MOLINA","CENTENE","WELLCARE","AMBETTER",
            "OSCAR","BRIGHT HEALTH","COVENTRY","MAGELLAN"]
CARRIER_WEIGHTS = [0.12,0.18,0.10,0.15,0.08,0.12,0.07,0.06,0.02,0.02,
                   0.02,0.01,0.01,0.01,0.01,0.01,0.01]

PLAN_TYPES = ["COMM","HMO","PPO","EPO","MCDASSIGN","MCRASSIGN","TPA",
              "INDEMNITY","POS","HDHP"]

# ─── ICD-10 codes (radiology-relevant) ────────────────────────────────────────
ICD10_CODES = [
    "K43.9","I67.2","I65.23","R51.9","Z76.89","R91.8","M54.5","J18.9",
    "I10",  "E11.9","N18.3","C34.10","R06.09","K57.30","M17.11","G89.29",
    "I63.9","R55",  "Z12.31","M50.20","K80.20","N32.89","I34.0","Z90.12",
    "R93.0","R93.1","R93.2","R93.3","R93.4","R93.5","R93.6","R93.89",
    "Z87.891","M79.3","G43.909","R00.0","J44.1","I25.10","N40.0",
]

# ─── CPT/procedure codes (radiology) ──────────────────────────────────────────
PROCEDURE_CODES = [
    "70553","71046","72148","73721","74177","76817","77067","71250","72195",
    "73223","74183","76700","77080","70450","71045","72141","73201","74176",
    "76536","76805","70486","71048","72158","73222","74178","76604","77065",
    "G9551","G9552","G9553","70544","73706","74175","76770","70540","71047",
]
PROCEDURE_DESCRIPTIONS = {
    "70553":"MRI BRAIN W/WO CONTRAST",
    "71046":"CHEST XRAY 2 VIEWS",
    "72148":"MRI LUMBAR SPINE W/O CONTRAST",
    "73721":"MRI JOINT LOWER EXTREMITY W/O",
    "74177":"CT ABDOMEN PELVIS W CONTRAST",
    "76817":"OB ULTRASOUND TRANSVAGINAL",
    "77067":"SCREENING MAMMOGRAPHY BILATERAL",
    "71250":"CT CHEST W/O CONTRAST",
    "72195":"MRI PELVIS W/O CONTRAST",
    "G9551":"FINAL REPORTS FOR ABDOMINAL IMAGING STUDIES WITHOUT AN INCID",
}

# ─── RingCentral / campaign pools ─────────────────────────────────────────────
CAMPAIGN_NAMES = ["SAPA","PMR","ACRB","NRA","NRAA","GSIA","IAI","CRC",
                  "ESR","MRB","ACR","TRI","GRH","RASFA","PXP","SMED"]
SKILL_NAMES    = ["IAI English","SAPA English","PMR English","NRA English",
                  "CRC English","Espanol","Inbound Main","Outbound Collection"]
TEAM_NAMES     = ["PXP - Sagility","Team Alpha","Team Beta","Team Gamma",
                  "RCM Offshore","Collections North","Collections South"]
AGENT_NAMES    = ["Jerome Viloria","Maria Santos","David Chen","Sarah Thompson",
                  "Michael Rivera","Jennifer Lee","Robert Garcia","Ashley Wilson",
                  "Christopher Martinez","Amanda Johnson","Brandon Smith","Nicole Brown"]
DISP_NAMES     = ["Payment Collected","Answering Machine","No Answer","Left Message",
                  "Patient Refused","Callback Scheduled","Wrong Number","Promise to Pay",
                  "Disconnected","In Queue","Resolved"]
IVR_TYPES      = ["Inbound - Pay-By-Phone","Inbound - Balance Inquiry",
                  "Inbound - Statement Question","Inbound - Insurance Question"]
RESULT_DESCS   = ["Answering Machine","No Answer","Payment Collected",
                  "Left Message","Callback","Disconnected","Refused"]

# ─── Adjustment taxonomy (4-bucket) ───────────────────────────────────────────
ADJUSTMENT_BUCKETS   = ["contractual","bad_debt","collection_agency",
                         "charity_care","refund_reversal","payment_plan","other"]
ADJUSTMENT_WEIGHTS   = [0.276,0.072,0.003,0.000,0.000,0.000,0.001]   # normalised from prod
ADJUSTMENT_TYPES     = {
    "contractual"     : ["Write Off","Contractual Adj","CO144","Contract Variance"],
    "bad_debt"        : ["Bad Debt WO","Timely Filing","Prior Auth Failure","AR Cleanup"],
    "collection_agency": ["Collection Agency","Agency Transfer"],
    "charity_care"    : ["Charity Care","Financial Hardship"],
    "refund_reversal" : ["Refund","Reversal","Takeback"],
    "payment_plan"    : ["Payment Plan"],
    "other"           : ["Other Adjustment","Miscellaneous"],
}
PROCESSING_TYPES = ["Payment","ADJUSTMENT","Adjustment","Self Payment",
                    "Self pay","Self-Pay Payment","Collection Pmt.",
                    "Time of Service","Facility Cc Pmt","Credit","Refund","Takeback"]

# ─── Statement levels ─────────────────────────────────────────────────────────
STATEMENT_LEVELS = ["Statement 1","Statement 2","Statement 3",
                    "Final Notice","Collections Notice"]

# ─── Location types ───────────────────────────────────────────────────────────
LOCATION_TYPES = ["Professional","Global","Facility","Outpatient"]

# ─── Birdeye review sources ────────────────────────────────────────────────────
BIRDEYE_SOURCES = ["Google","Yelp","Healthgrades","Facebook","Zocdoc"]
BIRDEYE_SOURCE_WEIGHTS = [0.65,0.15,0.10,0.07,0.03]

# ─── Phone formatting helpers ─────────────────────────────────────────────────
def fmt_phone(area: str, rest: str) -> str:
    """Format as (AAA)NNN-NNNN"""
    return f"({area}){rest[:3]}-{rest[3:7]}"

def norm_phone(raw: str) -> str:
    """Strip to 10-digit string"""
    import re
    return re.sub(r"\D", "", raw)[-10:]

STATE_AREA_CODES = {
    "TN": ["615","901","423","865","931"],
    "TX": ["713","214","512","210","817","972"],
    "OH": ["614","216","513","419","330","937"],
    "FL": ["305","407","813","904","954","561"],
    "GA": ["404","678","770","912","706","478"],
    "NV": ["702","725","775"],
}
