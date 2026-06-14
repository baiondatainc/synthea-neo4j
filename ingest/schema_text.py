"""
schema_text.py
──────────────
SGS — RP Knowledge Graph
Exact schema fed to LangChain GraphCypherQAChain as context.
Keep this in sync with ingestion.py whenever the graph changes.
"""

GRAPH_SCHEMA = """
Node labels and key properties
================================

(:Patient)
  source_db             string   e.g. "SAPA", "PMR", "ACRB"
  patient_id            integer  e.g. 1000000
  first_name            string
  last_name             string
  gender                string   "M" or "F"
  dob                   datetime
  state                 string   e.g. "TX", "TN", "OH", "FL", "GA", "NV"
  city                  string
  zip                   string
  race                  string
  ethnicity             string
  email                 string   (61% null)
  phone_norm            string   10-digit normalised phone
  propensity_grade      string   "A"-"F" (46% null)
  bad_address_indicator float    0.0 or 1.0
  -- Financial rollups (from navigation map) --
  total_charged         float    total billed to this patient
  total_paid            float    total payments received
  outstanding_balance   float    unpaid amount
  total_adjusted        float    total contractual + bad debt + other adjustments
  adj_contractual       float    contractual write-offs
  adj_bad_debt          float    bad debt write-offs  ← USE THIS not BadDebtAdjustments field
  adj_collection_agency float    collection agency adjustments
  adj_refund_reversal   float    refunds/reversals (negative)
  adj_other             float    other adjustments
  -- Cohort flags --
  payor_cohort          string   "self_pay" | "bai" | "fully_covered"
  call_tier             string   "zero" | "one" | "two_to_four" | "five_plus"
  is_catastrophe        boolean  true if total_calls_window >= 5
  is_friction           boolean
  is_clean              boolean
  is_self_pay           boolean
  is_bai                boolean  balance after insurance
  is_fully_covered      boolean
  is_sapa               boolean  patient at SAPA practice
  is_nraa               boolean  patient at NRAA practice
  is_tennessee          boolean
  is_atlanta_404        boolean
  multi_practice_flag   boolean  appears at 2+ practices
  practice_count        integer
  -- Activity counts --
  visit_count           float
  charge_count          integer
  statement_count       integer
  total_calls_window    float    total calls in 12-month window
  rv_in_calls           float    IVR inbound calls
  rv_out_calls          float    dialler outbound calls
  rc_calls              float    RingCentral attributed calls
  has_any_calls         boolean
  has_insurance         boolean
  active_window         boolean  active in 2025-05-01 to 2026-05-01 window
  carrier_name          string   insurance carrier e.g. "BCBS", "AETNA"
  plan_name             string
  plan_type             string   "HMO" | "PPO" | "EPO" | "MCRASSIGN" etc

(:Practice)
  code                  string   e.g. "SAPA", "PMR", "ACRB", "NRAA" (44 practices)

(:Location)
  source_db             string
  location_id           string
  name                  string   e.g. "SAPA Radiology Location 1"
  abbreviation          string
  npi                   string
  city                  string
  state                 string
  zip                   string
  location_type         string   "Professional" | "Global" | "Facility" | "Outpatient"
  phone_norm            string
  birdeye_avg_rating    float    average review rating (null if no reviews)
  birdeye_review_count  float    number of reviews
  birdeye_phi_review_count float reviews containing PHI
  birdeye_one_star_pct  float    % one-star reviews

(:Visit)
  source_db             string
  visit_id              string
  admit_date            datetime
  discharge_date        datetime
  location_id           string
  primary_insurance_plan   string  plan number
  visit_number          string
  history_number        string
  primary_auth_number   string

(:Charge)
  source_db             string
  charge_id             string
  charge_amount         float    billed amount e.g. 675.82
  procedure_code        string   CPT code e.g. "70553", "71046", "74177"
  procedure_description string   e.g. "MRI BRAIN W/WO CONTRAST"
  procedure_modality    string   "MRI" | "CT" | "XR" | "US" | "NM" | "DXA"
  service_date          datetime
  post_date             datetime
  balance               float    remaining balance on this charge
  current_responsible_level string "Primary" | "Secondary" | "Patient"
  place_of_service      string
  modifier              string   e.g. "26", "TC", "LT", "RT"
  dos_aging_bucket      string   "0-30" | "31-60" | "61-90" | "91-180" | "181-360" | "361+"
  line_status           string   "PT" | "INS" | "VOID" | "HOLD" | "CLEAN"
  is_voided             integer  1 if voided
  is_hold               integer  1 if on hold
  payment_plan_present  boolean
  icd10_1 through icd10_5  string  ICD-10 diagnosis codes

(:Transaction)
  source_db             string
  payment_id            string
  payment_amount        float    actual cash/check payment received
  adjustment_amount     float    contractual or bad-debt adjustment
  adjustment_bucket     string   "contractual" | "bad_debt" | "collection_agency" |
                                 "charity_care" | "refund_reversal" | "payment_plan" | "other"
                                 ← ALWAYS use adjustment_bucket, not ProcessingType
  adjustment_type       string   e.g. "CO144", "Write Off", "Bad Debt WO"
  processing_type       string   e.g. "Payment", "ADJUSTMENT" (49 variants — use bucket instead)
  post_date             datetime
  balance_after_post    float
  allowed_amount        float
  bad_debt_adjustments  float    ← WARNING: only 42% of true bad debt captured here. Use adj_bucket
  co_insurance_amount   float
  deductible_amount     float
  co_pay_amount         float
  denial_code           integer
  denial_note           string   e.g. "CO144 - contractual", "CO29 - timely filing"
  paysource             string   "Patient" | "Insurance" | "Agency"
  payment_method        string   "Manual" | "EDI" | "Portal" | "IVR" | "Check"
  transaction_type      string   "Standard" | "Reversal" | "Transfer"
  days_to_agency        float    days until sent to collection agency

(:Statement)
  statement_id          string
  patient_balance       float    balance on this statement e.g. 999.09
  total_balance         float    total account balance e.g. 3519.79
  statement_level       string   "Statement 1" | "Statement 2" | "Statement 3" |
                                 "Final Notice" | "Collections Notice"
  created_date          datetime
  released_date         datetime
  is_released           integer  1 if sent
  is_on_hold            integer
  email_successful      string   "Yes" | "No"
  text_successful       string   "Yes" | "No"

(:InsurancePlan)
  source_db             string
  plan_number           string
  plan_name             string   e.g. "BCBS PPO"
  plan_type             string   "HMO" | "PPO" | "EPO" | "MCDASSIGN" | "MCRASSIGN" | "TPA"
  carrier_name          string   "AETNA" | "BCBS" | "CIGNA" | "UNITED" | "HUMANA" |
                                 "MEDICARE" | "MEDICAID" | "TRICARE" | "MOLINA" etc

(:RCCall)
  contact_id            string
  campaign_name         string   e.g. "SAPA", "PMR", "NRA"
  skill_name            string
  agent_name            string
  team_name             string
  start_date            string   ← USE THIS for date (DQ-002: start_time is broken)
  agent_time            integer  seconds agent spent on call
  in_queue              integer  seconds in queue
  acw_time              integer  after-call work seconds
  total_time            integer  total call duration seconds
  abandon_time          integer  seconds before abandon (0 if not abandoned)
  abandon               string   "Y" | "N"
  sla                   integer  1 if SLA met, 0 if not
  disp_name             string   "Payment Collected" | "No Answer" | "Left Message" etc
  hold_time             integer  seconds on hold
  ani_norm              string   10-digit caller number (links to phone_bridge.phone_norm)
  rc_attributable       boolean  true if call can be attributed to a patient

(:IVRInbound)
  response_id           string
  account_id            string   = PatientID (DQ-003 confirmed)
  balance               float    account balance at time of call
  amount_paid           float    payment made during this call (null if no payment)
  ivr_type              string   "Inbound - Pay-By-Phone" | "Inbound - Balance Inquiry" etc
  call_datetime         datetime
  call_duration         float    minutes
  auth_success          boolean  patient authenticated successfully
  result_desc           string   "Approved" | "No Payment"
  facility_code         string   practice code

(:DiallerCall)
  account               string   internal account identifier
  account_id            string   = PatientID (DQ-003 confirmed)
  patient_balance       float    balance at time of call
  call_datetime         datetime (20.8% null — pipeline artefact)
  result_desc           string   "Payment Collected" | "No Answer" | "Left Message" etc
  service_loc           string   practice location code
  phone_norm            string   10-digit patient phone

(:PhoneBridge)
  source_db             string
  patient_id            integer
  phone_norm            string   10-digit phone number
  phone_type            string   "patient" | "cell" | "responsible_party" | "responsible_party_cell"
  rc_call_count         float    number of RC calls attributed to this phone
  campaign_count        float
  campaigns_contacted   string   comma-separated campaign names
  primary_campaign      string

(:Campaign)
  name                  string   e.g. "SAPA", "PMR", "NRA", "GSIA"
  source_db             string   practice code (8.9% null for non-Imagine practices)
  notes                 string

(:DiagnosisCode)
  code                  string   ICD-10 code e.g. "I10", "E11.9", "R51.9", "K43.9"

(:ProcedureCode)
  code                  string   CPT code e.g. "70553", "71046"
  description           string   e.g. "MRI BRAIN W/WO CONTRAST"
  modality              string   "MRI" | "CT" | "XR" | "US" | "NM"

(:BirdeyeReview)
  location              string   location name
  date_posted           string
  source                string   "Google" | "Yelp" | "Healthgrades" | "Facebook"
  rating                integer  1-5
  comment               string   (may be null)
  phi_flagged           boolean  true if review contains PHI (phone/email)


Relationships
================================

(Patient)-[:REGISTERED_AT]->(Practice)
(Patient)-[:HAD_VISIT]->(Visit)
(Patient)-[:HAS_CHARGE {charge_amount, service_date, modality}]->(Charge)
(Patient)-[:HAS_TRANSACTION {payment_amount, adjustment_amount, bucket}]->(Transaction)
(Patient)-[:RECEIVED_STATEMENT {patient_balance, total_balance, level, created_date}]->(Statement)
(Patient)-[:IDENTIFIED_BY_PHONE]->(PhoneBridge)
(Patient)-[:CALLED_IVR {amount_paid, balance, call_date}]->(IVRInbound)
(Patient)-[:CONTACTED_BY_DIALLER {patient_balance, call_date, result}]->(DiallerCall)

(Visit)-[:PERFORMED_AT]->(Location)
(Visit)-[:UNDER_PLAN {plan_type:"primary"}]->(InsurancePlan)

(Charge)-[:PART_OF_VISIT]->(Visit)
(Charge)-[:AT_LOCATION]->(Location)
(Charge)-[:DIAGNOSED_WITH]->(DiagnosisCode)
(Charge)-[:USES_PROCEDURE]->(ProcedureCode)

(Transaction)-[:SETTLES {payment_amount, adjustment_amount, adjustment_bucket, bad_debt, balance_after_post}]->(Charge)

(Location)-[:BELONGS_TO_PRACTICE]->(Practice)
(InsurancePlan)-[:ISSUED_BY_PRACTICE]->(Practice)

(RCCall)-[:PART_OF_CAMPAIGN]->(Campaign)
(RCCall)-[:ATTRIBUTED_TO_PHONE]->(PhoneBridge)

(Campaign)-[:RUN_BY]->(Practice)

(BirdeyeReview)-[:REVIEWS]->(Location)

(PhoneBridge)-[:BRIDGES_TO_PATIENT]->(Patient)   [reverse traversal]


Data Quality Notes (critical for correct Cypher)
================================

DQ-001: adj_bad_debt on Patient captures true bad debt ($6.4M total).
        Transaction.bad_debt_adjustments is only 42% of true bad debt.
        ALWAYS use: p.adj_bad_debt or WHERE t.adjustment_bucket = 'bad_debt'

DQ-002: RCCall.start_time is broken (all rows = 2026-05-04).
        ALWAYS use: r.start_date for date-based RC call queries.

DQ-003: IVRInbound.account_id = PatientID (confirmed).
        DiallerCall.account_id = PatientID (confirmed).
        Join on: MATCH (p:Patient) WHERE p.patient_id = toInteger(i.account_id)

ADJUSTMENT BUCKETS: never filter on processing_type directly (49 variants, inconsistent casing).
  Use: t.adjustment_bucket IN ['contractual','bad_debt','collection_agency','charity_care',
                                'refund_reversal','payment_plan','other']

SELF-PAY PATIENTS: WHERE p.is_self_pay = true
CATASTROPHE PATIENTS: WHERE p.is_catastrophe = true  (total_calls >= 5)
ACTIVE WINDOW: WHERE p.active_window = true  (2025-05-01 to 2026-05-01)
"""