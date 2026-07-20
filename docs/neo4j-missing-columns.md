# Neo4j Graph — Missing Columns & Relationships Gap Analysis

**Compared:** DuckDB domain tables (birdeye, campaign_map, charges, insurance, location, patient, patient_navigation_map, phone_bridge, ringcentral, rv_inbound, rv_outbound, statements, transactions, visits) vs actual Neo4j catalog (2026-07-06).

**Value rating** = does adding it help text2cypher answer real user questions?

---

## Patient (from `patient_navigation_map`, 51 props loaded)

| Missing column | Value | Why |
|---|---|---|
| `first_visit_date`, `last_visit_date` | **High** | "Patients not seen in 12 months", recency filters — very common questions, currently needs HAD_VISIT traversal + aggregation |
| `first_statement_date`, `last_statement_date` | **High** | Statement recency / aging without traversal |
| `transaction_count` | Medium | Visit/charge/statement counts loaded but not this one |
| `in_window_charged`, `in_window_paid`, `in_window_charge_count`, `in_window_statement_count` | Medium | Operational-window (12-mo) financials |
| `adj_charity_care`, `adj_payment_plan` | Medium | 5 of 7 adj buckets loaded; complete the set for "adjustments by type" |
| `has_visits`, `has_charges`, `has_transactions`, `has_statements`, `has_email`, `has_phone` | Medium | Cheap existence filters ("patients with no email") |
| `rv_in_calls_window`, `rv_out_calls_window`, `rc_calls_window`, `rc_attributed_calls` | Medium-Low | Channel-level call counts; `total_calls_window` covers most |
| `first_email_sent`, `first_text_sent` | Low | Niche digital-outreach timing |
| `has_inbound_calls`, `has_outbound_calls`, `has_ringcentral`, `has_campaign_assignment`, `has_in_window_activity`, `has_birdeye_at_primary_location` | Low | Redundant with counts/traversal |
| `last_location_id`, all `primary_location_*` | **Skip** | Covered by `HAD_VISIT → PERFORMED_AT` |
| All patient-level `birdeye_*` | **Skip** | Belongs on Location via `REVIEWS` |
| `campaign_*_via_phone` | **Skip** | PhoneBridge traversal covers it |
| `PatientAddress(2)`, raw phones, SSN, Suffix, ResponsibleParty block, `hashed_patient_id`, `_identity_key`, `_cohort` | **Skip** | PII / null / internal |

## Charge (`charges`)

| Missing column | Value | Why |
|---|---|---|
| `ReferringDoctorID`, `DoctorID` | **High** | Provider questions currently **unanswerable** — YAML `schema_notes` falsely claims provider fields exist on Visit. Biggest schema/prompt lie |
| `VoidedDate` | Medium | "Charges voided last month" — `is_voided` exists but no when |
| `NextFollowUpDate` | Medium-Low | A/R follow-up workflow |
| `ChargeCreateDate` | Low | `post_date` covers most timing |
| `ICD10Diagnosis6–10` | Low | 99.8–100% null; `DIAGNOSED_WITH` covers |
| `ChargeStatusBillStage`, `ChargeStatusReleased` | Low | Billing-ops internals |
| `Doctor2ID`, `OtherDoctorID`, `UserName`, `OrderNumber`, `InterpretationLocationID`, `Batchnumber`, `DepartmentCode`, auth numbers, `ActionDescription`, `SpecialUpdateFlag` | **Skip** | Mostly null or housekeeping |

## Transaction (`transactions`)

| Missing column | Value | Why |
|---|---|---|
| `patient_id` (on node) | Medium | Rel covers it, but inline id avoids a hop in aggregations |
| `InsurancePlan` | Medium | "Payments by carrier" without going via Visit |
| `isERSPayment` | Medium | ERS/ERA electronic remittance analysis |
| `RefundAmounts` | Medium-Low | Refund questions |
| `PaymentModule` | Low | Manual vs auto posting |
| `CheckNumber`, `ICNNumber`, `BatchNumber`, `PaymentType`, `PostDateWTime`, `SystemDateWithTime`, `SourceTableName`, `SpecialUpdateFlag` | **Skip** | Reconciliation internals |

## Visit (`visits`)

| Missing column | Value | Why |
|---|---|---|
| `patient_id` (on node) | Medium | Same reasoning as Transaction |
| `TertiaryInsurancePlanNum` | Low-Medium | Primary + secondary loaded but not tertiary — inconsistent |
| `Secondary/Tertiary` policy, auth, group | Low | 84–100% null |

## Statement (`statements`)

| Missing column | Value | Why |
|---|---|---|
| `Statement_Successful` | **High** | Email + text flags loaded but **not mail** — delivery-method answers are wrong (mail is the majority channel) |
| `first_email_sent_date`, `first_text_sent_date` | Low-Medium | Digital delivery timing |
| `HoldNote` | Low | 99.9% null |

## RCCall (`ringcentral`)

| Missing column | Value | Why |
|---|---|---|
| ⚠️ `start_date` stored as **STRING** | **High (fix, not add)** | Date-range filtering broken/string-compare. Load `Start_Date_clean` as DATE_TIME |
| `Disp_Comments` | Low-Medium | Free-text context, PII-heavy — probably skip |
| `Callback_Time`, `PostQueue`, `Routing_Time` | Low | Queue mechanics |
| `Disp_Code`, `Skill_No`, `Campaign_No`, `Agent_No`, `Team_No`, `Contact_Code`, `Master_Contact_ID`, `Media_Name`, `Contact_Name`, raw ANI, junk flags, `Logged` | **Skip** | Names loaded; codes redundant |

## IVRInbound (`rv_inbound`)

| Missing column | Value | Why |
|---|---|---|
| `CallTermLocation` | Medium | Where the IVR session ended — funnel/abandonment analysis |
| `TransferDuration` | Medium | Transferred-to-agent analysis |
| `TransactionTypeDesc` | Low-Medium | Sale vs refund on IVR payments |
| `CallerID` | Low | Phone attribution; PII-ish |
| `Comment`, `PatFirstName`, `PatLastName`, `BusinessUnit` | **Skip** | PII / redundant |

## DiallerCall (`rv_outbound`)

Fully covered — nothing worth adding. (`RESPPHONE_norm` 100% null; names/raw phones = PII, skip.)

## InsurancePlan, Campaign, PhoneBridge, BirdeyeReview

Nothing missing worth adding. Two fixes:
- `BirdeyeReview.date_posted` is STRING — parse to DATE_TIME so date filters work
- `birdeye.phi_ssn_count` is the only unloaded column (Low)

## Location (`location`)

| Missing column | Value | Why |
|---|---|---|
| `birdeye_median_rating`, `birdeye_one_or_two_star_pct` | Medium | Reputation questions beyond avg rating |
| `birdeye_one/two/three/four/five_star_count` | Medium | "Locations with most 1-star reviews" — distribution |
| `birdeye_first/last_review_date` | Low-Medium | Review recency |
| `LocationAddress2`, alternatives, raw phone/fax, `review_with_comment_count`, phi totals | **Skip** | Low signal |

---

## Missing relationships (vs YAML design)

| Relationship | Status | Value |
|---|---|---|
| `SAME_PERSON_AS` (Patient↔Patient) | **Not in graph** | **High** — `multi_practice_flag` exists but no edge to find the same person's other records. Multi-practice questions half-work |
| `DIALLER_PART_OF_CAMPAIGN` (DiallerCall→Campaign) | **Not in graph** | Medium — dialler calls orphaned from campaigns; RCCall→Campaign exists |

## Top 6 actions by impact

1. **Fix `RCCall.start_date` → DATE_TIME** (call date filters currently broken)
2. **Add `Statement.statement_successful`** (mail delivery — delivery questions are wrong without it)
3. **Add `Charge.referring_doctor_id` / `doctor_id`** and fix the YAML `schema_notes.provider` claim
4. **Add Patient first/last visit + statement dates** (recency is a top question class)
5. **Create `SAME_PERSON_AS`** edges (or remove multi-practice claims from the prompt)
6. **Fix `BirdeyeReview.date_posted` → DATE_TIME**