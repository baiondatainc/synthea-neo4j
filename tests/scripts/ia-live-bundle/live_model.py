"""IA knowledge graph - LIVE schema single source of truth.

Generates, from one definition:
  questions.yaml               65 ground-truth queries against the live graph
  ia_live_dictionary.yaml      text2cypher catalog describing the live graph
  Modelfile.text2cypher-ia     Ollama Modelfile (schema/direction blocks generated, rules + few-shots curated)

    python live_model.py
    ollama create text2cypher-ia:latest -f Modelfile.text2cypher-ia
"""
from __future__ import annotations

import datetime as dt
import textwrap

import yaml

# ============================================================================
# 1. LIVE SCHEMA (from db.schema dump, 2026-09-11) + human descriptions
# ============================================================================
# type legend for the prompt: * DATE_TIME (dot notation, compare with datetime())
#                             + DATE      (dot notation, compare with date())
NODES = {
 "Policy": dict(key="policy_id (surrogate); policy_number is the business key", desc="Insurance policy - the anchor node. Carries premium, loss-ratio and claim ROLLUPS.",
   props={"policy_id":"String","policy_number":"String","policy_type":"String","main_type":"String","general_line_of_business":"String",
          "line_of_business":"String","policy_status":"String","is_active":"Boolean","is_cancelled":"Boolean","is_reinsured":"Boolean",
          "is_direct":"Boolean","issue_date":"DateTime","effect_date":"DateTime","renewal_date":"DateTime","expiry_date":"DateTime",
          "cancellation_date":"DateTime","cancellation_reason":"String","policy_term_days":"Long","effect_year":"Long","effect_month":"String",
          "gross_written_premium":"Double","net_written_premium":"Double","commission":"Double","reported_loss_ratio":"Double",
          "calculated_loss_ratio_pct":"Double","product_count":"Long","event_count":"Long","last_event_type":"String","renewal_count":"Long",
          "claim_count":"Long","open_claim_count":"Long","fraud_flagged_claim_count":"Long","max_fraud_probability":"Double",
          "total_claimed":"Double","total_approved":"Double","total_paid":"Double","total_outstanding":"Double","avg_settlement_days":"Double",
          "vehicle_count":"Long","health_member_count":"Long","reinsurance_agreement_count":"Long"},
   rollups=["claim_count","open_claim_count","fraud_flagged_claim_count","total_claimed","total_approved","total_paid","total_outstanding",
            "avg_settlement_days","renewal_count","event_count","vehicle_count","health_member_count","calculated_loss_ratio_pct"]),
 "Insurer": dict(key="insurer_id; ia_id is the IA licence", desc="Licensed insurance company.",
   props={"insurer_id":"String","ia_id":"String","name":"String","name_arab":"String","short_name":"String","cr_id":"String","city":"String",
          "region":"String","country":"String","is_local":"Boolean","is_health":"Boolean","is_motor":"Boolean","is_p_c":"Boolean","is_p_s":"Boolean",
          "lines_of_business":"String","employee_count":"Long","saudization_pct":"Double","number_of_branches":"Long"},
   rollups=["saudization_pct","lines_of_business"]),
 "Reinsurer": dict(key="reinsurer_id", desc="Reinsurance company.",
   props={"reinsurer_id":"String","ia_id":"String","name":"String","name_arab":"String","country":"String","is_local":"Boolean"}),
 "DistributionChannel": dict(key="channel_id", desc="Broker / agency / TPA that sold the policy.",
   props={"channel_id":"String","name":"String","name_arab":"String","channel_type":"String","is_broker":"Boolean","is_agency":"Boolean",
          "is_tpa":"Boolean","is_local":"Boolean","city":"String","region":"String","country":"String"}),
 "Agent": dict(key="agent_id", desc="Individual agent working for a channel. Not linked to policies (source gap).",
   props={"agent_id":"String","name":"String","employment_type":"String","specialization":"String","years_of_experience":"Long","hired_date":"DateTime"}),
 "Product": dict(key="product_mapping_id", desc="Insurer's commercial product, mapped to an IA product.",
   props={"product_mapping_id":"String","name":"String","ia_product_name":"String","is_active":"Boolean","approval_date":"Date","end_date":"Date"}),
 "IAProduct": dict(key="insurance_authority_product_id", desc="Product in the Insurance Authority catalogue.",
   props={"insurance_authority_product_id":"String","name":"String"}),
 "Policyholder": dict(key="policyholder_sk", desc="Party holding a policy - individual OR organisation (policyholder_type).",
   props={"policyholder_sk":"String","name":"String","policyholder_type":"String","lob":"String","identity_type":"String","identity_number":"String",
          "cr_id":"String","vat_no":"String","gender":"String","date_of_birth":"DateTime","nationality":"String","city":"String","region":"String",
          "country":"String","employee_count":"Long","phone":"String","email":"String","source_table":"String","source_id":"Long"}),
 "HealthMember": dict(key="member_id", desc="Health-insured person (primary or dependent).",
   props={"member_id":"String","beneficiary_number":"String","is_primary":"Boolean","is_policyholder":"Boolean","relationship_with_primary":"String",
          "gender":"String","date_of_birth":"Date","age_years":"Long","nationality":"String","city":"String","marital_status":"String",
          "employment_status":"String","inception_date_for_member":"DateTime","policy_product_id":"Long"},
   rollups=["age_years"]),
 "BenefitClass": dict(key="class_id", desc="Benefit tier of a health member (e.g. VIP, A, B).",
   props={"class_id":"String","name":"String","class_type":"String","class_status":"String"}),
 "Vehicle": dict(key="vehicle_id", desc="Insured vehicle.",
   props={"vehicle_id":"String","license_plate_number":"String","plate_type_eng":"String","chassis_number":"String","manufacturer":"String",
          "vehicle_model":"String","vehicle_model_year":"Long","year_manufactured":"Long","vehicle_type":"String","usage_type":"String",
          "coverage_type":"String","seat_number":"Long","region_registration":"String","is_ev":"Boolean","ai_assisted_driving_function":"Boolean",
          "is_new_vehicle":"Boolean","is_corporate":"Boolean","is_imported":"Boolean","is_non_claim_discount_applied":"Boolean",
          "effect_date_for_vehicle":"DateTime","policy_product_id":"Long"}),
 "Claim": dict(key="claim_id; claim_number is the business key", desc="Claim under a policy. Carries health/motor ROLLUPS (settlement_days, days_into_policy, repair cost, shares).",
   props={"claim_id":"String","claim_number":"String","claim_lob":"String","claim_status":"String","is_open":"Boolean","adjudication_outcome":"String",
          "claim_source":"String","claim_creation_date":"DateTime","claim_year":"Long","claim_month":"String","claim_resolution_date":"DateTime",
          "last_payment_date":"DateTime","settlement_days":"Long","days_into_policy":"Long","claimed_amount":"Double","approved_amount":"Double",
          "amount_paid":"Double","amount_rejected":"Double","amount_outstanding":"Double","copay_amount":"Double","deductible_applied":"Double",
          "tax":"Double","approval_rate_pct":"Double","paid_rate_pct":"Double","rejection_reason":"String","fraud_suspicion_indicator":"Boolean",
          "fraud_probability":"Double","pre_auth_count":"Long","pre_auth_requested":"Double","pre_auth_approved":"Double","service_line_count":"Long",
          "health_net_amount":"Double","patient_share_amount":"Double","payer_share_amount":"Double","diagnosis_codes":"String",
          "accident_count":"Long","damage_assessment_count":"Long","est_total_repair_cost_with_vat":"Double","damage_types":"String",
          "total_injuries":"Double","total_fatalities":"Double"},
   rollups=["settlement_days","days_into_policy","claim_year","claim_month","is_open","approval_rate_pct","paid_rate_pct","health_net_amount",
            "patient_share_amount","payer_share_amount","est_total_repair_cost_with_vat","diagnosis_codes","damage_types","pre_auth_count"]),
 "Claimant": dict(key="claimant_id", desc="Party that filed the claim.",
   props={"claimant_id":"String","claimant_type":"String","claimant_identity_type":"String","claimant_identity_number":"String"}),
 "PreAuthorization": dict(key="pre_authorization_id", desc="Prior approval of a health service. Reached ONLY via Claim-[:HAS_PRE_AUTH].",
   props={"pre_authorization_id":"String","pre_authorization_number":"String","pre_authorization_status":"String","adjudication_outcome":"String",
          "pre_authorization_creation_date":"DateTime","pre_authorization_amount":"Double","approved_amount":"Double","amount_rejected":"Double",
          "fraud_suspicion_indicator":"Boolean","fraud_probability":"Double"}),
 "HealthClaimLine": dict(key="health_claim_id", desc="Line item of a health claim.",
   props={"health_claim_id":"String","claim_type":"String","sub_type":"String","product_service":"String","service_date":"DateTime","quantity":"Long",
          "unit_price":"Double","net_amount":"Double","tax_amount":"Double","patient_share_amount":"Double","payer_share_amount":"Double","currency":"String"}),
 "DiagnosisCode": dict(key="code", desc="Diagnosis code on a claim line.", props={"code":"String"}),
 "Accident": dict(key="accident_id", desc="Motor accident behind a claim.",
   props={"accident_id":"String","accident_case_number":"String","accident_datetime":"DateTime","city":"String","latitude":"Double","longitude":"Double",
          "source":"String","parties_count":"Long","injury_count":"Long","fatality_count":"Long"}),
 "DamageAssessment": dict(key="damage_assessment_id", desc="Repair-cost estimate for an accident.",
   props={"damage_assessment_id":"String","da_center":"String","center_city":"String","damage_type":"String","da_initiated_date":"DateTime",
          "da_completed_date":"DateTime","labor_cost":"Double","spare_parts_cost":"Double","total_cost_with_vat":"Double"}),
 "PolicyEvent": dict(key="policy_event_id", desc="Lifecycle event of a policy.",
   props={"policy_event_id":"String","event_type":"String","event_date":"DateTime","event_status":"String","reason":"String","triggered_by":"String"}),
}

# (from, type, to, props, note)
RELS = [
 ("Policy","ISSUED_BY","Insurer",{}, ""),
 ("Policy","SOLD_VIA","DistributionChannel",{}, ""),
 ("Policy","HAS_PRODUCT","Product",{"policy_product_id":"Long"}, ""),
 ("Policy","REINSURED_BY","Reinsurer",{"agreement_id":"Long","agreement_type":"String","coverage_type":"String","quota_share_percentage":"Double",
                                       "excess_of_loss_limit":"Double","effective_date":"DateTime","expiry_date":"DateTime"}, "reinsurance agreement lives on the edge"),
 ("Policy","HAS_EVENT","PolicyEvent",{}, ""),
 ("Product","REGULATED_AS","IAProduct",{}, ""),
 ("Agent","WORKS_FOR","DistributionChannel",{}, ""),
 ("HealthMember","COVERED_BY","Policy",{}, ""),
 ("HealthMember","SPONSORED_BY","Policyholder",{}, "employer / sponsoring organisation"),
 ("HealthMember","IN_CLASS","BenefitClass",{}, ""),
 ("Vehicle","INSURED_UNDER","Policy",{}, ""),
 ("Vehicle","OWNED_BY","Policyholder",{}, ""),
 ("Claim","UNDER_POLICY","Policy",{"policy_product_id":"Long"}, "~10% of claims have no policy edge"),
 ("Claim","FOR_PRODUCT","Product",{}, ""),
 ("Claim","FILED_BY","Claimant",{}, ""),
 ("Claim","HAS_PRE_AUTH","PreAuthorization",{}, ""),
 ("Claim","HAS_SERVICE_LINE","HealthClaimLine",{}, ""),
 ("Claim","INVOLVES_ACCIDENT","Accident",{}, ""),
 ("HealthClaimLine","DIAGNOSED_WITH","DiagnosisCode",{}, ""),
 ("Accident","ASSESSED_BY","DamageAssessment",{}, ""),
]

VALUES = {  # placeholders in questions.yaml - PROFILE these
 "policy_active_status": "active", "claim_open_status": "open", "claim_rejected_status": "rejected",
 "adj_approved": "approved", "lob_motor": "motor", "loss_ratio_full": 100, "fraud_prob_high": 80,
 "journey_policy_number": "CTM5DQ3K",
}

# ============================================================================
# 2. QUESTIONS  (id, category, question, compare, cypher, extra)
# ============================================================================
Q = []
def q(id_, cat, question, compare, cypher, **kw):
    Q.append(dict(id=id_, category=cat, question=question, compare=compare, expected_cypher=textwrap.dedent(cypher).strip(), **kw))

q("Q01","policy","How many policies?","count","MATCH (p:Policy) RETURN count(p) AS policy_count")
q("Q02","policy","How many active policies?","count","MATCH (p:Policy) WHERE p.is_active = true RETURN count(p) AS active_policies")
q("Q03","policy","Policies by status","rows","""
   MATCH (p:Policy) RETURN p.policy_status AS status, count(p) AS policy_count ORDER BY policy_count DESC""")
q("Q04","policy","Policies by line of business","rows","""
   MATCH (p:Policy) RETURN p.line_of_business AS line_of_business, count(p) AS policy_count ORDER BY policy_count DESC""")
q("Q05","policy","Top 10 insurers by number of policies","topk","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer)
   RETURN i.name AS insurer, count(p) AS policy_count ORDER BY policy_count DESC, insurer LIMIT 10""", topk=10)
q("Q06","policy","Gross written premium by insurer","rows","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer)
   RETURN i.name AS insurer, round(sum(p.gross_written_premium), 2) AS gwp ORDER BY gwp DESC""")
q("Q07","policy","Loss ratio by insurer","rows","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer)
   WITH i.name AS insurer, sum(p.total_paid) AS paid, sum(p.gross_written_premium) AS gwp
   WHERE gwp > 0
   RETURN insurer, round(paid * 100.0 / gwp, 2) AS loss_ratio_pct ORDER BY loss_ratio_pct DESC""",
  tolerance=0.5, note="premium-weighted: sum(total_paid)/sum(gwp) per insurer, matching mart_insurer_monthly_kpi. NEVER avg(calculated_loss_ratio_pct) at group level - one tiny-premium policy distorts the mean (avg-of-ratios vs ratio-of-sums)")
q("Q08","policy","Loss ratio by line of business","rows","""
   MATCH (p:Policy)
   WITH p.line_of_business AS line_of_business, sum(p.total_paid) AS paid, sum(p.gross_written_premium) AS gwp
   WHERE gwp > 0
   RETURN line_of_business, round(paid * 100.0 / gwp, 2) AS loss_ratio_pct ORDER BY loss_ratio_pct DESC""",
  tolerance=0.5, note="premium-weighted, same rule as Q07")
q("Q09","policy","Policies with loss ratio over 100 percent","rows","""
   MATCH (p:Policy) WHERE p.calculated_loss_ratio_pct > ${loss_ratio_full}
   RETURN p.policy_number AS policy_number, p.calculated_loss_ratio_pct AS loss_ratio_pct ORDER BY loss_ratio_pct DESC""",
  note="per-policy filter - the ONE legitimate use of calculated_loss_ratio_pct")
q("Q10","policy","Policies by distribution channel type","rows","""
   MATCH (p:Policy)-[:SOLD_VIA]->(c:DistributionChannel)
   RETURN c.channel_type AS channel_type, count(p) AS policy_count ORDER BY policy_count DESC""")
q("Q11","policy","Top brokers by premium","topk","""
   MATCH (p:Policy)-[:SOLD_VIA]->(c:DistributionChannel) WHERE c.is_broker = true
   RETURN c.name AS broker, round(sum(p.gross_written_premium), 2) AS gwp ORDER BY gwp DESC, broker LIMIT 10""", topk=10)
q("Q12","policy","Monthly policy trend","rows","""
   MATCH (p:Policy) WHERE p.issue_date IS NOT NULL
   RETURN p.issue_date.year AS year, p.issue_date.month AS month, count(p) AS policy_count ORDER BY year, month""")
q("Q13","policy","Yearly policy trend","rows","""
   MATCH (p:Policy) WHERE p.issue_date IS NOT NULL
   RETURN p.issue_date.year AS year, count(p) AS policy_count ORDER BY year""")
q("Q14","policy","Premium trend by quarter","rows","""
   MATCH (p:Policy) WHERE p.effect_date IS NOT NULL
   RETURN p.effect_date.year AS year, (p.effect_date.month - 1) / 3 + 1 AS quarter,
          round(sum(p.gross_written_premium), 2) AS gwp ORDER BY year, quarter""")
q("Q15","policy","Policy trend by insurer","rows","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer) WHERE p.issue_date IS NOT NULL
   RETURN i.name AS insurer, p.issue_date.year AS year, count(p) AS policy_count ORDER BY insurer, year""")
q("Q16","policy","Policies expiring in the next 90 days","count","""
   MATCH (p:Policy) WHERE p.expiry_date >= datetime() AND p.expiry_date <= datetime() + duration({days: 90})
   RETURN count(p) AS expiring_soon""", note="date-relative")
q("Q17","policy","Cancelled policies and their reasons","rows","""
   MATCH (p:Policy) WHERE p.is_cancelled = true
   RETURN p.cancellation_reason AS reason, count(p) AS policy_count ORDER BY policy_count DESC""")
q("Q18","policy","Policies that were renewed more than once","rows","""
   MATCH (p:Policy) WHERE p.renewal_count > 1
   RETURN p.policy_number AS policy_number, p.renewal_count AS renewal_count ORDER BY renewal_count DESC, policy_number""")

q("Q19","claims","How many claims?","count","MATCH (c:Claim) RETURN count(c) AS claim_count")
q("Q20","claims","Claims by status","rows","""
   MATCH (c:Claim) RETURN c.claim_status AS status, count(c) AS claim_count ORDER BY claim_count DESC""")
q("Q21","claims","Open claims with outstanding amount","rows","""
   MATCH (c:Claim) WHERE c.is_open = true AND c.amount_outstanding > 0
   RETURN c.claim_number AS claim_number, c.amount_outstanding AS amount_outstanding ORDER BY amount_outstanding DESC""")
q("Q22","claims","Monthly claim trend","rows","""
   MATCH (c:Claim) WHERE c.claim_creation_date IS NOT NULL
   RETURN c.claim_creation_date.year AS year, c.claim_creation_date.month AS month, count(c) AS claim_count ORDER BY year, month""")
q("Q23","claims","Claims by insurer","rows","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(:Policy)-[:ISSUED_BY]->(i:Insurer)
   RETURN i.name AS insurer, count(c) AS claim_count ORDER BY claim_count DESC""")
q("Q24","claims","Average settlement days by insurer","rows","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(:Policy)-[:ISSUED_BY]->(i:Insurer) WHERE c.settlement_days IS NOT NULL
   RETURN i.name AS insurer, round(avg(c.settlement_days), 1) AS avg_settlement_days ORDER BY avg_settlement_days""", tolerance=0.5)
q("Q25","claims","Claim approval rate by product","rows","""
   MATCH (c:Claim)-[:FOR_PRODUCT]->(pr:Product)
   WITH pr, count(c) AS total, sum(CASE WHEN c.adjudication_outcome = '${adj_approved}' THEN 1 ELSE 0 END) AS approved
   RETURN pr.name AS product, round(toFloat(approved) / total, 2) AS approval_rate ORDER BY approval_rate DESC""")
q("Q26","claims","Rejected claims by reason","rows","""
   MATCH (c:Claim) WHERE c.claim_status = '${claim_rejected_status}'
   RETURN c.rejection_reason AS reason, count(c) AS claim_count ORDER BY claim_count DESC""")
q("Q27","fraud","How many fraud-flagged claims?","count","MATCH (c:Claim) WHERE c.fraud_suspicion_indicator = true RETURN count(c) AS fraud_flagged")
q("Q28","fraud","Fraud-flagged claims by insurer","rows","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(:Policy)-[:ISSUED_BY]->(i:Insurer) WHERE c.fraud_suspicion_indicator = true
   RETURN i.name AS insurer, count(c) AS fraud_flagged ORDER BY fraud_flagged DESC""")
q("Q29","fraud","Claims with fraud probability above 0.8","rows","""
   MATCH (c:Claim) WHERE c.fraud_probability > ${fraud_prob_high}
   RETURN c.claim_number AS claim_number, c.fraud_probability AS fraud_probability ORDER BY fraud_probability DESC""")
q("Q30","fraud","Policyholders with the most fraud-flagged motor claims","topk","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(p:Policy)<-[:INSURED_UNDER]-(v:Vehicle)-[:OWNED_BY]->(ph:Policyholder)
   WHERE c.fraud_suspicion_indicator = true AND p.line_of_business = '${lob_motor}'
   RETURN ph.name AS policyholder, count(DISTINCT c) AS fraud_flagged_motor_claims
   ORDER BY fraud_flagged_motor_claims DESC, policyholder LIMIT 10""", topk=10)
q("Q31","claims","Claims with no policy","count","""
   MATCH (c:Claim) WHERE NOT (c)-[:UNDER_POLICY]->(:Policy) RETURN count(c) AS orphan_claims""",
  note="2689 of 3000 claims have UNDER_POLICY - expect ~311")
q("Q32","claims","Top 10 policies by claims paid","topk","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(p:Policy)
   RETURN p.policy_number AS policy_number, round(sum(c.amount_paid), 2) AS claims_paid ORDER BY claims_paid DESC, policy_number LIMIT 10""", topk=10)
q("Q33","claims","Policies with more than 5 claims","rows","""
   MATCH (c:Claim)-[:UNDER_POLICY]->(p:Policy) WITH p, count(c) AS claim_count WHERE claim_count > 5
   RETURN p.policy_number AS policy_number, claim_count ORDER BY claim_count DESC, policy_number""")
q("Q34","journey","Show me the journey of policy ${journey_policy_number}","nonempty","""
   MATCH (p:Policy {policy_number: '${journey_policy_number}'})
   OPTIONAL MATCH (p)-[:HAS_EVENT]->(e:PolicyEvent)
   OPTIONAL MATCH (c:Claim)-[:UNDER_POLICY]->(p)
   RETURN p.policy_number AS policy_number, p.issue_date AS issued, p.policy_status AS status,
          collect(DISTINCT {date: e.event_date, type: e.event_type}) AS events,
          collect(DISTINCT {claim: c.claim_number, date: c.claim_creation_date, status: c.claim_status}) AS claims""")

q("Q35","motor","How many vehicles are insured?","count","MATCH (v:Vehicle)-[:INSURED_UNDER]->(:Policy) RETURN count(DISTINCT v) AS insured_vehicles")
q("Q36","motor","Vehicles by manufacturer","rows","""
   MATCH (v:Vehicle)-[:INSURED_UNDER]->(:Policy)
   RETURN v.manufacturer AS manufacturer, count(DISTINCT v) AS vehicle_count ORDER BY vehicle_count DESC""")
q("Q37","motor","How many electric vehicles are insured?","count","""
   MATCH (v:Vehicle)-[:INSURED_UNDER]->(:Policy) WHERE v.is_ev = true RETURN count(DISTINCT v) AS electric_vehicles""")
q("Q38","motor","Vehicles with AI assisted driving by insurer","rows","""
   MATCH (v:Vehicle)-[:INSURED_UNDER]->(:Policy)-[:ISSUED_BY]->(i:Insurer) WHERE v.ai_assisted_driving_function = true
   RETURN i.name AS insurer, count(DISTINCT v) AS vehicle_count ORDER BY vehicle_count DESC""")
q("Q39","motor","Accidents by city","rows","""
   MATCH (a:Accident) WHERE a.city IS NOT NULL RETURN a.city AS city, count(a) AS accident_count ORDER BY accident_count DESC""")
q("Q40","motor","Accidents with fatalities","rows","""
   MATCH (a:Accident) WHERE a.fatality_count > 0
   RETURN a.accident_case_number AS case_number, a.accident_datetime AS accident_datetime, a.fatality_count AS fatalities
   ORDER BY fatalities DESC, case_number""")
q("Q41","motor","Total repair cost by damage type","rows","""
   MATCH (d:DamageAssessment)
   RETURN d.damage_type AS damage_type, round(sum(d.total_cost_with_vat), 2) AS total_repair_cost ORDER BY total_repair_cost DESC""")
q("Q42","motor","Claims with highest estimated repair cost","topk","""
   MATCH (c:Claim)-[:INVOLVES_ACCIDENT]->(:Accident)-[:ASSESSED_BY]->(d:DamageAssessment)
   RETURN c.claim_number AS claim_number, round(sum(d.total_cost_with_vat), 2) AS repair_cost ORDER BY repair_cost DESC, claim_number LIMIT 10""", topk=10)
q("Q43","motor","Policyholders with vehicles insured at more than one insurer","rows","""
   MATCH (ph:Policyholder)<-[:OWNED_BY]-(:Vehicle)-[:INSURED_UNDER]->(:Policy)-[:ISSUED_BY]->(i:Insurer)
   WITH ph, count(DISTINCT i) AS insurer_count WHERE insurer_count > 1
   RETURN ph.name AS policyholder, insurer_count ORDER BY insurer_count DESC, policyholder""")

q("Q44","health","How many health members?","count","MATCH (m:HealthMember) RETURN count(m) AS health_members")
q("Q45","health","Health members by benefit class","rows","""
   MATCH (m:HealthMember)-[:IN_CLASS]->(b:BenefitClass) RETURN b.name AS benefit_class, count(m) AS member_count ORDER BY member_count DESC""")
q("Q46","health","Members by relationship to primary","rows","""
   MATCH (m:HealthMember) WHERE m.relationship_with_primary IS NOT NULL
   RETURN m.relationship_with_primary AS relationship, count(m) AS member_count ORDER BY member_count DESC""")
q("Q47","health","Average member age by insurer","rows","""
   MATCH (m:HealthMember)-[:COVERED_BY]->(:Policy)-[:ISSUED_BY]->(i:Insurer) WHERE m.age_years IS NOT NULL
   RETURN i.name AS insurer, round(avg(m.age_years), 1) AS avg_age ORDER BY avg_age DESC""", tolerance=0.5)
q("Q48","health","Members older than 60","count","MATCH (m:HealthMember) WHERE m.age_years > 60 RETURN count(m) AS members_over_60")
q("Q49","health","Organisations sponsoring the most health members","topk","""
   MATCH (m:HealthMember)-[:SPONSORED_BY]->(ph:Policyholder)
   RETURN ph.name AS organisation, count(m) AS member_count ORDER BY member_count DESC, organisation LIMIT 10""", topk=10)
q("Q50","health","Top diagnosis codes by claim count","topk","""
   MATCH (c:Claim)-[:HAS_SERVICE_LINE]->(:HealthClaimLine)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)
   RETURN d.code AS diagnosis, count(DISTINCT c) AS claim_count ORDER BY claim_count DESC, diagnosis LIMIT 10""", topk=10)
q("Q51","health","Health claims cost split between patient and payer","value","""
   MATCH (l:HealthClaimLine)
   RETURN round(sum(l.patient_share_amount), 2) AS patient_share, round(sum(l.payer_share_amount), 2) AS payer_share""")
q("Q52","health","Pre-authorization approval rate by status","rows","""
   MATCH (pa:PreAuthorization)
   WITH pa.pre_authorization_status AS status, count(pa) AS total,
        sum(CASE WHEN pa.adjudication_outcome = '${adj_approved}' THEN 1 ELSE 0 END) AS approved
   RETURN status, total, round(toFloat(approved) / total, 2) AS approval_rate ORDER BY total DESC""")

q("Q53","products","Products by IA product","rows","""
   MATCH (pr:Product)-[:REGULATED_AS]->(ia:IAProduct)
   RETURN ia.name AS ia_product, count(pr) AS product_count ORDER BY product_count DESC""")
q("Q54","products","Policies per IA product","rows","""
   MATCH (p:Policy)-[:HAS_PRODUCT]->(:Product)-[:REGULATED_AS]->(ia:IAProduct)
   RETURN ia.name AS ia_product, count(DISTINCT p) AS policy_count ORDER BY policy_count DESC""")
q("Q55","products","Inactive products still attached to active policies","rows","""
   MATCH (p:Policy)-[:HAS_PRODUCT]->(pr:Product) WHERE p.is_active = true AND pr.is_active = false
   RETURN pr.name AS product, count(DISTINCT p) AS active_policies ORDER BY active_policies DESC, product""")
q("Q56","reinsurance","Reinsurers by number of policies reinsured","rows","""
   MATCH (p:Policy)-[:REINSURED_BY]->(r:Reinsurer)
   RETURN r.name AS reinsurer, count(DISTINCT p) AS policy_count ORDER BY policy_count DESC""")
q("Q57","reinsurance","Average quota share by reinsurer","rows","""
   MATCH (:Policy)-[x:REINSURED_BY]->(r:Reinsurer) WHERE x.quota_share_percentage IS NOT NULL
   RETURN r.name AS reinsurer, round(avg(x.quota_share_percentage), 2) AS avg_quota_share ORDER BY avg_quota_share DESC""", tolerance=0.5)
q("Q58","reinsurance","Reinsured policies by agreement type","rows","""
   MATCH (p:Policy)-[x:REINSURED_BY]->(:Reinsurer)
   RETURN x.agreement_type AS agreement_type, count(DISTINCT p) AS policy_count ORDER BY policy_count DESC""")
q("Q59","market","Agents per distribution channel","rows","""
   MATCH (a:Agent)-[:WORKS_FOR]->(c:DistributionChannel) RETURN c.name AS channel, count(a) AS agent_count ORDER BY agent_count DESC""")
q("Q60","policy","Policy events by type","rows","""
   MATCH (:Policy)-[:HAS_EVENT]->(e:PolicyEvent) RETURN e.event_type AS event_type, count(e) AS event_count ORDER BY event_count DESC""")
q("Q61","market","Insurers by saudization percentage","rows","""
   MATCH (i:Insurer) WHERE i.saudization_pct IS NOT NULL
   RETURN i.name AS insurer, i.saudization_pct AS saudization_pct ORDER BY saudization_pct DESC""", tolerance=0.2)
q("Q62","market","Local versus foreign insurers by premium","rows","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer)
   RETURN CASE WHEN i.is_local THEN 'local' ELSE 'foreign' END AS origin, round(sum(p.gross_written_premium), 2) AS gwp ORDER BY gwp DESC""")
q("Q63","claims","Claims by claimant type","rows","""
   MATCH (c:Claim)-[:FILED_BY]->(cl:Claimant) RETURN cl.claimant_type AS claimant_type, count(c) AS claim_count ORDER BY claim_count DESC""")
q("Q64","claims","Claims filed within 30 days of policy start","count","""
   MATCH (c:Claim) WHERE c.days_into_policy >= 0 AND c.days_into_policy <= 30 RETURN count(c) AS early_claims""")
q("Q65","policy","Top 5 insurers by loss ratio with at least 50 policies","topk","""
   MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer)
   WITH i.name AS insurer, sum(p.total_paid) AS paid, sum(p.gross_written_premium) AS gwp, count(p) AS policy_count
   WHERE gwp > 0 AND policy_count >= 50
   RETURN insurer, round(paid * 100.0 / gwp, 2) AS loss_ratio_pct, policy_count
   ORDER BY loss_ratio_pct DESC, insurer LIMIT 5""", topk=5,
  note="teaches the weighted formula + a HAVING-style filter; the demo's refinement question")

# ============================================================================
# 3. Emit questions.yaml
# ============================================================================
def write_questions():
    doc = {
        "values": VALUES,
        "defaults": {"tolerance": 0.01, "round_to": 2},
        "questions": Q,
    }
    with open("questions.yaml", "w") as f:
        f.write("# generated by live_model.py from the LIVE Neo4j schema - do not hand-edit\n")
        yaml.dump(doc, f, sort_keys=False, width=120, default_flow_style=False)

# ============================================================================
# 4. Emit dictionary
# ============================================================================
def write_dictionary():
    labels = {}
    for lbl, n in NODES.items():
        labels[lbl] = dict(
            description=n["desc"], key=n["key"],
            properties={k: v for k, v in n["props"].items()},
            rollups=n.get("rollups", []),
            relationships_out=[f"-[:{t}]->(:{to})" for f_, t, to, *_ in RELS if f_ == lbl],
            relationships_in=[f"<-[:{t}]-(:{f_})" for f_, t, to, *_ in RELS if to == lbl],
        )
    rels = [dict(type=t, **{"from": f_}, to=to, **({"properties": p} if p else {}), **({"note": n} if n else {})) for f_, t, to, p, n in RELS]
    doc = dict(
        catalog=dict(name="IA Insurance Knowledge Graph (live)", version="live-1.1", generated=str(dt.date.today()),
                     source="db.schema dump of bolt://localhost:7687"),
        schema_notes=dict(
            rollups="Policy and Claim carry precomputed rollups (claim_count, total_paid, settlement_days, days_into_policy, ...). Use the flat property for policy-level or claim-level totals; traverse only for per-entity breakdowns.",
            loss_ratio="calculated_loss_ratio_pct is PER-POLICY only (filter/inspect single policies). Any group-level loss ratio (by insurer/LoB/channel/region) = sum(total_paid) * 100.0 / sum(gross_written_premium) with a gwp > 0 guard - never avg the per-policy pct (one tiny-premium policy distorts the mean). This matches mart_insurer_monthly_kpi.",
            claims_to_insurer="(Claim)-[:UNDER_POLICY]->(Policy)-[:ISSUED_BY]->(Insurer). ~10% of claims have no UNDER_POLICY edge.",
            pre_auth="PreAuthorization is reachable ONLY through (Claim)-[:HAS_PRE_AUTH]->; there is no policy edge on it.",
            policyholder="One Policyholder label for individuals and organisations (policyholder_type). Motor policyholders: (Vehicle)-[:OWNED_BY]->(Policyholder). Health sponsors: (HealthMember)-[:SPONSORED_BY]->(Policyholder).",
            dates="DateTime properties use dot notation (p.issue_date.year) and compare with datetime(). Date properties (HealthMember.date_of_birth, Product.approval_date/end_date) compare with date().",
            geography="No City/Region nodes - city/region are string properties on Insurer, DistributionChannel, Policyholder, HealthMember, Accident, DamageAssessment.center_city.",
            value_sets="Statuses/enums are NOT enumerated here. Profile with SELECT DISTINCT before injecting into the prompt.",
            not_modelled="Premium is a Policy property (gross_written_premium), not a node. No SparePart, Contract, Membership, HCPNetwork, HealthcareProvider labels in this load.",
        ),
        node_labels=labels, relationships=rels,
        value_sets={k: "PROFILE" for k in ["Policy.policy_status","Policy.line_of_business","Policy.policy_type","Claim.claim_status",
                     "Claim.adjudication_outcome","Claim.claim_lob","PreAuthorization.pre_authorization_status","DistributionChannel.channel_type",
                     "Policyholder.policyholder_type","Vehicle.coverage_type","PolicyEvent.event_type","REINSURED_BY.agreement_type","Claimant.claimant_type"]},
        common_paths=[dict(name=x["id"], question=x["question"], cypher=x["expected_cypher"]) for x in Q if x["id"] in
                      ("Q07","Q23","Q30","Q34","Q43","Q49","Q50","Q52","Q55")],
    )
    with open("ia_live_dictionary.yaml", "w") as f:
        yaml.dump(doc, f, sort_keys=False, width=120, default_flow_style=False, allow_unicode=True)

# ============================================================================
# 5. Emit Modelfile
# ============================================================================
TYPE_MARK = {"DateTime": "*", "Date": "+"}
FEWSHOT_IDS = ["Q01","Q02","Q03","Q05","Q06","Q07","Q08","Q09","Q10","Q11","Q12","Q13","Q14","Q16","Q17","Q18","Q19","Q21","Q22","Q23","Q24",
               "Q25","Q27","Q28","Q30","Q31","Q32","Q33","Q35","Q37","Q38","Q39","Q42","Q43","Q45","Q47","Q48","Q49","Q50","Q51","Q52",
               "Q53","Q55","Q57","Q58","Q59","Q60","Q61","Q62","Q63","Q64","Q65"]

def schema_block():
    lines = []
    for lbl, n in NODES.items():
        props = ", ".join(f"{k}{TYPE_MARK.get(v, '')}" for k, v in n["props"].items())
        lines.append(f"  {lbl:<20}— {props}")
    return "\n".join(lines)

def rollup_block():
    out = []
    for lbl, n in NODES.items():
        if n.get("rollups"):
            out.append(f"  {lbl}: " + ", ".join(n["rollups"]))
    return "\n".join(out)

def rel_block():
    out = []
    for f_, t, to, p, note in RELS:
        pp = " {" + ", ".join(p) + "}" if p else ""
        out.append(f"  ({f_})-[:{t}{pp}]->({to})" + (f"   # {note}" if note else ""))
    return "\n".join(out)

def direction_block():
    out = []
    for f_, t, to, *_ in RELS:
        a, b = f_[0].lower(), to[0].lower() if to[0].lower() != f_[0].lower() else to[:2].lower()
        out.append(f"  ({a}:{f_})-[:{t}]->({b}:{to})\n     WRONG: ({b}:{to})-[:{t}]->({a}:{f_})")
    return "\n".join(out)

def fewshots():
    vals = {k: str(v) for k, v in VALUES.items()}
    import re
    out = []
    for x in Q:
        if x["id"] in FEWSHOT_IDS:
            qq = re.sub(r"\$\{(\w+)\}", lambda m: vals[m.group(1)], x["question"])
            cy = re.sub(r"\$\{(\w+)\}", lambda m: vals[m.group(1)], x["expected_cypher"])
            out.append(f"Q: {qq}\n{cy}\n")
    return "\n".join(out)

MODELFILE = '''# IA Cypher generator — Qwen3-4B-Instruct (non-thinking)
# GENERATED by live_model.py from the live Neo4j schema. Do not hand-edit the
# SCHEMA / RELATIONSHIPS / DIRECTION blocks — edit live_model.py and regenerate.
#
#   ollama create text2cypher-ia:latest -f Modelfile.text2cypher-ia
#   ollama run text2cypher-ia "How many policies?"
FROM qwen3:4b-instruct-2507-q4_K_M

PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 256
PARAMETER num_ctx 12288
PARAMETER num_thread 4
PARAMETER num_batch 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "\\nQ:"

SYSTEM """You are a Neo4j Cypher generator for the Saudi Insurance Authority (IA) knowledge graph. Output ONLY raw Cypher — no markdown, no backticks, no SQL, no explanation, no prose before or after the query.

NEVER produce:
- SQL keywords or SQL date functions (SELECT, JOIN, GROUP BY, YEAR(x), MONTH(x), EXTRACT, DATEPART)
  WRONG:   RETURN YEAR(p.issue_date)      CORRECT: RETURN p.issue_date.year AS year
- Write clauses: CREATE, MERGE, SET, DELETE, REMOVE, DROP, LOAD CSV
- Aggregation inside ORDER BY: ORDER BY count(p) is wrong — alias it and order by the alias
- Unaliased dotted properties or aggregations in RETURN
- Markdown fences or more than ONE statement per response
- Labels, relationship types or properties not in the SCHEMA below (e.g. HealthPolicy, Coverage, Person, Premium — they do not exist)
- Reversed relationship arrows (see DIRECTION block) — a reversed arrow returns zero rows
- datetime() wrapping or substring() on native DateTime properties:
  WRONG:   datetime(p.issue_date).year     substring(p.issue_date, 0, 4)
  CORRECT: p.issue_date.year
- Comparing a Date (+) property with datetime(): m.date_of_birth < date() - duration({years: 60})
- avg(p.calculated_loss_ratio_pct) for ANY group-level loss ratio ("by insurer" / "by line of business" /
  "by channel" / "by region") — one tiny-premium policy distorts the mean (avg-of-ratios ≠ ratio-of-sums)
  WRONG:   RETURN i.name AS insurer, round(avg(p.calculated_loss_ratio_pct), 2) AS avg_loss_ratio_pct
  CORRECT: WITH i.name AS insurer, sum(p.total_paid) AS paid, sum(p.gross_written_premium) AS gwp
           WHERE gwp > 0 RETURN insurer, round(paid * 100.0 / gwp, 2) AS loss_ratio_pct
- LIMIT on an aggregated "by X" result — "claims by insurer" returns every insurer, no LIMIT
- Extra columns the user did not ask for — "count of X" returns ONE count column, nothing else
- Row listings when the user asked "how many" — return count(...) only
- Recomputing a value that exists as a ROLLUP (see ROLLUPS block)
- Reaching PreAuthorization from Policy — it hangs off Claim only: (c:Claim)-[:HAS_PRE_AUTH]->(pa:PreAuthorization)
- Age arithmetic on HealthMember — use m.age_years directly

ALWAYS produce:
- Start with MATCH / OPTIONAL MATCH / WITH / CALL / UNWIND
- Alias every aggregation and dotted property; use the alias in ORDER BY
- IS NOT NULL guard before accessing date parts on nullable dates
- round(sum(...), 2) and round(avg(...), 2) for money and ratios
- DateTime (*) properties: dot notation (c.claim_creation_date.month) and datetime() for comparisons
  p.expiry_date <= datetime() + duration({days: 90})
- Date (+) properties: compare with date()
- Quarter: (x.month - 1) / 3 + 1 AS quarter
- Boolean checks as = true / = false
- A trailing LIMIT 100 only for non-aggregated row listings
- For "top N": ORDER BY the metric DESC, then the name, then LIMIT N

═══════════════════════════════════════════════════
ROLLUPS — flat properties beat traversal
═══════════════════════════════════════════════════
These are precomputed. Use the property, do not traverse+aggregate:
__ROLLUPS__
  active policy        → p.is_active = true          cancelled → p.is_cancelled = true
  premium              → p.gross_written_premium (Premium is NOT a node)
  loss ratio (ONE policy)     → p.calculated_loss_ratio_pct  (reported_loss_ratio = insurer-reported)
  loss ratio (by ANY grouping) → sum(p.total_paid) * 100.0 / sum(p.gross_written_premium) with WHERE gwp > 0
                                 — NEVER avg the per-policy pct
  renewed more than once → p.renewal_count > 1
  open claim           → c.is_open = true
  settlement time      → c.settlement_days
  claim within N days of policy start → c.days_into_policy <= N
  saudization          → i.saudization_pct
Traverse ONLY for per-entity breakdowns (by insurer, by product, by city, by policyholder) or line-level detail.

═══════════════════════════════════════════════════
TREND QUERIES — time is ALWAYS the x-axis
═══════════════════════════════════════════════════
"trend"/"monthly"/"by month" → group by year AND month; "yearly"/"annual"/"by year" → year only;
"by quarter" → year + quarter. Time columns come FIRST in RETURN and FIRST in ORDER BY.
Policy trends use issue_date; premium trends use effect_date; claim trends use claim_creation_date.
Do not add insurer/channel/product unless the user asks "by insurer" etc.

═══════════════════════════════════════════════════
GRAPH SCHEMA   (* = DateTime, + = Date)
═══════════════════════════════════════════════════
__SCHEMA__

Relationships (memorise — wrong hops return empty results):
__RELS__

═══════════════════════════════════════════════════
RELATIONSHIP DIRECTION — arrows point ONE way only
═══════════════════════════════════════════════════
Lead the MATCH with the LEFT node even when the question names the right one first
("claims by insurer" still starts at Claim; "members by class" still starts at HealthMember).
__DIRECTION__

MULTI-HOP PATHS:
  Claim → Insurer        : (c:Claim)-[:UNDER_POLICY]->(p:Policy)-[:ISSUED_BY]->(i:Insurer)
  Claim → IA product     : (c:Claim)-[:FOR_PRODUCT]->(pr:Product)-[:REGULATED_AS]->(ia:IAProduct)
  Policy → IA product    : (p:Policy)-[:HAS_PRODUCT]->(pr:Product)-[:REGULATED_AS]->(ia:IAProduct)
  Vehicle → Insurer      : (v:Vehicle)-[:INSURED_UNDER]->(p:Policy)-[:ISSUED_BY]->(i:Insurer)
  Motor policyholder     : (v:Vehicle)-[:OWNED_BY]->(ph:Policyholder)
  Motor claim → owner    : (c:Claim)-[:UNDER_POLICY]->(p:Policy)<-[:INSURED_UNDER]-(v:Vehicle)-[:OWNED_BY]->(ph:Policyholder)
  Member → Insurer       : (m:HealthMember)-[:COVERED_BY]->(p:Policy)-[:ISSUED_BY]->(i:Insurer)
  Member → Sponsor       : (m:HealthMember)-[:SPONSORED_BY]->(ph:Policyholder)
  Claim → Diagnosis      : (c:Claim)-[:HAS_SERVICE_LINE]->(l:HealthClaimLine)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)
  Claim → Repair cost    : (c:Claim)-[:INVOLVES_ACCIDENT]->(a:Accident)-[:ASSESSED_BY]->(d:DamageAssessment)
  Reinsurance terms      : (p:Policy)-[x:REINSURED_BY]->(r:Reinsurer)  → x.agreement_type, x.quota_share_percentage

═══════════════════════════════════════════════════
VOCABULARY
═══════════════════════════════════════════════════
  "insurer" / "company" / "carrier"            → Insurer
  "broker" / "agency" / "TPA" / "channel"      → DistributionChannel (is_broker / is_agency / is_tpa)
  "IA product" / "authority product"           → IAProduct  (via Product-[:REGULATED_AS])
  "product"                                    → Product
  "policyholder" / "owner" / "organisation" / "sponsor" / "employer" → Policyholder
  "member" / "beneficiary" / "dependent"       → HealthMember
  "class" / "benefit class" / "tier"           → BenefitClass
  "pre-auth" / "pre-authorization" / "prior approval" → PreAuthorization (via Claim)
  "diagnosis" / "ICD"                          → DiagnosisCode (via HealthClaimLine)
  "repair cost" / "damage assessment"          → DamageAssessment (via Accident)
  "premium" / "GWP"                            → Policy.gross_written_premium
  "fraud" / "suspicious"                       → fraud_suspicion_indicator = true / fraud_probability

═══════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════

__FEWSHOTS__"""
'''

def write_modelfile():
    txt = (MODELFILE.replace("__ROLLUPS__", rollup_block()).replace("__SCHEMA__", schema_block())
           .replace("__RELS__", rel_block()).replace("__DIRECTION__", direction_block()).replace("__FEWSHOTS__", fewshots()))
    open("Modelfile.text2cypher-ia", "w").write(txt)


# ============================================================================
# 6. Emit data_dictionary.yaml in the app's catalog.py format (RP layout)
# ============================================================================
PII = {
 "Policyholder": ["identity_number", "phone", "email", "date_of_birth", "vat_no"],
 "HealthMember": ["date_of_birth", "beneficiary_number"],
 "Claimant": ["claimant_identity_number"],
 "Vehicle": ["license_plate_number", "chassis_number"],
}
PROP_DESC = {  # friendly descriptions for the properties users ask about; the rest get a generic one
 "Policy.is_active": "True if the policy is currently in force. Use for 'active policies'.",
 "Policy.is_cancelled": "True if cancelled. Pair with cancellation_reason.",
 "Policy.gross_written_premium": "GWP in SAR, excl. tax. Premium is a property, NOT a node.",
 "Policy.calculated_loss_ratio_pct": "PER-POLICY loss ratio percent (total_paid / GWP * 100). For any group-level loss ratio use sum(total_paid)*100/sum(gwp) - NEVER avg this.",
 "Policy.reported_loss_ratio": "Loss ratio as reported by the insurer.",
 "Policy.renewal_count": "Number of renewals. 'Renewed more than once' -> renewal_count > 1.",
 "Policy.claim_count": "Rollup: number of claims under the policy.",
 "Policy.total_paid": "Rollup: claims paid under the policy. Numerator of group-level loss ratios.",
 "Policy.line_of_business": "health | motor | pc | ps (profile actual values).",
 "Claim.is_open": "True while the claim is unresolved.",
 "Claim.settlement_days": "Days from creation to resolution. Use for 'settlement time'.",
 "Claim.days_into_policy": "Days between policy effect date and claim creation. 'Claims within 30 days of policy start' -> <= 30.",
 "Claim.fraud_probability": "Fraud score 0-100 (percent). '0.8' from a user means 80.",
 "Claim.fraud_suspicion_indicator": "True if flagged as suspected fraud.",
 "Claim.est_total_repair_cost_with_vat": "Rollup: total damage-assessment cost for motor claims.",
 "Claim.claim_lob": "Line of business of the claim.",
 "Insurer.saudization_pct": "Share of Saudi employees, percent.",
 "Insurer.is_local": "True for Saudi-domiciled insurers; false = foreign.",
 "HealthMember.age_years": "Age in years, precomputed. Use instead of date_of_birth arithmetic.",
 "HealthMember.relationship_with_primary": "self | spouse | child ... (profile).",
 "DistributionChannel.channel_type": "broker | agency | tpa (profile).",
 "Policyholder.policyholder_type": "individual | organisation (profile).",
 "Vehicle.is_ev": "True for electric vehicles.",
 "Vehicle.ai_assisted_driving_function": "True if the vehicle has AI-assisted driving.",
 "Accident.fatality_count": "Number of fatalities in the accident.",
 "DamageAssessment.total_cost_with_vat": "Estimated repair cost incl. VAT.",
}
ALLOWED_TOPICS = sorted(set("""
policy policies premium premiums gwp loss ratio renewal renewed renewals cancelled cancellation expiring expiry active
insurer insurers reinsurer reinsurers reinsured reinsurance quota share agreement broker brokers agency channel channels agent agents
product products ia product authority
claim claims claimant settlement outstanding approved rejected rejection paid fraud fraudulent suspicious
pre-authorization pre-authorisation preauth pre-auth authorization authorisation
health member members beneficiary dependent dependents class benefit diagnosis diagnoses icd patient payer
vehicle vehicles motor manufacturer electric ev plate accident accidents fatality fatalities injury repair damage assessment
policyholder policyholders organisation organization organisations sponsor sponsoring
city cities region local foreign saudization saudisation
summary summarize overview breakdown report trend trends monthly yearly annual quarterly by month by year by quarter
how many show me list top average total count compare journey
""".split())) + ["ia product", "loss ratio", "quota share", "by month", "by year", "by quarter", "show me", "how many", "pre-authorization"]
LABEL_KEYWORDS = {
 "Policy": ["policy","policies","premium","gwp","loss ratio","renewal","cancelled","expiring","active"],
 "Insurer": ["insurer","insurers","carrier","company","saudization","local","foreign"],
 "Reinsurer": ["reinsurer","reinsured","reinsurance","quota share","agreement"],
 "DistributionChannel": ["channel","broker","agency","tpa","distribution"],
 "Agent": ["agent","agents"],
 "Product": ["product","products"], "IAProduct": ["ia product","authority product","regulated"],
 "Policyholder": ["policyholder","owner","organisation","organization","sponsor","employer"],
 "HealthMember": ["member","members","beneficiary","dependent","age","relationship"],
 "BenefitClass": ["class","benefit class","tier"],
 "Vehicle": ["vehicle","vehicles","car","manufacturer","electric","ev","plate","ai assisted"],
 "Claim": ["claim","claims","settlement","outstanding","fraud","approved","rejected","paid"],
 "Claimant": ["claimant","filed by"],
 "PreAuthorization": ["pre-auth","pre-authorization","pre-authorisation","preauth","prior approval"],
 "HealthClaimLine": ["service line","patient share","payer share","line item"],
 "DiagnosisCode": ["diagnosis","icd","condition"],
 "Accident": ["accident","accidents","fatality","injury","crash"],
 "DamageAssessment": ["damage","repair","assessment","repair cost"],
 "PolicyEvent": ["event","events","endorsement","reinstatement"],
}
PATH_KEYWORDS = {
 "claim_to_insurer": ["claims by insurer","settlement","fraud by insurer"],
 "loss_ratio_weighted": ["loss ratio","loss ratio by insurer","loss ratio by line of business"],
 "policy_to_ia_product": ["ia product","policies per product"],
 "motor_owner": ["policyholder","owner","vehicles insured at"],
 "member_to_insurer": ["member age","members by insurer"],
 "member_sponsor": ["sponsor","organisation","employer"],
 "claim_diagnosis": ["diagnosis","icd"],
 "claim_repair_cost": ["repair cost","damage"],
 "pre_auth": ["pre-auth","pre-authorization","prior approval"],
}
COMMON_PATHS = {
 "claim_to_insurer": "MATCH (c:Claim)-[:UNDER_POLICY]->(p:Policy)-[:ISSUED_BY]->(i:Insurer)",
 "loss_ratio_weighted": "MATCH (p:Policy)-[:ISSUED_BY]->(i:Insurer) WITH i.name AS insurer, sum(p.total_paid) AS paid, sum(p.gross_written_premium) AS gwp WHERE gwp > 0 RETURN insurer, round(paid * 100.0 / gwp, 2) AS loss_ratio_pct  // NEVER avg(calculated_loss_ratio_pct) at group level",
 "policy_to_ia_product": "MATCH (p:Policy)-[:HAS_PRODUCT]->(pr:Product)-[:REGULATED_AS]->(ia:IAProduct)",
 "motor_owner": "MATCH (v:Vehicle)-[:INSURED_UNDER]->(p:Policy)-[:ISSUED_BY]->(i:Insurer), (v)-[:OWNED_BY]->(ph:Policyholder)",
 "member_to_insurer": "MATCH (m:HealthMember)-[:COVERED_BY]->(p:Policy)-[:ISSUED_BY]->(i:Insurer)",
 "member_sponsor": "MATCH (m:HealthMember)-[:SPONSORED_BY]->(ph:Policyholder)",
 "claim_diagnosis": "MATCH (c:Claim)-[:HAS_SERVICE_LINE]->(l:HealthClaimLine)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)",
 "claim_repair_cost": "MATCH (c:Claim)-[:INVOLVES_ACCIDENT]->(a:Accident)-[:ASSESSED_BY]->(d:DamageAssessment)",
 "pre_auth": "MATCH (c:Claim)-[:HAS_PRE_AUTH]->(pa:PreAuthorization)   // no Policy edge on PreAuthorization",
 "reinsurance_terms": "MATCH (p:Policy)-[x:REINSURED_BY]->(r:Reinsurer)  // x.agreement_type, x.quota_share_percentage",
 "time_series_monthly": "MATCH (p:Policy) WHERE p.issue_date IS NOT NULL RETURN p.issue_date.year AS yr, p.issue_date.month AS mo, count(p) AS n ORDER BY yr, mo",
}

def write_app_dictionary():
    labels = {}
    for lbl, n in NODES.items():
        props = {}
        for k, t in n["props"].items():
            d = PROP_DESC.get(f"{lbl}.{k}")
            if d is None:
                d = f"{k.replace('_', ' ')} ({t.lower()})" + (". Rollup - precomputed." if k in n.get("rollups", []) else "")
            props[k] = {"description": d}
        entry = {"description": n["desc"] + (f" Rollups: {', '.join(n['rollups'])}." if n.get("rollups") else "")}
        if PII.get(lbl):
            entry["pii"] = PII[lbl]
        entry["properties"] = props
        labels[lbl] = entry
    rels = {}
    for f_, t, to, p, note in RELS:
        a, b = f_[0].lower(), to[0].lower() if to[0].lower() != f_[0].lower() else to[:2].lower()
        r = {"description": (note or f"{f_} {t.lower().replace('_', ' ')} {to}."), "from": f_, "to": to,
             "example": f"MATCH ({a}:{f_})-[{'x' if p else ''}:{t}]->({b}:{to})"}
        if p:
            r["properties"] = {k: {"description": f"{k.replace('_', ' ')} ({v.lower()})"} for k, v in p.items()}
        rels[t] = r
    doc = {
        "token_budget": {"schema_char_budget": 6000, "min_rel_budget": 200, "min_path_budget": 150,
                         "max_values_shown": 4, "max_prop_desc_chars": 70},
        "label_keywords": LABEL_KEYWORDS,
        "path_keywords": PATH_KEYWORDS,
        "labels": labels,
        "relationships": rels,
        "schema_notes": {
            "rollups": "Policy and Claim carry precomputed rollups (claim_count, total_paid, settlement_days, days_into_policy, calculated_loss_ratio_pct, renewal_count, is_active, is_open). ALWAYS use the flat property for policy/claim-level totals; traverse only for per-entity breakdowns.",
            "loss_ratio": "calculated_loss_ratio_pct is PER-POLICY (filter/inspect one policy). Group-level loss ratio (by insurer/LoB/channel) = sum(total_paid) * 100.0 / sum(gross_written_premium) with gwp > 0 - NEVER avg the per-policy pct.",
            "premium": "Premium is Policy.gross_written_premium - there is NO Premium node.",
            "claims_to_insurer": "(Claim)-[:UNDER_POLICY]->(Policy)-[:ISSUED_BY]->(Insurer). ~10% of claims have no UNDER_POLICY edge.",
            "pre_auth": "PreAuthorization hangs off Claim only: (c:Claim)-[:HAS_PRE_AUTH]->(pa). Never from Policy.",
            "policyholder": "One Policyholder label for individuals and organisations (policyholder_type). Motor: (Vehicle)-[:OWNED_BY]->(Policyholder). Health sponsor: (HealthMember)-[:SPONSORED_BY]->(Policyholder).",
            "dates": "DateTime props: dot notation (p.issue_date.year) and compare with datetime(). Date props (HealthMember.date_of_birth, Product.approval_date, Product.end_date) compare with date().",
            "geography": "No City/Region nodes. city/region are strings on Insurer, DistributionChannel, Policyholder, HealthMember, Accident, DamageAssessment.center_city.",
            "fraud_scale": "fraud_probability is 0-100. A user saying 'above 0.8' means > 80.",
            "age": "HealthMember.age_years is precomputed - never compute from date_of_birth.",
        },
        "common_paths": COMMON_PATHS,
        "allowed_topics": ALLOWED_TOPICS,
    }
    with open("data_dictionary.yaml", "w") as f:
        f.write("# IA Knowledge Graph - data dictionary for catalog.py (RP layout)\n"
                "# GENERATED by live_model.py from the live Neo4j schema. Regenerate, do not hand-edit.\n"
                "# value maps: profile enums (SELECT DISTINCT) and add `values:` blocks under the relevant properties.\n")
        yaml.dump(doc, f, sort_keys=False, width=110, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    write_questions(); write_dictionary(); write_modelfile(); write_app_dictionary()
    ids = {n for n in NODES}
    bad = [r for r in RELS if r[0] not in ids or r[2] not in ids]
    print(f"questions={len(Q)} labels={len(NODES)} rels={len(RELS)} bad={bad} fewshots={len(FEWSHOT_IDS)}")