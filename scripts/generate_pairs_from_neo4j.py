#!/usr/bin/env python3
"""
generate_pairs_from_neo4j.py
────────────────────────────
Grounded Q→Cypher pair generator for RP LoRA training.
All labels, properties, and relationship directions are from
data_catalog.yaml (updated 2026-07-06, 50k patient graph).

Node counts (updated):
  Patient 50,000  Visit 199,948   Charge 474,607
  Transaction 758,534  Statement 142,639  PhoneBridge 65,930
  RCCall 5,000    IVRInbound 400  DiallerCall 150
  Location 300    InsurancePlan 2,000  Campaign 45
  DiagnosisCode 39  ProcedureCode 36  BirdeyeReview 404
  Practice 44

True relationship directions (source → target):
  (Patient)     -[:HAD_VISIT]->          (Visit)
  (Patient)     -[:HAS_CHARGE]->         (Charge)
  (Patient)     -[:HAS_TRANSACTION]->    (Transaction)
  (Patient)     -[:RECEIVED_STATEMENT]-> (Statement)
  (Patient)     -[:IDENTIFIED_BY_PHONE]->(PhoneBridge)
  (Patient)     -[:REGISTERED_AT]->      (Practice)
  (Patient)     -[:CALLED_IVR]->         (IVRInbound)
  (Patient)     -[:CONTACTED_BY_DIALLER]->(DiallerCall)
  (PhoneBridge) -[:BRIDGES_TO_PATIENT]-> (Patient)    [reverse edge]
  (RCCall)      -[:ATTRIBUTED_TO_PHONE]->(PhoneBridge)
  (RCCall)      -[:PART_OF_CAMPAIGN]->   (Campaign)
  (Charge)      -[:PART_OF_VISIT]->      (Visit)
  (Charge)      -[:AT_LOCATION]->        (Location)
  (Charge)      -[:DIAGNOSED_WITH]->     (DiagnosisCode)
  (Charge)      -[:USES_PROCEDURE]->     (ProcedureCode)
  (Transaction) -[:SETTLES]->            (Charge)      ← NOT Charge→Transaction
  (Visit)       -[:PERFORMED_AT]->       (Location)
  (Visit)       -[:UNDER_PLAN]->         (InsurancePlan)
  (Location)    -[:BELONGS_TO_PRACTICE]->(Practice)
  (InsurancePlan)-[:ISSUED_BY_PRACTICE]->(Practice)
  (BirdeyeReview)-[:REVIEWS]->           (Location)
  (Campaign)    -[:RUN_BY]->             (Practice)

Usage:
    python scripts/generate_pairs_from_neo4j.py \\
        --uri bolt://localhost:7687 \\
        --user neo4j --password <pw> \\
        --sample-patients 5000 \\
        --num-pairs 5000 \\
        --output data/raw/generated_pairs.parquet

    # Dry run — connect, sample, print stats only:
    python scripts/generate_pairs_from_neo4j.py \\
        --uri bolt://localhost:7687 --user neo4j --password <pw> \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class GraphSampler:
    """
    Pulls representative value samples from Neo4j for use in
    grounded question templates.

    Thin tables (count < 500) → load all.
    Fat tables (Charge, Transaction, Visit) → random window sample.
    """

    def __init__(self, driver, n_patients: int = 5000, seed: int = 42):
        self.driver     = driver
        self.n_patients = n_patients
        self.seed       = seed
        random.seed(seed)
        self.ctx: dict[str, Any] = {}

    def _run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as s:
            return s.run(cypher, **params).data()

    def _sample(self, cypher: str, n: int) -> list[dict]:
        rows = self._run(cypher)
        return random.sample(rows, min(n, len(rows))) if rows else []

    def load(self) -> dict[str, Any]:
        print("\nSampling graph values from Neo4j...")

        # ── Patient ───────────────────────────────────────────────────────────
        print("  Patient...")
        pats = self._sample(
            "MATCH (p:Patient) RETURN "
            "p.patient_id AS patient_id, p.first_name AS first_name, "
            "p.last_name AS last_name, p.source_db AS source_db, "
            "p.outstanding_balance AS outstanding_balance, "
            "p.payor_cohort AS payor_cohort, p.is_self_pay AS is_self_pay, "
            "p.call_tier AS call_tier, p.propensity_grade AS propensity_grade, "
            "p.propensity_desc AS propensity_desc, "
            "p.is_friction AS is_friction, p.adj_bad_debt AS adj_bad_debt, "
            "p.total_charged AS total_charged, p.total_paid AS total_paid, "
            "p.gender AS gender, p.state AS state, p.zip AS zip, "
            "p.is_tennessee AS is_tennessee, "
            "p.multi_practice_flag AS multi_practice_flag, "
            "p.has_insurance AS has_insurance, "
            "p.is_sapa AS is_sapa, p.is_nraa AS is_nraa, "
            "p.is_bai AS is_bai, p.is_catastrophe AS is_catastrophe, "
            "p.statement_count AS statement_count, "
            "p.charge_count AS charge_count, "
            "p.visit_count AS visit_count "
            "LIMIT 200000",
            self.n_patients
        )
        self.ctx["patients"]      = pats
        self.ctx["source_dbs"]    = list({p["source_db"] for p in pats if p.get("source_db")})
        self.ctx["payor_cohorts"] = list({p["payor_cohort"] for p in pats if p.get("payor_cohort")})
        self.ctx["call_tiers"]    = sorted({p["call_tier"] for p in pats if p.get("call_tier")})
        self.ctx["prop_grades"]   = sorted({p["propensity_grade"] for p in pats if p.get("propensity_grade")})
        self.ctx["prop_descs"]    = list({p["propensity_desc"] for p in pats if p.get("propensity_desc")})
        self.ctx["genders"]       = list({p["gender"] for p in pats if p.get("gender")})
        self.ctx["states_pat"]    = list({p["state"] for p in pats if p.get("state")})
        balances = sorted([p["outstanding_balance"] for p in pats
                           if p.get("outstanding_balance") and p["outstanding_balance"] > 0])
        self.ctx["balance_values"] = balances

        # ── Charge ────────────────────────────────────────────────────────────
        print("  Charge...")
        charges = self._sample(
            "MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge) RETURN "
            "c.charge_id AS charge_id, c.charge_amount AS charge_amount, "
            "c.balance AS balance, c.line_status AS line_status, "
            "c.dos_aging_bucket AS dos_aging_bucket, c.is_voided AS is_voided, "
            "c.procedure_modality AS procedure_modality, "
            "c.current_responsible_level AS resp_level, "
            "c.place_of_service AS pos, "
            "c.modifier AS modifier, "
            "c.source_db AS source_db LIMIT 500000",
            5000
        )
        self.ctx["charges"]        = charges
        self.ctx["aging_buckets"]  = list({c["dos_aging_bucket"] for c in charges if c.get("dos_aging_bucket")})
        self.ctx["line_statuses"]  = list({c["line_status"] for c in charges if c.get("line_status")})
        self.ctx["resp_levels"]    = list({c["resp_level"] for c in charges if c.get("resp_level")})
        self.ctx["modalities"]     = list({c["procedure_modality"] for c in charges if c.get("procedure_modality")})
        self.ctx["pos_codes"]      = list({c["pos"] for c in charges if c.get("pos")})
        charge_amounts = sorted([c["charge_amount"] for c in charges
                                 if c.get("charge_amount") and c["charge_amount"] > 0])
        self.ctx["charge_amounts"] = charge_amounts

        # ── Transaction ───────────────────────────────────────────────────────
        print("  Transaction...")
        txns = self._sample(
            "MATCH (p:Patient)-[:HAS_TRANSACTION]->(t:Transaction) RETURN "
            "t.transaction_type AS txn_type, t.paysource AS paysource, "
            "t.payment_method AS payment_method, "
            "t.adjustment_bucket AS adj_bucket, "
            "t.denial_code AS denial_code, "
            "t.payment_amount AS payment_amount, "
            "t.adjustment_amount AS adj_amount, "
            "t.adjustment_type AS adj_type, "
            "t.processing_type AS proc_type, "
            "p.source_db AS source_db LIMIT 500000",
            5000
        )
        self.ctx["transactions"]  = txns
        self.ctx["txn_types"]     = list({t["txn_type"] for t in txns if t.get("txn_type")})
        self.ctx["paysources"]    = list({t["paysource"] for t in txns if t.get("paysource")})
        self.ctx["pay_methods"]   = list({t["payment_method"] for t in txns if t.get("payment_method")})
        self.ctx["adj_buckets"]   = list({t["adj_bucket"] for t in txns if t.get("adj_bucket")})
        self.ctx["adj_types"]     = list({t["adj_type"] for t in txns if t.get("adj_type")})
        self.ctx["denial_codes"]  = list({t["denial_code"] for t in txns if t.get("denial_code")})

        # ── Visit ─────────────────────────────────────────────────────────────
        print("  Visit...")
        visits = self._sample(
            "MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit) RETURN "
            "v.visit_id AS visit_id, v.source_db AS source_db, "
            "v.primary_insurance_plan AS primary_insurance_plan, "
            "v.secondary_insurance_plan AS secondary_insurance_plan, "
            "v.location_id AS location_id, "
            "p.patient_id AS patient_id LIMIT 300000",
            5000
        )
        self.ctx["visits"] = visits

        # ── Statement ─────────────────────────────────────────────────────────
        print("  Statement...")
        stmts = self._sample(
            "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement) RETURN "
            "s.statement_id AS stmt_id, s.statement_level AS level, "
            "s.patient_balance AS patient_balance, "
            "s.total_balance AS total_balance, "
            "s.is_on_hold AS is_on_hold, s.is_released AS is_released, "
            "s.text_successful AS text_successful, "
            "s.email_successful AS email_successful, "
            "s.source_db AS source_db, "
            "p.patient_id AS patient_id LIMIT 300000",
            5000
        )
        self.ctx["statements"]  = stmts
        self.ctx["stmt_levels"] = sorted({s["level"] for s in stmts if s.get("level")})

        # ── Location (all 300) ────────────────────────────────────────────────
        print("  Location...")
        locs = self._run(
            "MATCH (l:Location) RETURN "
            "l.location_id AS location_id, l.name AS name, "
            "l.city AS city, l.state AS state, "
            "l.birdeye_avg_rating AS avg_rating, "
            "l.birdeye_review_count AS review_count, "
            "l.birdeye_one_star_pct AS one_star_pct, "
            "l.source_db AS source_db, l.location_type AS loc_type "
            "LIMIT 10000"
        )
        self.ctx["locations"]  = locs
        self.ctx["loc_states"] = list({l["state"] for l in locs if l.get("state")})
        self.ctx["loc_cities"] = list({l["city"] for l in locs if l.get("city")})
        self.ctx["loc_names"]  = [l["name"] for l in locs if l.get("name")]
        self.ctx["loc_ids"]    = [l["location_id"] for l in locs if l.get("location_id")]
        self.ctx["loc_types"]  = list({l["loc_type"] for l in locs if l.get("loc_type")})

        # ── InsurancePlan (2000 — sample 200) ─────────────────────────────────
        print("  InsurancePlan...")
        plans = self._sample(
            "MATCH (ip:InsurancePlan) RETURN "
            "ip.plan_name AS plan_name, ip.carrier_name AS carrier_name, "
            "ip.plan_type AS plan_type, ip.plan_number AS plan_number, "
            "ip.source_db AS source_db LIMIT 10000",
            200
        )
        self.ctx["insurance_plans"] = plans
        self.ctx["carriers"]        = list({p["carrier_name"] for p in plans if p.get("carrier_name")})
        self.ctx["plan_types"]      = list({p["plan_type"] for p in plans if p.get("plan_type")})
        self.ctx["plan_names"]      = list({p["plan_name"] for p in plans if p.get("plan_name")})

        # ── RCCall (5000 — sample 2000) ───────────────────────────────────────
        print("  RCCall...")
        rccalls = self._sample(
            "MATCH (rc:RCCall) RETURN "
            "rc.rccallId AS rccall_id, "
            "rc.agent_name AS agent_name, rc.skill_name AS skill_name, "
            "rc.campaign_name AS campaign_name, rc.team_name AS team_name, "
            "rc.sla AS sla, rc.agent_time AS agent_time, "
            "rc.total_time AS total_time, rc.disp_name AS disp_name, "
            "rc.rc_attributable AS rc_attributable, "
            "rc.pre_queue AS pre_queue, rc.in_queue AS in_queue, "
            "rc.tags AS tags LIMIT 10000",
            2000
        )
        self.ctx["rccalls"]       = rccalls
        self.ctx["skill_names"]   = list({r["skill_name"] for r in rccalls if r.get("skill_name")})
        self.ctx["team_names"]    = list({r["team_name"] for r in rccalls if r.get("team_name")})
        self.ctx["agent_names"]   = list({r["agent_name"] for r in rccalls if r.get("agent_name")})
        self.ctx["disp_names"]    = list({r["disp_name"] for r in rccalls if r.get("disp_name")})
        self.ctx["camp_names_rc"] = list({r["campaign_name"] for r in rccalls if r.get("campaign_name")})

        # ── Campaign (45 — all) ───────────────────────────────────────────────
        print("  Campaign...")
        cams = self._run(
            "MATCH (cam:Campaign) RETURN "
            "cam.name AS name, cam.source_db AS source_db "
            "LIMIT 100"
        )
        self.ctx["campaigns"]      = cams
        self.ctx["campaign_names"] = [c["name"] for c in cams if c.get("name")]
        self.ctx["camp_src_dbs"]   = list({c["source_db"] for c in cams if c.get("source_db")})

        # ── PhoneBridge (65k — sample 3000) ───────────────────────────────────
        print("  PhoneBridge...")
        pbs = self._sample(
            "MATCH (pb:PhoneBridge) RETURN "
            "pb.phone_type AS phone_type, "
            "pb.primary_campaign AS primary_campaign, "
            "pb.rc_call_count AS rc_call_count, "
            "pb.campaign_count AS campaign_count, "
            "pb.source_db AS source_db LIMIT 100000",
            3000
        )
        self.ctx["phone_bridges"]   = pbs
        self.ctx["phone_types"]     = list({p["phone_type"] for p in pbs if p.get("phone_type")})
        self.ctx["camp_names_pb"]   = list({p["primary_campaign"] for p in pbs if p.get("primary_campaign")})

        # ── IVRInbound (400 — all) ────────────────────────────────────────────
        print("  IVRInbound...")
        ivrs = self._sample(
            "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound) RETURN "
            "ivr.ivr_type AS ivr_type, ivr.amount_paid AS amount_paid, "
            "ivr.balance AS balance, ivr.result_desc AS result_desc, "
            "ivr.auth_success AS auth_success, "
            "ivr.facility_code AS facility_code, "
            "p.patient_id AS patient_id LIMIT 10000",
            400
        )
        self.ctx["ivr_calls"]       = ivrs
        self.ctx["ivr_types"]       = list({i["ivr_type"] for i in ivrs if i.get("ivr_type")})
        self.ctx["ivr_results"]     = list({i["result_desc"] for i in ivrs if i.get("result_desc")})
        self.ctx["ivr_facilities"]  = list({i["facility_code"] for i in ivrs if i.get("facility_code")})

        # ── DiallerCall (150 — all) ────────────────────────────────────────────
        print("  DiallerCall...")
        dcs = self._sample(
            "MATCH (p:Patient)-[:CONTACTED_BY_DIALLER]->(dc:DiallerCall) RETURN "
            "dc.result_desc AS result_desc, dc.service_loc AS service_loc, "
            "dc.patient_balance AS patient_balance, "
            "p.patient_id AS patient_id LIMIT 10000",
            150
        )
        self.ctx["dialler_calls"] = dcs
        self.ctx["dial_results"]  = list({d["result_desc"] for d in dcs if d.get("result_desc")})
        self.ctx["dial_locs"]     = list({d["service_loc"] for d in dcs if d.get("service_loc")})

        # ── DiagnosisCode (39 — all) ───────────────────────────────────────────
        print("  DiagnosisCode...")
        diags = self._run("MATCH (d:DiagnosisCode) RETURN d.code AS code LIMIT 100")
        self.ctx["diag_codes"]    = [d["code"] for d in diags if d.get("code")]
        self.ctx["diag_prefixes"] = list({c[0] for c in self.ctx["diag_codes"] if c})

        # ── ProcedureCode (36 — all) ───────────────────────────────────────────
        print("  ProcedureCode...")
        procs = self._run(
            "MATCH (pc:ProcedureCode) RETURN "
            "pc.code AS code, pc.description AS description, "
            "pc.modality AS modality LIMIT 100"
        )
        self.ctx["proc_codes"]      = [p["code"] for p in procs if p.get("code")]
        self.ctx["proc_modalities"] = list({p["modality"] for p in procs if p.get("modality")})
        self.ctx["proc_descs"]      = [p["description"] for p in procs if p.get("description")]

        # ── Practice (44 — all) ────────────────────────────────────────────────
        print("  Practice...")
        practices = self._run("MATCH (pr:Practice) RETURN pr.code AS code LIMIT 100")
        self.ctx["practice_codes"] = [p["code"] for p in practices if p.get("code")]

        # ── BirdeyeReview (404 — all) ──────────────────────────────────────────
        print("  BirdeyeReview...")
        brs = self._sample(
            "MATCH (br:BirdeyeReview) RETURN "
            "br.rating AS rating, br.source AS source, "
            "br.phi_flagged AS phi_flagged LIMIT 10000",
            404
        )
        self.ctx["br_sources"] = list({b["source"] for b in brs if b.get("source")})

        # ── Multi-hop: PhoneBridge attribution ────────────────────────────────
        print("  Phone bridge paths...")
        pb_paths = self._sample(
            "MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)"
            "<-[:ATTRIBUTED_TO_PHONE]-(rc:RCCall) RETURN "
            "p.patient_id AS patient_id, "
            "p.outstanding_balance AS balance, "
            "pb.primary_campaign AS campaign, "
            "rc.skill_name AS skill, "
            "rc.campaign_name AS rc_campaign LIMIT 50000",
            2000
        )
        self.ctx["pb_paths"] = pb_paths

        # ── Multi-hop: IVR payments ────────────────────────────────────────────
        print("  IVR payment paths...")
        ivr_pay = self._sample(
            "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound) "
            "WHERE ivr.amount_paid > 0 RETURN "
            "p.patient_id AS patient_id, "
            "ivr.amount_paid AS amount_paid, "
            "ivr.balance AS balance LIMIT 10000",
            400
        )
        self.ctx["ivr_payments"] = ivr_pay

        # ── Multi-hop: SETTLES paths ───────────────────────────────────────────
        print("  SETTLES paths...")
        settles = self._sample(
            "MATCH (t:Transaction)-[:SETTLES]->(c:Charge) RETURN "
            "t.transaction_type AS txn_type, t.paysource AS paysource, "
            "t.denial_code AS denial_code, "
            "c.balance AS charge_balance, "
            "c.charge_amount AS charge_amount LIMIT 200000",
            5000
        )
        self.ctx["settles_paths"]        = settles
        self.ctx["denial_codes_settles"] = list({s["denial_code"] for s in settles if s.get("denial_code")})

        self._print_summary()
        return self.ctx

    def _print_summary(self):
        print(f"\n  Sampling complete:")
        for key, label in [
            ("patients","patients"), ("charges","charges"),
            ("transactions","transactions"), ("visits","visits"),
            ("statements","statements"), ("rccalls","rc_calls"),
            ("ivr_calls","ivr_calls"), ("phone_bridges","phone_bridges"),
        ]:
            print(f"    {label:<20} {len(self.ctx.get(key,[])):>6} rows")

    def pick(self, key: str, default: Any = "UNKNOWN") -> Any:
        lst = self.ctx.get(key, [])
        return random.choice(lst) if lst else default

    def pick_balance_threshold(self, pct: float = 0.5) -> float:
        vals = self.ctx.get("balance_values", [500.0])
        if not vals:
            return 500.0
        return round(vals[max(0, int(len(vals) * pct) - 1)], 0)

    def pick_charge_threshold(self, pct: float = 0.5) -> float:
        vals = self.ctx.get("charge_amounts", [500.0])
        if not vals:
            return 500.0
        return round(vals[max(0, int(len(vals) * pct) - 1)], 0)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _add(rows, question, cypher, category, difficulty, **grounding):
    rows.append({
        "question":         question.strip(),
        "cypher":           cypher.strip(),
        "category":         category,
        "difficulty":       difficulty,
        "grounding_values": json.dumps(grounding),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def gen_financial(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.10:
            _add(rows,
                "What is the total outstanding balance and patient count across all patients?",
                "MATCH (p:Patient)\nWHERE p.outstanding_balance > 0\n"
                "RETURN count(p) AS patient_count,\n"
                "       sum(p.outstanding_balance) AS total_outstanding\nLIMIT 1",
                "financial_ar", "easy")

        elif r < 0.20:
            bal = ctx.pick_balance_threshold(0.6)
            _add(rows,
                f"Which patients have an outstanding balance over ${int(bal):,}?",
                f"MATCH (p:Patient)\nWHERE p.outstanding_balance > {bal}\n"
                "RETURN p.patient_id, p.first_name, p.last_name,\n"
                "       p.outstanding_balance, p.payor_cohort, p.source_db\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "financial_ar", "easy", balance_threshold=bal)

        elif r < 0.30:
            cohort = ctx.pick("payor_cohorts", "self_pay")
            _add(rows,
                f"What is the total charged, total paid, and outstanding balance for the '{cohort}' payor cohort?",
                f"MATCH (p:Patient)\nWHERE p.payor_cohort = '{cohort}'\n"
                "RETURN count(p) AS patients,\n"
                "       sum(p.total_charged) AS total_charged,\n"
                "       sum(p.total_paid) AS total_paid,\n"
                "       sum(p.outstanding_balance) AS outstanding\nLIMIT 1",
                "financial_ar", "medium", cohort=cohort)

        elif r < 0.40:
            bucket = ctx.pick("aging_buckets", "90+")
            _add(rows,
                f"Show all non-voided charges in the '{bucket}' aging bucket.",
                f"MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)\n"
                f"WHERE c.dos_aging_bucket = '{bucket}' AND c.is_voided = false\n"
                "RETURN p.patient_id, c.charge_id, c.charge_amount,\n"
                "       c.balance, c.line_status\n"
                "ORDER BY c.balance DESC\nLIMIT 50",
                "financial_ar", "medium", bucket=bucket)

        elif r < 0.49:
            paysrc = ctx.pick("paysources", "ers")
            _add(rows,
                f"What is the total collected via '{paysrc}' payments?",
                f"MATCH (p:Patient)-[:HAS_TRANSACTION]->(t:Transaction)\n"
                f"WHERE t.paysource = '{paysrc}' AND t.payment_amount > 0\n"
                "RETURN count(t) AS txn_count,\n"
                "       sum(t.payment_amount) AS total_collected\nLIMIT 1",
                "financial_ar", "easy", paysource=paysrc)

        elif r < 0.57:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"What is the total bad debt written off at practice '{src}'?",
                f"MATCH (p:Patient)\nWHERE p.source_db = '{src}' AND p.adj_bad_debt > 0\n"
                "RETURN count(p) AS patients_with_bad_debt,\n"
                "       sum(p.adj_bad_debt) AS total_bad_debt\nLIMIT 1",
                "financial_ar", "medium", source_db=src)

        elif r < 0.65:
            dcode = ctx.pick("denial_codes_settles", "CO-97")
            _add(rows,
                f"Which charges have been denied with code '{dcode}'?",
                f"MATCH (t:Transaction)-[:SETTLES]->(c:Charge)\n"
                f"MATCH (p:Patient)-[:HAS_CHARGE]->(c)\n"
                f"WHERE t.denial_code = '{dcode}'\n"
                "RETURN p.patient_id, c.charge_id, t.denial_note,\n"
                "       c.charge_amount, c.balance\n"
                "ORDER BY c.balance DESC\nLIMIT 30",
                "financial_ar", "hard", denial_code=dcode)

        elif r < 0.73:
            level = ctx.pick("resp_levels", "patient")
            _add(rows,
                f"Show charges where the current responsible level is '{level}'.",
                f"MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)\n"
                f"WHERE c.current_responsible_level = '{level}'\n"
                "  AND c.balance > 0 AND c.is_voided = false\n"
                "RETURN p.patient_id, c.charge_id, c.balance,\n"
                "       c.charge_amount, c.line_status\n"
                "ORDER BY c.balance DESC\nLIMIT 50",
                "financial_ar", "medium", resp_level=level)

        elif r < 0.81:
            adj = ctx.pick("adj_buckets", "bad_debt")
            _add(rows,
                f"What is the total adjustment amount for the '{adj}' adjustment bucket?",
                f"MATCH (p:Patient)-[:HAS_TRANSACTION]->(t:Transaction)\n"
                f"WHERE t.adjustment_bucket = '{adj}'\n"
                "RETURN count(t) AS adjustments,\n"
                "       sum(t.adjustment_amount) AS total_adjusted\nLIMIT 1",
                "financial_ar", "medium", adj_bucket=adj)

        elif r < 0.89:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"Compare total billed vs total collected for practice '{src}'.",
                f"MATCH (p:Patient)\nWHERE p.source_db = '{src}'\n"
                "RETURN count(p) AS patients,\n"
                "       sum(p.total_charged) AS total_billed,\n"
                "       sum(p.total_paid) AS total_paid,\n"
                "       sum(p.outstanding_balance) AS still_outstanding\nLIMIT 1",
                "financial_ar", "medium", source_db=src)

        else:
            chg_bal = ctx.pick_charge_threshold(0.75)
            _add(rows,
                f"Show charges over ${int(chg_bal):,} with their procedure and diagnosis codes.",
                f"MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)\n"
                f"-[:DIAGNOSED_WITH]->(d:DiagnosisCode)\n"
                f"WHERE c.charge_amount > {chg_bal}\n"
                "MATCH (c)-[:USES_PROCEDURE]->(pc:ProcedureCode)\n"
                "RETURN p.patient_id, c.charge_id, c.charge_amount,\n"
                "       d.code AS icd10, pc.code AS cpt\n"
                "ORDER BY c.charge_amount DESC\nLIMIT 20",
                "financial_ar", "hard", charge_threshold=chg_bal)
    return rows


def gen_contact_center(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.10:
            _add(rows,
                "How many RingCentral calls are recorded and what is the average handle time?",
                "MATCH (rc:RCCall)\n"
                "RETURN count(rc) AS total_calls,\n"
                "       avg(rc.agent_time) AS avg_agent_seconds,\n"
                "       avg(rc.total_time) AS avg_total_seconds\nLIMIT 1",
                "contact_center", "easy")

        elif r < 0.20:
            skill = ctx.pick("skill_names", "PMR English")
            _add(rows,
                f"How many RingCentral calls were handled under the '{skill}' skill?",
                f"MATCH (rc:RCCall)\nWHERE rc.skill_name = '{skill}'\n"
                "RETURN count(rc) AS call_count,\n"
                "       avg(rc.agent_time) AS avg_handle_seconds,\n"
                "       sum(CASE WHEN rc.sla = 1 THEN 1 ELSE 0 END) AS sla_met\nLIMIT 1",
                "contact_center", "medium", skill=skill)

        elif r < 0.30:
            agent = ctx.pick("agent_names", "Smith")
            _add(rows,
                f"Show call performance metrics for agent '{agent}'.",
                f"MATCH (rc:RCCall)\nWHERE rc.agent_name = '{agent}'\n"
                "RETURN count(rc) AS calls,\n"
                "       avg(rc.agent_time) AS avg_handle_seconds,\n"
                "       avg(rc.acw_time) AS avg_acw_seconds,\n"
                "       sum(CASE WHEN rc.sla = 1 THEN 1 ELSE 0 END) AS sla_met\nLIMIT 1",
                "contact_center", "medium", agent=agent)

        elif r < 0.40:
            _add(rows,
                "Which patients were contacted via IVR and made a payment?",
                "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound)\n"
                "WHERE ivr.amount_paid > 0\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       ivr.response_id, ivr.amount_paid, ivr.balance\n"
                "ORDER BY ivr.amount_paid DESC\nLIMIT 50",
                "contact_center", "medium")

        elif r < 0.49:
            _add(rows,
                "What is the total amount collected via IVR payments?",
                "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound)\n"
                "WHERE ivr.amount_paid > 0\n"
                "RETURN count(ivr) AS paying_sessions,\n"
                "       sum(ivr.amount_paid) AS total_ivr_collected\nLIMIT 1",
                "contact_center", "easy")

        elif r < 0.58:
            result = ctx.pick("dial_results", "No Answer")
            _add(rows,
                f"How many dialler calls resulted in '{result}'?",
                f"MATCH (p:Patient)-[:CONTACTED_BY_DIALLER]->(dc:DiallerCall)\n"
                f"WHERE dc.result_desc = '{result}'\n"
                "RETURN count(dc) AS call_count,\n"
                "       avg(dc.patient_balance) AS avg_patient_balance\nLIMIT 1",
                "contact_center", "easy", result=result)

        elif r < 0.67:
            camp = ctx.pick("campaign_names", "SAPA")
            _add(rows,
                f"Show RingCentral call stats for the '{camp}' campaign.",
                f"MATCH (rc:RCCall)-[:PART_OF_CAMPAIGN]->(cam:Campaign)\n"
                f"WHERE cam.name = '{camp}'\n"
                "RETURN count(rc) AS total_calls,\n"
                "       avg(rc.total_time) AS avg_total_seconds,\n"
                "       avg(rc.agent_time) AS avg_handle_seconds,\n"
                "       sum(CASE WHEN rc.sla = 1 THEN 1 ELSE 0 END) AS sla_met\nLIMIT 1",
                "contact_center", "medium", campaign=camp)

        elif r < 0.76:
            team = ctx.pick("team_names", "Team A")
            _add(rows,
                f"What is the SLA performance for team '{team}'?",
                f"MATCH (rc:RCCall)\nWHERE rc.team_name = '{team}'\n"
                "RETURN count(rc) AS total_calls,\n"
                "       sum(CASE WHEN rc.sla = 1 THEN 1 ELSE 0 END) AS sla_met,\n"
                "       avg(rc.in_queue) AS avg_queue_seconds\nLIMIT 1",
                "contact_center", "medium", team=team)

        elif r < 0.85:
            disp = ctx.pick("disp_names", "Completed")
            _add(rows,
                f"How many RingCentral calls had a disposition of '{disp}'?",
                f"MATCH (rc:RCCall)\nWHERE rc.disp_name = '{disp}'\n"
                "RETURN count(rc) AS call_count,\n"
                "       avg(rc.agent_time) AS avg_handle_seconds\nLIMIT 1",
                "contact_center", "easy", disposition=disp)

        elif r < 0.92:
            ivr_type = ctx.pick("ivr_types", "inbound")
            _add(rows,
                f"Show IVR call volume and payment rate for IVR type '{ivr_type}'.",
                f"MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound)\n"
                f"WHERE ivr.ivr_type = '{ivr_type}'\n"
                "RETURN count(ivr) AS sessions,\n"
                "       sum(CASE WHEN ivr.amount_paid > 0 THEN 1 ELSE 0 END) AS paid_count,\n"
                "       sum(ivr.amount_paid) AS total_collected\nLIMIT 1",
                "contact_center", "medium", ivr_type=ivr_type)

        else:
            _add(rows,
                "Which patients were contacted by the dialler but have not made an IVR payment?",
                "MATCH (p:Patient)-[:CONTACTED_BY_DIALLER]->(:DiallerCall)\n"
                "WHERE NOT (p)-[:CALLED_IVR]->(:IVRInbound)\n"
                "  AND p.outstanding_balance > 0\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.outstanding_balance, p.call_tier\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "contact_center", "hard")
    return rows


def gen_phone_bridge(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.18:
            _add(rows,
                "How many RingCentral calls can be attributed to a known patient via the phone bridge?",
                "MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                "<-[:ATTRIBUTED_TO_PHONE]-(rc:RCCall)\n"
                "WHERE rc.rc_attributable = true\n"
                "RETURN count(rc) AS attributable_calls,\n"
                "       count(DISTINCT p.patient_id) AS unique_patients\nLIMIT 1",
                "phone_bridge_attribution", "medium")

        elif r < 0.35:
            camp = ctx.pick("camp_names_pb", "SAPA")
            _add(rows,
                f"Which patients were reached via the phone bridge for the '{camp}' campaign?",
                f"MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                f"<-[:ATTRIBUTED_TO_PHONE]-(rc:RCCall)\n"
                f"WHERE pb.primary_campaign = '{camp}'\n"
                "RETURN p.patient_id, p.source_db, p.outstanding_balance,\n"
                "       count(rc) AS rc_calls, pb.phone_type\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "phone_bridge_attribution", "hard", campaign=camp)

        elif r < 0.50:
            ptype = ctx.pick("phone_types", "cell")
            _add(rows,
                f"Show phone bridge records with phone type '{ptype}' and their RC call count.",
                f"MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                f"WHERE pb.phone_type = '{ptype}'\n"
                "RETURN p.patient_id, pb.rc_call_count,\n"
                "       pb.campaign_count, pb.primary_campaign\n"
                "ORDER BY pb.rc_call_count DESC\nLIMIT 50",
                "phone_bridge_attribution", "medium", phone_type=ptype)

        elif r < 0.63:
            _add(rows,
                "Which patients have a phone bridge record but zero attributed RingCentral calls?",
                "MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                "WHERE pb.rc_call_count = 0 OR pb.rc_call_count IS NULL\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.outstanding_balance, pb.phone_type\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "phone_bridge_attribution", "medium")

        elif r < 0.76:
            bal = ctx.pick_balance_threshold(0.7)
            _add(rows,
                f"Show patients with a balance over ${int(bal):,} who have RingCentral calls attributed through the phone bridge.",
                f"MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                f"<-[:ATTRIBUTED_TO_PHONE]-(rc:RCCall)\n"
                f"WHERE p.outstanding_balance > {bal}\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.outstanding_balance, count(rc) AS rc_calls,\n"
                "       pb.primary_campaign\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "phone_bridge_attribution", "hard", balance_threshold=bal)

        elif r < 0.87:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"How many phone bridge records belong to practice '{src}'?",
                f"MATCH (pb:PhoneBridge)\nWHERE pb.source_db = '{src}'\n"
                "RETURN count(pb) AS bridge_count,\n"
                "       sum(pb.rc_call_count) AS total_rc_calls,\n"
                "       avg(pb.rc_call_count) AS avg_calls\nLIMIT 1",
                "phone_bridge_attribution", "easy", source_db=src)

        else:
            _add(rows,
                "What is the total and average RC call count across all phone bridge records?",
                "MATCH (pb:PhoneBridge)\n"
                "RETURN count(pb) AS bridge_records,\n"
                "       sum(pb.rc_call_count) AS total_rc_calls,\n"
                "       avg(pb.rc_call_count) AS avg_calls_per_bridge\nLIMIT 1",
                "phone_bridge_attribution", "easy")
    return rows


def gen_visit(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.14:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"How many visits are recorded at practice '{src}'?",
                f"MATCH (v:Visit)\nWHERE v.source_db = '{src}'\n"
                "RETURN count(v) AS visit_count\nLIMIT 1",
                "visit_clinical", "easy", source_db=src)

        elif r < 0.27:
            state = ctx.pick("loc_states", "TN")
            _add(rows,
                f"Show visits performed at locations in {state}.",
                f"MATCH (v:Visit)-[:PERFORMED_AT]->(l:Location)\n"
                f"WHERE l.state = '{state}'\n"
                "RETURN v.visit_id, v.source_db,\n"
                "       l.name AS location_name, l.city\n"
                "ORDER BY v.admit_date DESC\nLIMIT 50",
                "visit_clinical", "medium", state=state)

        elif r < 0.40:
            _add(rows,
                "Which patients have the most visits?",
                "MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)\n"
                "RETURN p.patient_id, p.source_db, count(v) AS visit_count\n"
                "ORDER BY visit_count DESC\nLIMIT 20",
                "visit_clinical", "easy")

        elif r < 0.52:
            _add(rows,
                "Show patients who had a visit and have charges linked to that visit.",
                "MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)\n"
                "MATCH (c:Charge)-[:PART_OF_VISIT]->(v)\n"
                "RETURN p.patient_id, v.visit_id,\n"
                "       count(c) AS charge_count,\n"
                "       sum(c.charge_amount) AS total_charged\n"
                "ORDER BY total_charged DESC\nLIMIT 25",
                "visit_clinical", "hard")

        elif r < 0.63:
            modality = ctx.pick("modalities", "MRI")
            _add(rows,
                f"How many charges are for '{modality}' procedure modality?",
                f"MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)\n"
                f"WHERE c.procedure_modality = '{modality}'\n"
                "RETURN count(c) AS charge_count,\n"
                "       sum(c.charge_amount) AS total_billed,\n"
                "       avg(c.charge_amount) AS avg_charge\nLIMIT 1",
                "visit_clinical", "easy", modality=modality)

        elif r < 0.74:
            carrier = ctx.pick("carriers", "MEDICARE")
            _add(rows,
                f"Show visits covered under '{carrier}' insurance.",
                f"MATCH (v:Visit)-[:UNDER_PLAN]->(ip:InsurancePlan)\n"
                f"WHERE ip.carrier_name = '{carrier}'\n"
                "RETURN v.visit_id, v.source_db,\n"
                "       ip.plan_name, ip.plan_type\n"
                "ORDER BY v.admit_date DESC\nLIMIT 50",
                "visit_clinical", "medium", carrier=carrier)

        elif r < 0.84:
            _add(rows,
                "Show visits that have both primary and secondary insurance plans recorded.",
                "MATCH (v:Visit)\n"
                "WHERE v.primary_insurance_plan IS NOT NULL\n"
                "  AND v.secondary_insurance_plan IS NOT NULL\n"
                "RETURN v.visit_id, v.source_db,\n"
                "       v.primary_insurance_plan, v.secondary_insurance_plan\n"
                "ORDER BY v.admit_date DESC\nLIMIT 50",
                "visit_clinical", "medium")

        else:
            loc = ctx.pick("loc_names", "Main Campus")
            _add(rows,
                f"How many charges were posted at location '{loc}'?",
                f"MATCH (c:Charge)-[:AT_LOCATION]->(l:Location)\n"
                f"WHERE l.name = '{loc}'\n"
                "RETURN count(c) AS charge_count,\n"
                "       sum(c.charge_amount) AS total_billed\nLIMIT 1",
                "visit_clinical", "medium", location=loc)
    return rows


def gen_patient_demographics(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.13:
            cohort = ctx.pick("payor_cohorts", "self_pay")
            _add(rows,
                f"How many patients are in the '{cohort}' payor cohort?",
                f"MATCH (p:Patient)\nWHERE p.payor_cohort = '{cohort}'\n"
                "RETURN count(p) AS patient_count,\n"
                "       sum(p.outstanding_balance) AS total_balance\nLIMIT 1",
                "patient_demographics", "easy", cohort=cohort)

        elif r < 0.25:
            tier = ctx.pick("call_tiers", "A")
            _add(rows,
                f"Show patients in call tier '{tier}' with their outstanding balance.",
                f"MATCH (p:Patient)\nWHERE p.call_tier = '{tier}'\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.outstanding_balance, p.payor_cohort\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "patient_demographics", "easy", tier=tier)

        elif r < 0.37:
            grade = ctx.pick("prop_grades", "B")
            _add(rows,
                f"Show the average outstanding balance for propensity grade '{grade}' patients.",
                f"MATCH (p:Patient)\nWHERE p.propensity_grade = '{grade}'\n"
                "RETURN count(p) AS patients,\n"
                "       avg(p.outstanding_balance) AS avg_balance,\n"
                "       sum(p.outstanding_balance) AS total_balance\nLIMIT 1",
                "patient_demographics", "medium", grade=grade)

        elif r < 0.48:
            _add(rows,
                "Which patients are marked as friction and have a balance over $300?",
                "MATCH (p:Patient)\n"
                "WHERE p.is_friction = true AND p.outstanding_balance > 300\n"
                "RETURN p.patient_id, p.source_db, p.call_tier,\n"
                "       p.outstanding_balance, p.payor_cohort\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 50",
                "patient_demographics", "medium")

        elif r < 0.58:
            _add(rows,
                "How many patients are in Tennessee versus other states?",
                "MATCH (p:Patient)\n"
                "RETURN p.is_tennessee AS in_tennessee,\n"
                "       count(p) AS patient_count,\n"
                "       sum(p.outstanding_balance) AS total_balance\n"
                "ORDER BY in_tennessee DESC\nLIMIT 2",
                "patient_demographics", "easy")

        elif r < 0.67:
            gender = ctx.pick("genders", "F")
            _add(rows,
                f"What is the total outstanding balance for {gender} patients?",
                f"MATCH (p:Patient)\nWHERE p.gender = '{gender}'\n"
                "RETURN count(p) AS patient_count,\n"
                "       sum(p.outstanding_balance) AS total_balance,\n"
                "       avg(p.outstanding_balance) AS avg_balance\nLIMIT 1",
                "patient_demographics", "easy", gender=gender)

        elif r < 0.76:
            _add(rows,
                "Show patients registered at multiple practices.",
                "MATCH (p:Patient)\nWHERE p.multi_practice_flag = true\n"
                "RETURN p.patient_id, p.practice_count,\n"
                "       p.outstanding_balance, p.source_db\n"
                "ORDER BY p.practice_count DESC\nLIMIT 25",
                "patient_demographics", "medium")

        elif r < 0.85:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"How many self-pay patients are at practice '{src}'?",
                f"MATCH (p:Patient)\n"
                f"WHERE p.source_db = '{src}' AND p.is_self_pay = true\n"
                "RETURN count(p) AS self_pay_patients,\n"
                "       sum(p.outstanding_balance) AS total_balance\nLIMIT 1",
                "patient_demographics", "easy", source_db=src)

        else:
            _add(rows,
                "What is the gender distribution of patients by payor cohort?",
                "MATCH (p:Patient)\n"
                "WHERE p.gender IS NOT NULL AND p.payor_cohort IS NOT NULL\n"
                "RETURN p.payor_cohort AS cohort, p.gender AS gender,\n"
                "       count(p) AS patient_count\n"
                "ORDER BY cohort, patient_count DESC\nLIMIT 20",
                "patient_demographics", "hard")
    return rows


def gen_statements(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.17:
            _add(rows,
                "How many statements have been sent and to how many unique patients?",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "RETURN count(s) AS total_statements,\n"
                "       count(DISTINCT p.patient_id) AS unique_patients\nLIMIT 1",
                "statement_collections", "easy")

        elif r < 0.33:
            lvl = ctx.pick("stmt_levels", "3")
            _add(rows,
                f"Which patients are at statement level {lvl}?",
                f"MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                f"WHERE s.statement_level = '{lvl}'\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       s.patient_balance, s.is_on_hold, s.is_released\n"
                "ORDER BY s.patient_balance DESC\nLIMIT 50",
                "statement_collections", "medium", level=lvl)

        elif r < 0.49:
            _add(rows,
                "Which statements are currently on hold?",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "WHERE s.is_on_hold = true\n"
                "RETURN p.patient_id, s.statement_id,\n"
                "       s.total_balance, s.statement_level, s.created_date\n"
                "ORDER BY s.total_balance DESC\nLIMIT 50",
                "statement_collections", "easy")

        elif r < 0.63:
            _add(rows,
                "Show patients who have received more than one statement.",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "WITH p, count(s) AS stmt_count\n"
                "WHERE stmt_count > 1\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       stmt_count, p.outstanding_balance\n"
                "ORDER BY stmt_count DESC\nLIMIT 25",
                "statement_collections", "medium")

        elif r < 0.76:
            _add(rows,
                "What is the total patient balance grouped by statement level?",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "RETURN s.statement_level AS level,\n"
                "       count(s) AS stmt_count,\n"
                "       sum(s.patient_balance) AS total_patient_balance\n"
                "ORDER BY level\nLIMIT 10",
                "statement_collections", "medium")

        elif r < 0.87:
            _add(rows,
                "Which statements were successfully delivered via text or email?",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "WHERE s.text_successful = 'Y' OR s.email_successful = 'Y'\n"
                "RETURN p.patient_id, s.statement_id,\n"
                "       s.text_successful, s.email_successful,\n"
                "       s.patient_balance\n"
                "ORDER BY s.patient_balance DESC\nLIMIT 50",
                "statement_collections", "medium")

        else:
            _add(rows,
                "Show released statements and the patient balance they carried at release.",
                "MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)\n"
                "WHERE s.is_released = true\n"
                "RETURN p.patient_id, s.statement_id,\n"
                "       s.released_date, s.patient_balance, s.statement_level\n"
                "ORDER BY s.released_date DESC\nLIMIT 50",
                "statement_collections", "easy")
    return rows


def gen_location(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.20:
            state = ctx.pick("loc_states", "TN")
            _add(rows,
                f"Which locations are in {state} and what is their Birdeye rating?",
                f"MATCH (l:Location)\nWHERE l.state = '{state}'\n"
                "RETURN l.location_id, l.name, l.city,\n"
                "       l.birdeye_avg_rating, l.birdeye_review_count\n"
                "ORDER BY l.birdeye_avg_rating DESC\nLIMIT 20",
                "location_facility", "easy", state=state)

        elif r < 0.38:
            _add(rows,
                "Which locations have the most charges performed there?",
                "MATCH (c:Charge)-[:AT_LOCATION]->(l:Location)\n"
                "RETURN l.location_id, l.name, l.city,\n"
                "       count(c) AS charge_count,\n"
                "       sum(c.charge_amount) AS total_billed\n"
                "ORDER BY total_billed DESC\nLIMIT 20",
                "location_facility", "medium")

        elif r < 0.54:
            _add(rows,
                "Show locations with PHI-flagged Birdeye reviews.",
                "MATCH (br:BirdeyeReview)-[:REVIEWS]->(l:Location)\n"
                "WHERE br.phi_flagged = true\n"
                "RETURN l.location_id, l.name,\n"
                "       count(br) AS phi_review_count\n"
                "ORDER BY phi_review_count DESC\nLIMIT 10",
                "location_facility", "medium")

        elif r < 0.70:
            _add(rows,
                "Show locations with more than 10 Birdeye reviews sorted by one-star percentage.",
                "MATCH (l:Location)\n"
                "WHERE l.birdeye_review_count > 10\n"
                "RETURN l.name, l.city, l.state,\n"
                "       l.birdeye_one_star_pct, l.birdeye_avg_rating,\n"
                "       l.birdeye_review_count\n"
                "ORDER BY l.birdeye_one_star_pct DESC\nLIMIT 15",
                "location_facility", "medium")

        elif r < 0.84:
            _add(rows,
                "How many locations belong to each practice?",
                "MATCH (l:Location)-[:BELONGS_TO_PRACTICE]->(pr:Practice)\n"
                "RETURN pr.code AS practice, count(l) AS location_count\n"
                "ORDER BY location_count DESC\nLIMIT 15",
                "location_facility", "easy")

        else:
            _add(rows,
                "Which locations have the most visits performed there?",
                "MATCH (v:Visit)-[:PERFORMED_AT]->(l:Location)\n"
                "RETURN l.location_id, l.name, l.city, l.state,\n"
                "       count(v) AS visit_count\n"
                "ORDER BY visit_count DESC\nLIMIT 15",
                "location_facility", "medium")
    return rows


def gen_insurance(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.25:
            carrier = ctx.pick("carriers", "MEDICARE")
            _add(rows,
                f"How many visits are under a '{carrier}' insurance plan?",
                f"MATCH (v:Visit)-[:UNDER_PLAN]->(ip:InsurancePlan)\n"
                f"WHERE ip.carrier_name = '{carrier}'\n"
                "RETURN count(v) AS visit_count,\n"
                "       count(DISTINCT ip.plan_name) AS plan_count\nLIMIT 1",
                "insurance_payer", "medium", carrier=carrier)

        elif r < 0.48:
            ptype = ctx.pick("plan_types", "HMO")
            _add(rows,
                f"Show all insurance plans of type '{ptype}'.",
                f"MATCH (ip:InsurancePlan)\nWHERE ip.plan_type = '{ptype}'\n"
                "RETURN ip.plan_name, ip.carrier_name,\n"
                "       ip.plan_number, ip.source_db\n"
                "ORDER BY ip.carrier_name\nLIMIT 20",
                "insurance_payer", "easy", plan_type=ptype)

        elif r < 0.68:
            _add(rows,
                "Which insurance plan is linked to the most visits?",
                "MATCH (v:Visit)-[:UNDER_PLAN]->(ip:InsurancePlan)\n"
                "RETURN ip.plan_name, ip.carrier_name,\n"
                "       count(v) AS visit_count\n"
                "ORDER BY visit_count DESC\nLIMIT 10",
                "insurance_payer", "medium")

        elif r < 0.84:
            src = ctx.pick("source_dbs", "RADM")
            _add(rows,
                f"How many insurance plans belong to practice '{src}'?",
                f"MATCH (ip:InsurancePlan)\nWHERE ip.source_db = '{src}'\n"
                "RETURN count(ip) AS plan_count,\n"
                "       count(DISTINCT ip.carrier_name) AS carrier_count\nLIMIT 1",
                "insurance_payer", "easy", source_db=src)

        else:
            _add(rows,
                "Show insurance plans issued by each practice.",
                "MATCH (ip:InsurancePlan)-[:ISSUED_BY_PRACTICE]->(pr:Practice)\n"
                "RETURN pr.code AS practice,\n"
                "       ip.plan_name, ip.carrier_name, ip.plan_type\n"
                "ORDER BY pr.code\nLIMIT 20",
                "insurance_payer", "hard")
    return rows


def gen_diagnosis_procedure(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.22:
            _add(rows,
                "What are the most common diagnosis codes across all charges?",
                "MATCH (c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)\n"
                "RETURN d.code AS icd10, count(c) AS charge_count\n"
                "ORDER BY charge_count DESC\nLIMIT 15",
                "diagnosis_procedure", "easy")

        elif r < 0.42:
            prefix = ctx.pick("diag_prefixes", "Z")
            _add(rows,
                f"How many charges have a diagnosis code starting with '{prefix}'?",
                f"MATCH (c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)\n"
                f"WHERE d.code STARTS WITH '{prefix}'\n"
                "RETURN count(c) AS charge_count,\n"
                "       sum(c.charge_amount) AS total_billed\nLIMIT 1",
                "diagnosis_procedure", "medium", prefix=prefix)

        elif r < 0.60:
            modality = ctx.pick("proc_modalities", "MRI")
            _add(rows,
                f"What is the average charge amount for '{modality}' procedure codes?",
                f"MATCH (c:Charge)-[:USES_PROCEDURE]->(pc:ProcedureCode)\n"
                f"WHERE pc.modality = '{modality}'\n"
                "RETURN count(c) AS uses,\n"
                "       avg(c.charge_amount) AS avg_charge,\n"
                "       sum(c.charge_amount) AS total_billed\nLIMIT 1",
                "diagnosis_procedure", "medium", modality=modality)

        elif r < 0.78:
            _add(rows,
                "Show the top CPT codes by total charge amount.",
                "MATCH (c:Charge)-[:USES_PROCEDURE]->(pc:ProcedureCode)\n"
                "RETURN pc.code AS cpt, pc.description,\n"
                "       count(c) AS uses,\n"
                "       sum(c.charge_amount) AS total_billed\n"
                "ORDER BY total_billed DESC\nLIMIT 10",
                "diagnosis_procedure", "medium")

        else:
            _add(rows,
                "Show charges with more than one diagnosis code recorded.",
                "MATCH (c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)\n"
                "WITH c, count(d) AS diag_count\n"
                "WHERE diag_count > 1\n"
                "RETURN c.charge_id, c.source_db,\n"
                "       diag_count, c.charge_amount\n"
                "ORDER BY diag_count DESC\nLIMIT 20",
                "diagnosis_procedure", "hard")
    return rows


def gen_cross_category(ctx: GraphSampler, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        r = random.random()
        if r < 0.25:
            bal = ctx.pick_balance_threshold(0.6)
            _add(rows,
                f"Show patients with balance over ${int(bal):,} who have been contacted by dialler and have a phone bridge record.",
                f"MATCH (p:Patient)-[:CONTACTED_BY_DIALLER]->(:DiallerCall)\n"
                f"MATCH (p)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
                f"WHERE p.outstanding_balance > {bal}\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.outstanding_balance, pb.primary_campaign\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 25",
                "cross_category", "hard", balance_threshold=bal)

        elif r < 0.50:
            state = ctx.pick("loc_states", "TN")
            _add(rows,
                f"Show patients who had a visit at a location in {state} and still have an outstanding balance.",
                f"MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)-[:PERFORMED_AT]->(l:Location)\n"
                f"WHERE l.state = '{state}' AND p.outstanding_balance > 0\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       l.name AS location, p.outstanding_balance\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 25",
                "cross_category", "hard", state=state)

        elif r < 0.72:
            _add(rows,
                "Show the financial summary for patients who also made an IVR payment.",
                "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound)\n"
                "WHERE ivr.amount_paid > 0\n"
                "WITH p, sum(ivr.amount_paid) AS ivr_total\n"
                "RETURN p.patient_id, p.source_db,\n"
                "       p.total_charged, p.total_paid,\n"
                "       p.outstanding_balance, ivr_total\n"
                "ORDER BY p.outstanding_balance DESC\nLIMIT 20",
                "cross_category", "hard")

        else:
            _add(rows,
                "What is the total billed per practice for charges at locations belonging to that practice?",
                "MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)-[:AT_LOCATION]->(l:Location)\n"
                "MATCH (l)-[:BELONGS_TO_PRACTICE]->(pr:Practice)\n"
                "RETURN pr.code AS practice,\n"
                "       count(DISTINCT p.patient_id) AS unique_patients,\n"
                "       count(c) AS charges,\n"
                "       sum(c.charge_amount) AS total_billed\n"
                "ORDER BY total_billed DESC\nLIMIT 10",
                "cross_category", "hard")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_BUDGET = {
    "financial_ar":             0.22,
    "contact_center":           0.14,
    "phone_bridge_attribution": 0.12,
    "visit_clinical":           0.12,
    "patient_demographics":     0.10,
    "statement_collections":    0.09,
    "location_facility":        0.07,
    "insurance_payer":          0.06,
    "diagnosis_procedure":      0.05,
    "cross_category":           0.03,
}

GENERATORS = {
    "financial_ar":             gen_financial,
    "contact_center":           gen_contact_center,
    "phone_bridge_attribution": gen_phone_bridge,
    "visit_clinical":           gen_visit,
    "patient_demographics":     gen_patient_demographics,
    "statement_collections":    gen_statements,
    "location_facility":        gen_location,
    "insurance_payer":          gen_insurance,
    "diagnosis_procedure":      gen_diagnosis_procedure,
    "cross_category":           gen_cross_category,
}


def generate_all(ctx: GraphSampler, total: int) -> list[dict]:
    all_rows = []
    print(f"\nGenerating {total:,} question/Cypher pairs...")
    for cat, frac in CATEGORY_BUDGET.items():
        n = max(1, round(total * frac))
        rows = GENERATORS[cat](ctx, n)
        all_rows.extend(rows)
        print(f"  {cat:<35} {len(rows):>5} pairs")
    random.shuffle(all_rows)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded Q→Cypher pairs from Neo4j (50k patient graph)"
    )
    parser.add_argument("--uri",             default="bolt://localhost:7687")
    parser.add_argument("--user",            default="neo4j")
    parser.add_argument("--password",        default=None)
    parser.add_argument("--sample-patients", type=int, default=5000,
                        help="Patients to sample for value grounding (default 5000 = 10%%)")
    parser.add_argument("--num-pairs",       type=int, default=5000,
                        help="Total Q/Cypher pairs to generate (default 5000)")
    parser.add_argument("--output",          default="data/raw/generated_pairs.parquet")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--dry-run",         action="store_true",
                        help="Sample values, print stats, do not generate pairs")
    args = parser.parse_args()

    pw = args.password or os.getenv("NEO4J_PASSWORD", "")
    if not pw:
        print("NEO4J_PASSWORD not set. Pass --password or set the env var.")
        sys.exit(1)

    print("=" * 62)
    print("  RP Text2Cypher — Neo4j Grounded Pair Generator")
    print(f"  Neo4j:          {args.uri}")
    print(f"  Patient sample: {args.sample_patients:,} of 50,000")
    print(f"  Target pairs:   {args.num_pairs:,}")
    print(f"  Timestamp:      {datetime.now().isoformat()}")
    print("=" * 62)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))
    try:
        sampler = GraphSampler(driver, n_patients=args.sample_patients, seed=args.seed)
        sampler.load()

        if args.dry_run:
            print("\nDry run complete — no pairs written.")
            for k in ["source_dbs","payor_cohorts","call_tiers","prop_grades",
                       "aging_buckets","skill_names","campaign_names","carriers",
                       "plan_types","diag_prefixes","proc_modalities","loc_states"]:
                print(f"  {k}: {sampler.ctx.get(k)}")
            return

        pairs = generate_all(sampler, args.num_pairs)

    finally:
        driver.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pairs)
    df.to_parquet(out, index=False)

    print(f"\nSaved {len(df):,} pairs → {out}")
    print(f"\nDifficulty:")
    for diff in ["easy", "medium", "hard"]:
        n = len(df[df["difficulty"] == diff])
        print(f"  {diff:<8} {n:>5}  ({n/len(df)*100:.1f}%)")
    print(f"\nNext step:")
    print(f"  python scripts/validate_pairs.py \\")
    print(f"    --input  {out} \\")
    print(f"    --uri    {args.uri} --user {args.user} \\")
    print(f"    --build-splits")


if __name__ == "__main__":
    main()