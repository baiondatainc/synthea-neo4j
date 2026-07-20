#!/usr/bin/env python3
"""
demo_inference.py  — unified RP Text2Cypher demo
─────────────────────────────────────────────────
Supports three modes:
  --mode ollama   : use Ollama model (text2cypher, rp-cypher, etc.)
  --mode lora     : use LoRA adapter directly (lora_adapter_v3)
  --mode router   : keyword router (no model, always works)

Usage:
    # Ollama (recommended for demo):
    python demo_inference.py --mode ollama --model text2cypher \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>

    # LoRA adapter:
    python demo_inference.py --mode lora --model ./lora_adapter_v3 \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>

    # Single question:
    python demo_inference.py --mode ollama --model text2cypher \\
        --question "How many patients?" \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>

    # Run preset demo questions:
    python demo_inference.py --mode ollama --model text2cypher --demo \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import warnings, logging, os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.ERROR)


# ── System prompt — identical to train_lora.py ────────────────────────────────
SYSTEM_PROMPT = """You are a Neo4j Cypher generator for the RP (RP) knowledge graph.
Output ONLY raw Cypher. No markdown. No explanations. ALWAYS include LIMIT. Alias every property.

Labels: Patient, Visit, Charge, Transaction, Statement, RCCall, IVRInbound,
        DiallerCall, PhoneBridge, Campaign, Location, InsurancePlan,
        Practice, DiagnosisCode, ProcedureCode, BirdeyeReview

EXACT properties:
Patient:      patient_id, source_db, gender, state, payor_cohort, call_tier,
              propensity_grade, is_self_pay, is_friction, is_tennessee,
              outstanding_balance, total_charged, total_paid, adj_bad_debt
Charge:       charge_id, charge_amount, balance, line_status, dos_aging_bucket,
              procedure_modality, procedure_code, is_voided, is_hold,
              current_responsible_level, service_date, post_date
Transaction:  payment_id, transaction_type, paysource, payment_method,
              payment_amount, adjustment_amount, adjustment_bucket, denial_code
RCCall:       rccallId, agent_name, team_name, skill_name, campaign_name,
              sla, agent_time, total_time, in_queue, disp_name, rc_attributable
IVRInbound:   response_id, ivr_type, amount_paid, balance, result_desc, call_datetime
DiallerCall:  account, result_desc, patient_balance, service_loc, call_datetime
Statement:    statement_id, statement_level, patient_balance, total_balance,
              is_on_hold, is_released, text_successful, email_successful
Location:     location_id, name, city, state, birdeye_avg_rating, birdeye_review_count
PhoneBridge:  phone_type, primary_campaign, rc_call_count, campaign_count
InsurancePlan: plan_name, carrier_name, plan_type, plan_number
Visit:        visit_id, source_db, admit_date, primary_insurance_plan
Campaign:     name  DiagnosisCode: code  ProcedureCode: code, description, modality
BirdeyeReview: rating, phi_flagged, source  Practice: code

Relationships:
(Patient)-[:HAD_VISIT]->(Visit)  (Patient)-[:HAS_CHARGE]->(Charge)
(Patient)-[:HAS_TRANSACTION]->(Transaction)  (Patient)-[:RECEIVED_STATEMENT]->(Statement)
(Patient)-[:IDENTIFIED_BY_PHONE]->(PhoneBridge)  (Patient)-[:CALLED_IVR]->(IVRInbound)
(Patient)-[:CONTACTED_BY_DIALLER]->(DiallerCall)  (Patient)-[:REGISTERED_AT]->(Practice)
(Transaction)-[:SETTLES]->(Charge)  (RCCall)-[:ATTRIBUTED_TO_PHONE]->(PhoneBridge)
(RCCall)-[:PART_OF_CAMPAIGN]->(Campaign)  (Charge)-[:AT_LOCATION]->(Location)
(Charge)-[:DIAGNOSED_WITH]->(DiagnosisCode)  (Charge)-[:USES_PROCEDURE]->(ProcedureCode)
(Charge)-[:PART_OF_VISIT]->(Visit)  (Visit)-[:PERFORMED_AT]->(Location)
(Visit)-[:UNDER_PLAN]->(InsurancePlan)  (Location)-[:BELONGS_TO_PRACTICE]->(Practice)
(BirdeyeReview)-[:REVIEWS]->(Location)

"""

# ── Safety ────────────────────────────────────────────────────────────────────
WRITE_RE    = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.I)
LIMIT_RE    = re.compile(r"\bLIMIT\s+\d+", re.I)
REPET_RE    = re.compile(r"(\bAS\b[\s,]*){6,}", re.I)
PII_KEYS    = {"first_name","last_name","middle_name","dob","email",
               "phone","phone_norm","cell_norm","patient_id"}

def safety_check(cypher: str) -> tuple[bool, Optional[str]]:
    if not cypher or len(cypher.strip()) < 5: return False, "empty"
    if WRITE_RE.search(cypher):               return False, "write keyword blocked"
    if REPET_RE.search(cypher):               return False, "repetition loop detected"
    if not any(k in cypher.upper() for k in ("MATCH","CALL","WITH","UNWIND")):
        return False, "no Cypher clause"
    return True, None

def inject_limit(cypher: str, cap: int = 100) -> str:
    return cypher if LIMIT_RE.search(cypher) else cypher.rstrip(";") + f"\nLIMIT {cap}"

def redact(rows: list) -> list:
    return [{k: "[redacted]" if k.lower() in PII_KEYS else v
             for k, v in row.items()} for row in rows]

def clean_cypher(raw: str) -> str:
    raw = re.sub(r"```(?:cypher|sql)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"```", "", raw).strip()
    for kw in ("MATCH","CALL","WITH","UNWIND","OPTIONAL"):
        if kw in raw.upper():
            return raw[raw.upper().index(kw):].strip()
    return raw


# ── Ollama generator ──────────────────────────────────────────────────────────
def generate_ollama(question: str, model: str,
                    host: str = "http://localhost:11434") -> Optional[str]:
    payload = json.dumps({
        "model":  model,
        "stream": False,
        "options": {"temperature": 0, "top_k": 1, "repeat_penalty": 1.05,
                    "num_predict": 300},
        "messages": [{"role": "user", "content": question}],
    }).encode()
    try:
        req = urllib.request.Request(
            f"{host}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            return json.loads(resp.read()).get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


# ── LoRA adapter generator ────────────────────────────────────────────────────
_model_cache = {}

def load_lora(adapter_path: str):
    if adapter_path in _model_cache:
        return _model_cache[adapter_path]

    import torch
    import settings
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from peft import PeftModel

    print(f"  Loading {"Qwen/Qwen2.5-Coder-7B-Instruct"} + {adapter_path} ...")
    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length=4096, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base)
    tokenizer = get_chat_template(tokenizer, "qwen-2.5")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print("  ✓ Model loaded")
    _model_cache[adapter_path] = (model, tokenizer)
    return model, tokenizer


def generate_lora(question: str, adapter_path: str) -> Optional[str]:
    import torch
    model, tokenizer = load_lora(adapter_path)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT.strip()},
        {"role": "user",      "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
    ).strip()
    return clean_cypher(response) if response else None


# ── Keyword router (fallback, no model needed) ────────────────────────────────
ROUTES = [
    (["outstanding balance","balance greater","balance over","how many patients have"],
     "MATCH (p:Patient) WHERE p.outstanding_balance > 0\n"
     "RETURN count(p) AS patients, sum(p.outstanding_balance) AS total_outstanding LIMIT 1"),
    (["self-pay","self pay","uninsured"],
     "MATCH (p:Patient) WHERE p.is_self_pay = true\n"
     "RETURN count(p) AS self_pay_patients, sum(p.outstanding_balance) AS total_balance LIMIT 1"),
    (["payor cohort","payer cohort","cohort"],
     "MATCH (p:Patient) WHERE p.payor_cohort IS NOT NULL\n"
     "RETURN p.payor_cohort AS cohort, count(p) AS patients,\n"
     "       sum(p.outstanding_balance) AS total_balance\n"
     "ORDER BY total_balance DESC LIMIT 10"),
    (["how many patients","total patients","patient count"],
     "MATCH (p:Patient) RETURN count(p) AS total_patients LIMIT 1"),
    (["ringcentral","rc call","how many calls","total calls"],
     "MATCH (rc:RCCall)\n"
     "RETURN count(rc) AS total_calls, avg(rc.agent_time) AS avg_handle_seconds LIMIT 1"),
    (["sla","service level"],
     "MATCH (rc:RCCall) RETURN rc.sla AS sla_met, count(rc) AS calls\n"
     "ORDER BY sla_met DESC LIMIT 2"),
    (["ivr payment","ivr collected","ivr paid"],
     "MATCH (p:Patient)-[:CALLED_IVR]->(ivr:IVRInbound)\n"
     "WHERE ivr.amount_paid > 0\n"
     "RETURN count(ivr) AS sessions, sum(ivr.amount_paid) AS total_collected LIMIT 1"),
    (["agent","top agent","agent performance"],
     "MATCH (rc:RCCall) WHERE rc.agent_name IS NOT NULL\n"
     "RETURN rc.agent_name AS agent, count(rc) AS calls,\n"
     "       avg(rc.agent_time) AS avg_handle_seconds\n"
     "ORDER BY calls DESC LIMIT 10"),
    (["birdeye","rating","best location","highest rating"],
     "MATCH (l:Location) WHERE l.birdeye_avg_rating IS NOT NULL\n"
     "RETURN l.name, l.city, l.state, l.birdeye_avg_rating AS avg_rating\n"
     "ORDER BY avg_rating DESC LIMIT 10"),
    (["diagnosis","icd","diagnosis code"],
     "MATCH (c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)\n"
     "RETURN d.code AS icd10, count(c) AS charge_count\n"
     "ORDER BY charge_count DESC LIMIT 10"),
    (["procedure","cpt","procedure code"],
     "MATCH (c:Charge)-[:USES_PROCEDURE]->(pc:ProcedureCode)\n"
     "RETURN pc.code AS cpt, pc.description, count(c) AS uses\n"
     "ORDER BY uses DESC LIMIT 10"),
    (["propensity","propensity grade","likelihood"],
     "MATCH (p:Patient) WHERE p.propensity_grade IS NOT NULL\n"
     "RETURN p.propensity_grade AS grade, count(p) AS patients,\n"
     "       avg(p.outstanding_balance) AS avg_balance\n"
     "ORDER BY grade LIMIT 10"),
    (["statement","statement level"],
     "MATCH (s:Statement)\n"
     "RETURN s.statement_level AS level, count(s) AS count\n"
     "ORDER BY level LIMIT 10"),
    (["tennessee","tn patient"],
     "MATCH (p:Patient) WHERE p.is_tennessee = true\n"
     "RETURN count(p) AS tn_patients, sum(p.outstanding_balance) AS total_balance LIMIT 1"),
    (["aging bucket","90+","overdue"],
     "MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)\n"
     "WHERE c.dos_aging_bucket = '90+' AND c.is_voided = false\n"
     "RETURN count(c) AS charges, sum(c.balance) AS total_balance LIMIT 1"),
    (["phone bridge","attribution","attributed"],
     "MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)\n"
     "RETURN count(pb) AS bridge_records, sum(pb.rc_call_count) AS total_calls LIMIT 1"),
    (["bad debt","written off"],
     "MATCH (p:Patient) WHERE p.adj_bad_debt > 0\n"
     "RETURN count(p) AS patients, sum(p.adj_bad_debt) AS total_bad_debt LIMIT 1"),
    (["friction","hard to reach"],
     "MATCH (p:Patient) WHERE p.is_friction = true\n"
     "RETURN count(p) AS friction_patients,\n"
     "       sum(p.outstanding_balance) AS total_balance LIMIT 1"),
    (["campaign"],
     "MATCH (rc:RCCall)-[:PART_OF_CAMPAIGN]->(cam:Campaign)\n"
     "RETURN cam.name AS campaign, count(rc) AS calls\n"
     "ORDER BY calls DESC LIMIT 10"),
    (["insurance plan","carrier","plan"],
     "MATCH (ip:InsurancePlan)\n"
     "RETURN count(ip) AS total_plans, count(DISTINCT ip.carrier_name) AS carriers LIMIT 1"),
    (["practice","by practice","per practice"],
     "MATCH (p:Patient)\n"
     "RETURN p.source_db AS practice, count(p) AS patients,\n"
     "       sum(p.outstanding_balance) AS total_balance\n"
     "ORDER BY total_balance DESC LIMIT 15"),
]

def route(question: str) -> Optional[str]:
    q = question.lower()
    best, score = None, 0
    for keywords, cypher in ROUTES:
        s = sum(1 for kw in keywords if kw in q)
        if s > score:
            score, best = s, cypher
    return best if score > 0 else None


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(question: str, driver, mode: str,
                 model_arg: str, ollama_host: str,
                 verbose: bool = True) -> dict:
    result = {"question": question, "cypher": None,
              "rows": [], "status": None, "error": None}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Q: {question}")

    # Generate
    if mode == "ollama":
        raw = generate_ollama(question, model_arg, ollama_host)
        cypher = clean_cypher(raw) if raw else None
    elif mode == "lora":
        cypher = generate_lora(question, model_arg)
    else:  # router
        cypher = route(question)

    if not cypher:
        result["status"] = "no_output"
        result["error"]  = "No Cypher generated"
        if verbose: print("  ✗ No output")
        return result

    result["cypher"] = cypher
    if verbose:
        print(f"\n  Cypher:")
        for line in cypher.split("\n"):
            print(f"    {line}")

    ok, err = safety_check(cypher)
    if not ok:
        result["status"] = "blocked"
        result["error"]  = err
        if verbose: print(f"  ✗ {err}")
        return result

    # EXPLAIN
    try:
        with driver.session() as s:
            s.run(f"EXPLAIN {cypher}")
        if verbose: print("  ✓ EXPLAIN passed")
    except Exception as e:
        result["status"] = "explain_failed"
        result["error"]  = str(e)[:150]
        if verbose: print(f"  ✗ EXPLAIN: {str(e)[:80]}")
        return result

    # Execute
    try:
        with driver.session() as s:
            rows = s.run(inject_limit(cypher)).data()
        rows = redact(rows)
        result["rows"]   = rows
        result["status"] = "success"
        if verbose:
            print(f"  ✓ {len(rows)} rows")
            for i, row in enumerate(rows[:8], 1):
                print(f"    {i}. {row}")
            if len(rows) > 8:
                print(f"    ... {len(rows)-8} more")
    except Exception as e:
        result["status"] = "exec_failed"
        result["error"]  = str(e)[:150]
        if verbose: print(f"  ✗ Execute: {str(e)[:80]}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "How many patients have an outstanding balance?",
    "Total outstanding balance by payor cohort",
    "How many patients are self-pay?",
    "Top agents by RingCentral call volume",
    "IVR payment collection total",
    "Which locations have the highest Birdeye rating?",
    "Most common diagnosis codes",
    "Propensity grade breakdown with average balance",
    "How many patients in Tennessee?",
    "Statement level distribution",
    "SLA breakdown for RingCentral calls",
    "Top charges by amount",
]


def main():
    parser = argparse.ArgumentParser(description="RP Text2Cypher Demo")
    parser.add_argument("--mode",    default="ollama",
                        choices=["ollama","lora","router"],
                        help="ollama=Ollama API, lora=adapter, router=keyword (default: ollama)")
    parser.add_argument("--model",   default="text2cypher",
                        help="Ollama model name OR path to LoRA adapter")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--uri",     default="bolt://localhost:7687")
    parser.add_argument("--user",    default="neo4j")
    parser.add_argument("--password",default=None)
    parser.add_argument("--question",default=None)
    parser.add_argument("--demo",    action="store_true")
    args = parser.parse_args()

    pw = args.password or os.getenv("NEO4J_PASSWORD","")
    if not pw:
        print("NEO4J_PASSWORD not set.")
        sys.exit(1)

    print("=" * 60)
    print(f"  RP Knowledge Graph — Text2Cypher")
    print(f"  Mode:  {args.mode}")
    print(f"  Model: {args.model}")
    print("=" * 60)

    # Load LoRA model eagerly if needed
    if args.mode == "lora":
        print("\nLoading LoRA model (~30s)...")
        load_lora(args.model)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))
    try:
        with driver.session() as s:
            s.run("RETURN 1")
        print("  ✓ Neo4j connected")
    except Exception as e:
        print(f"  ✗ Neo4j: {e}")
        sys.exit(1)

    if args.question:
        run_pipeline(args.question, driver, args.mode, args.model, args.ollama_host)
        driver.close()
        return

    if args.demo:
        passed = 0
        for q in DEMO_QUESTIONS:
            r = run_pipeline(q, driver, args.mode, args.model, args.ollama_host)
            if r["status"] == "success": passed += 1
        print(f"\n{'='*60}")
        print(f"  {passed}/{len(DEMO_QUESTIONS)} succeeded")
        print("=" * 60)
        driver.close()
        return

    print(f"\nInteractive mode (quit to exit)\n")
    while True:
        try:
            q = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit","exit","q",""): break
        run_pipeline(q, driver, args.mode, args.model, args.ollama_host)

    driver.close()

if __name__ == "__main__":
    main()