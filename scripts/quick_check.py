#!/usr/bin/env python3
"""
quick_check.py
──────────────
Smoke-test a trained adapter: generate 5 queries and show raw output.
Takes ~60 seconds. Run this BEFORE the full eval.

Usage:
    python scripts/quick_check.py --model ./lora_adapter_v3
"""
import argparse, os, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings

SYSTEM_PROMPT = """You are a Cypher query generator for the RP knowledge graph.
Rules: Output ONLY Cypher. No markdown. Every query needs LIMIT. Alias all aggregations. Read-only.

Nodes: Patient, Visit, Charge, Transaction, Statement, PhoneBridge,
       RCCall, IVRInbound, DiallerCall, Location, InsurancePlan,
       Practice, Campaign, DiagnosisCode, ProcedureCode, BirdeyeReview

Key properties:
  Patient:     outstanding_balance, total_charged, total_paid, adj_bad_debt,
               payor_cohort, is_self_pay, is_friction, is_tennessee, call_tier,
               propensity_grade, gender, source_db, patient_id
  Charge:      charge_amount, balance, dos_aging_bucket, procedure_modality,
               is_voided, is_hold, current_responsible_level
  Transaction: payment_amount, paysource, payment_method, adjustment_bucket,
               denial_code, transaction_type
  RCCall:      agent_name, skill_name, campaign_name, sla, agent_time, total_time
  Statement:   statement_level, patient_balance, is_on_hold, is_released
  Location:    name, city, state, birdeye_avg_rating, birdeye_review_count

Relationships (source → target):
  (Patient)-[:HAD_VISIT]->(Visit)
  (Patient)-[:HAS_CHARGE]->(Charge)
  (Patient)-[:HAS_TRANSACTION]->(Transaction)
  (Patient)-[:RECEIVED_STATEMENT]->(Statement)
  (Patient)-[:IDENTIFIED_BY_PHONE]->(PhoneBridge)
  (Patient)-[:CALLED_IVR]->(IVRInbound)
  (Patient)-[:CONTACTED_BY_DIALLER]->(DiallerCall)
  (Transaction)-[:SETTLES]->(Charge)
  (RCCall)-[:ATTRIBUTED_TO_PHONE]->(PhoneBridge)
  (RCCall)-[:PART_OF_CAMPAIGN]->(Campaign)
  (Charge)-[:AT_LOCATION]->(Location)
  (Charge)-[:DIAGNOSED_WITH]->(DiagnosisCode)
  (Charge)-[:USES_PROCEDURE]->(ProcedureCode)
  (Charge)-[:PART_OF_VISIT]->(Visit)
  (Visit)-[:PERFORMED_AT]->(Location)
  (Visit)-[:UNDER_PLAN]->(InsurancePlan)
  (Location)-[:BELONGS_TO_PRACTICE]->(Practice)
  (BirdeyeReview)-[:REVIEWS]->(Location)

"""

TEST_QUESTIONS = [
    "How many patients have an outstanding balance greater than zero?",
    "What is the total outstanding balance grouped by payor cohort?",
    "How many RingCentral calls are recorded and what is the average handle time?",
    "Which locations have the highest Birdeye average rating?",
    "Show patients who have RingCentral calls attributed through the phone bridge.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from peft import PeftModel

    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.LORA_BASE_MODEL,
        max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base)
    tokenizer = get_chat_template(tokenizer, "gemma")
    model = PeftModel.from_pretrained(base, args.model)
    model.eval()
    print("✓ Loaded\n")

    passed = 0
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}] {q}")
        messages  = [{"role": "user", "content": SYSTEM_PROMPT + q}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)
        attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=200, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        response = tokenizer.decode(
            out[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        print(f"    {response[:300]}")

        # Simple quality checks
        ok = ("MATCH" in response.upper() and
              "RETURN" in response.upper() and
              "LIMIT" in response.upper() and
              "AS AS AS" not in response.upper())
        status = "✓ LOOKS GOOD" if ok else "✗ PROBLEM"
        print(f"    {status}\n")
        if ok:
            passed += 1

    print(f"{'='*50}")
    print(f"  {passed}/{len(TEST_QUESTIONS)} questions produced valid-looking Cypher")
    if passed == len(TEST_QUESTIONS):
        print("  ✓ Proceed to full eval_runner.py")
    elif passed >= 3:
        print("  ✓ Acceptable — run full eval to get precise scores")
    else:
        print("  ✗ Model still hallucinating — check training loss was < 0.5")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()