#!/usr/bin/env python3
"""
Quick inference script for testing the Text2Cypher LoRA model.

Generates Cypher from a natural language question using the trained adapter,
runs it through guardrails, optionally executes on Neo4j, and redacts results.

Example:
    python scripts/infer.py \\
        --model ./lora_adapter_v1 \\
        --question "Which patients have a balance over $500?" \\
        --execute \\
        --uri bolt://localhost:7687 --user neo4j --password <pw>
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

sys.path.insert(0, str(Path(__file__).parent.parent))

import settings
from runtime.guardrails import check_cypher
from runtime.redaction import redact_rows

SYSTEM_PROMPT = (
    "You are an expert Cypher query generator for the RP knowledge graph.\n\n"
    "Rules:\n"
    "- Output ONLY valid Cypher. No markdown, no explanation.\n"
    "- Every query must have a LIMIT clause.\n"
    "- Use datetime() to wrap ISO 8601 date strings.\n"
    "- Alias all aggregations (count(*) AS total, sum(x) AS total_x).\n"
    "- Read-only only: no CREATE, MERGE, SET, DELETE, REMOVE.\n"
    "- One MATCH statement per query.\n\n"
    "Schema summary:\n"
    "  Nodes: Patient, Visit, Charge, Transaction, Statement,\n"
    "         PhoneBridge, RCCall, IVRInbound, DiallerCall,\n"
    "         Location, InsurancePlan, Practice, Campaign,\n"
    "         DiagnosisCode, ProcedureCode, BirdeyeReview\n"
    "  Key paths:\n"
    "    (Patient)-[:HAD_VISIT]->(Visit)-[:PERFORMED_AT]->(Location)\n"
    "    (Patient)-[:HAS_CHARGE]->(Charge)<-[:SETTLES]-(Transaction)\n"
    "    (Patient)-[:IDENTIFIED_BY_PHONE]->(PhoneBridge)<-[:ATTRIBUTED_TO_PHONE]-(RCCall)\n"
    "    (Charge)-[:PART_OF_VISIT]->(Visit)-[:UNDER_PLAN]->(InsurancePlan)\n"
    "    (Charge)-[:AT_LOCATION]->(Location)-[:BELONGS_TO_PRACTICE]->(Practice)\n"
    "    (Charge)-[:DIAGNOSED_WITH]->(DiagnosisCode)\n"
    "    (Charge)-[:USES_PROCEDURE]->(ProcedureCode)\n"
)


def load_model(adapter_path: str):
    print(f"Loading model...")
    print(f"  Base:    {settings.LORA_BASE_MODEL}")
    print(f"  Adapter: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=settings.LORA_BASE_MODEL,
        max_seq_length=getattr(settings, "MAX_SEQ_LENGTH", 2048),
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, "gemma")

    # Load LoRA weights on top
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print(f"  Model loaded")
    return model, tokenizer


def generate_cypher(question: str, model, tokenizer,
                    max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,
            top_k=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract the Cypher block (starts with MATCH or CALL)
    for keyword in ("MATCH", "CALL", "WITH"):
        if keyword in response.upper():
            idx = response.upper().index(keyword)
            return response[idx:].strip()
    return response


def execute_query(cypher: str, uri: str, user: str, password: str) -> list[dict]:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as s:
            row_limit = getattr(settings, "CYPHER_ROW_LIMIT", 100)
            limited = cypher if "LIMIT" in cypher.upper() else f"{cypher} LIMIT {row_limit}"
            timeout = getattr(settings, "CYPHER_TIMEOUT_SECONDS", 10)
            return s.run(limited).data()
    finally:
        driver.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--execute",  action="store_true")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default=None)
    parser.add_argument("--no-redact", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    print(f"\nQuestion: {args.question}")
    cypher = generate_cypher(args.question, model, tokenizer)
    print(f"\nGenerated Cypher:\n{cypher}")

    guard_result = check_cypher(cypher)
    guard_pass, guard_err = guard_result.ok, guard_result.reason
    print(f"\nGuardrail: {'PASS' if guard_pass else f'FAIL: {guard_err}'}")

    if not guard_pass:
        print("Query blocked by guardrail. Not executing.")
        return

    if args.execute:
        pw = args.password or os.getenv("NEO4J_PASSWORD", "")
        if not pw:
            print("Neo4j password required. Set NEO4J_PASSWORD or pass --password.")
            return
        try:
            print(f"\nExecuting on Neo4j {args.uri}...")
            rows = execute_query(cypher, args.uri, args.user, pw)
            if not args.no_redact:
                rows = redact_rows(rows)
            print(f"Returned {len(rows)} rows")
            for i, row in enumerate(rows[:10], 1):
                print(f"  {i}. {row}")
            if len(rows) > 10:
                print(f"  ... and {len(rows) - 10} more")
        except Exception as e:
            print(f"Execution failed: {e}")


if __name__ == "__main__":
    main()