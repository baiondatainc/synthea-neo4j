#!/usr/bin/env python3
"""
eval_runner.py  (fixed 2026-07-07 v3)
──────────────────────────────────────
Fixes:
  1. Handles both guardrail return types:
       - GuardrailResult object (.ok / .reason)  — new runtime/guardrails.py
       - plain (bool, str) tuple                  — older version
  2. Attention mask set explicitly to avoid pad==eos warning
  3. max_new_tokens only (no max_length conflict)
  4. Token repetition guard — if output is "AS AS AS..." the model needs retraining
     (training data format mismatch); we detect and skip gracefully

Usage:
    python scripts/eval_runner.py \\
        --model ./lora_adapter_v1 \\
        --eval  data/eval/eval.json \\
        --out   eval_results/v1.json \\
        --uri   bolt://localhost:7687 \\
        --user  neo4j --password <pw>
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from neo4j import GraphDatabase
from tqdm import tqdm
from prompt import SYSTEM_PROMPT

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings


# ── Guardrail wrapper — handles both return types ─────────────────────────────

def _call_guardrail(cypher: str) -> tuple[bool, Optional[str]]:
    """
    Call check_cypher() and normalise the return value.
    Handles:
      - GuardrailResult object  (.ok, .reason)
      - plain (bool, str) tuple
    """
    try:
        from runtime.guardrails import check_cypher
        result = check_cypher(cypher)
        # GuardrailResult dataclass
        if hasattr(result, "ok"):
            return bool(result.ok), result.reason
        # Plain tuple (bool, str)
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), result[1]
        # Fallback — treat as passing
        return True, None
    except Exception as e:
        return False, f"guardrail import error: {e}"


# ── JSON serialiser safe for numpy types ──────────────────────────────────────

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception:
            pass
        return super().default(obj)



REPETITION_RE = re.compile(r"(\bAS\b\s*){5,}", re.IGNORECASE)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_path: str):
    print(f"  Base:    {settings.LORA_BASE_MODEL}")
    print(f"  Adapter: {model_path}")
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from peft import PeftModel

        base, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = settings.LORA_BASE_MODEL,
            max_seq_length = getattr(settings, "MAX_SEQ_LENGTH", 2048),
            dtype          = None,
            load_in_4bit   = True,
        )
        FastLanguageModel.for_inference(base)
        tokenizer = get_chat_template(tokenizer, "gemma")
        model     = PeftModel.from_pretrained(base, model_path)
        model.eval()
        print("  ✓ Loaded via Unsloth + PeftModel")
        return model, tokenizer

    except Exception as e:
        print(f"  Unsloth failed ({e}), trying plain HuggingFace...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(settings.LORA_BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            settings.LORA_BASE_MODEL,
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported()
                          else torch.float16,
            device_map  = "auto",
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
        print("  ✓ Loaded via HuggingFace + PeftModel")
        return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def generate_cypher(question: str, model, tokenizer,
                    debug: bool = False) -> Optional[str]:
    messages = [{"role": "user", "content": SYSTEM_PROMPT + question}]

    # Simple tokenisation — no return_dict, no attention_mask complexity
    # Unsloth's patched tokenizer handles pad/eos correctly internally
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors        = "pt",
        add_generation_prompt = True,
    ).to(model.device)

    # Build attention mask manually: 1 everywhere (no padding in single-seq inference)
    attention_mask = torch.ones_like(input_ids)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids          = input_ids,
                attention_mask     = attention_mask,
                max_new_tokens     = 256,
                do_sample          = False,
                pad_token_id       = tokenizer.eos_token_id,
                eos_token_id       = tokenizer.eos_token_id,
                use_cache          = True,
                repetition_penalty = 1.3,
            )

        new_tokens = outputs[0][input_ids.shape[-1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if debug:
            print(f"\n  [DEBUG] question : {question[:80]}")
            print(f"  [DEBUG] raw resp : {response[:300]}")

        # Detect catastrophic repetition — model trained on wrong format
        if REPETITION_RE.search(response):
            if debug:
                print("  [DEBUG] ⚠ Repetition detected — model may need retraining")
            return None

        # Extract Cypher
        for kw in ("MATCH", "CALL", "WITH"):
            if kw in response.upper():
                idx = response.upper().index(kw)
                return response[idx:].strip()

        return response if len(response) > 10 else None

    except Exception as e:
        if debug:
            print(f"  [DEBUG] generate error: {e}")
        return None


# ── Neo4j execution ───────────────────────────────────────────────────────────

def execute_cypher(cypher: str, driver) -> tuple[bool, Optional[int], Optional[str]]:
    row_limit = getattr(settings, "CYPHER_ROW_LIMIT", 100)
    try:
        with driver.session() as s:
            limited = cypher if "LIMIT" in cypher.upper() else f"{cypher} LIMIT {row_limit}"
            rows    = s.run(limited).data()
        return True, len(rows), None
    except Exception as e:
        return False, None, str(e)[:120]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    df    = pd.DataFrame(results)
    total = len(df)

    gen   = int(df["generated"].sum())
    guard = int((df["generated"] & df["guardrail_pass"]).sum())
    exe   = int((df["generated"] & df["guardrail_pass"] & df["executable"]).sum())
    exe_n = int(df["executable"].sum())
    rows  = int(df[df["executable"] == True]["returns_rows"].fillna(False).sum())

    metrics = {
        "total_questions":     total,
        "generation_rate":     round(gen   / total * 100, 2) if total else 0.0,
        "guardrail_pass_rate": round(guard / total * 100, 2) if total else 0.0,
        "executable_rate":     round(exe   / total * 100, 2) if total else 0.0,
        "returns_rows_rate":   round(rows  / exe_n * 100, 2) if exe_n else 0.0,
    }

    by_cat = {}
    for cat in df["category"].unique():
        cdf  = df[df["category"] == cat]
        cn   = len(cdf)
        cgen = int(cdf["generated"].sum())
        cexe = int((cdf["generated"] & cdf["guardrail_pass"] & cdf["executable"]).sum())
        by_cat[str(cat)] = {
            "total":           cn,
            "generated":       cgen,
            "executable":      cexe,
            "generation_rate": round(cgen / cn * 100, 1) if cn else 0.0,
            "executable_rate": round(cexe / cn * 100, 1) if cn else 0.0,
        }
    metrics["by_category"] = by_cat
    return metrics


def gate_check(metrics: dict) -> bool:
    gen_floor   = getattr(settings, "GENERATION_RATE_FLOOR",     0.869) * 100
    guard_floor = getattr(settings, "GUARDRAIL_PASS_RATE_FLOOR",  0.869) * 100
    exe_floor   = getattr(settings, "EXECUTABLE_RATE_FLOOR",     0.990) * 100

    ok = True
    print("\nPromotion gate:")
    for label, actual, floor in [
        ("Generation rate",     metrics["generation_rate"],     gen_floor),
        ("Guardrail-pass rate", metrics["guardrail_pass_rate"], guard_floor),
        ("Executable rate",     metrics["executable_rate"],     exe_floor),
    ]:
        passed = actual >= floor
        mark   = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {label:<22} {actual:6.1f}%  vs  {floor:.1f}%  {mark}")
        if not passed: ok = False

    print(f"\n  Decision: {'PROMOTE ✅' if ok else 'HOLD ❌'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--eval",     default="data/eval/eval.json")
    parser.add_argument("--out",      default="eval_results/latest.json")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default=None)
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()

    pw = args.password or os.getenv("NEO4J_PASSWORD", "")
    if not pw:
        print("NEO4J_PASSWORD not set.")
        sys.exit(1)

    print("=" * 62)
    print("  Text2Cypher Evaluation")
    print(f"  Model:     {args.model}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 62)

    if not Path(args.eval).exists():
        print(f"Eval file not found: {args.eval}")
        sys.exit(1)

    with open(args.eval) as f:
        eval_data = json.load(f)
    questions = eval_data.get("questions", eval_data)
    if args.limit:
        questions = questions[:args.limit]
    print(f"Eval set: {len(questions)} questions\n")

    print("Loading model...")
    model, tokenizer = load_model(args.model)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, pw))

    results = []
    for i, item in enumerate(tqdm(questions, desc="Evaluating")):
        question = item["question"]
        debug    = (i == 0) and not args.no_debug

        cypher    = generate_cypher(question, model, tokenizer, debug=debug)
        generated = bool(
            cypher and len(cypher) > 8
            and any(k in cypher.upper() for k in ("MATCH", "CALL", "WITH", "RETURN"))
        )

        guard_pass, guard_err = False, "not generated"
        if generated:
            guard_pass, guard_err = _call_guardrail(cypher)

        executable = False
        row_count  = None
        exec_err   = None
        if generated and guard_pass:
            executable, row_count, exec_err = execute_cypher(cypher, driver)

        results.append({
            "question":        question,
            "category":        str(item.get("category", "unknown")),
            "difficulty":      str(item.get("difficulty", "medium")),
            "generated_cypher": cypher,
            "generated":       bool(generated),
            "guardrail_pass":  bool(guard_pass),
            "guardrail_error": str(guard_err) if guard_err else None,
            "executable":      bool(executable),
            "row_count":       int(row_count) if row_count is not None else None,
            "exec_error":      str(exec_err) if exec_err else None,
            "returns_rows":    bool((row_count or 0) > 0) if executable else None,
        })

    driver.close()

    metrics = compute_metrics(results)

    print(f"\n{'='*62}")
    print(f"  Results")
    print(f"{'='*62}")
    print(f"  Generation rate:    {metrics['generation_rate']:.1f}%")
    print(f"  Guardrail pass:     {metrics['guardrail_pass_rate']:.1f}%")
    print(f"  Executable rate:    {metrics['executable_rate']:.1f}%")
    print(f"  Returns rows:       {metrics['returns_rows_rate']:.1f}%")

    print(f"\n  By category:")
    for cat, m in sorted(metrics["by_category"].items()):
        bar = "█" * int(m["generation_rate"] / 10)
        print(f"  {cat:<35} gen={m['generation_rate']:5.1f}%  "
              f"exe={m['executable_rate']:5.1f}%  {bar}")

    promote = gate_check(metrics)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "model": args.model,
             "metrics": metrics, "promote": promote, "results": results},
            f, indent=2, cls=SafeEncoder,
        )
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()