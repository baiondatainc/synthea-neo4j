#!/usr/bin/env python3
"""
quick_check.py  v2 — aligned with train_lora.py v6 (Qwen3-4B-Instruct-2507)
──────────────────────────────────────────────────────────────────────────
Smoke-test a trained adapter: generate the TEST_QUESTIONS and show raw output.
Run this BEFORE the full eval.

Fixes vs v1 (which crashed with LoRA size mismatches):
  1. Base model comes from prompt.LORA_BASE_MODEL (Qwen3-4B), NOT
     settings.LORA_BASE_MODEL (stale Qwen2.5-Coder-7B → shape mismatch
     2560 vs 3584).
  2. Adapter is loaded by passing its path straight to
     FastLanguageModel.from_pretrained — unsloth reads adapter_config.json,
     pulls the right base itself, and attaches the adapter. No PeftModel,
     no chance of pairing the adapter with the wrong base.
  3. Chat template: NOT "gemma". Uses the same qwen3-instruct/native ChatML
     fallback chain as training.
  4. Prompt format matches training exactly: SYSTEM_PROMPT in the system
     role, question in the user role (v1 concatenated both into user).
  5. max_seq_length 10240 (v1's 2048 silently truncated the ~7.8k-token
     system prompt).
  6. Quality check: LIMIT only required for non-aggregate queries, plus the
     WRONG_PROPS / write-clause gates from prompt.py.

Usage:
    python model_tuning/quick_check.py --model ./lora_adapter_q3_v1
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from prompt import (
    SYSTEM_PROMPT, TEST_QUESTIONS, LORA_BASE_MODEL,
    WRITE_RE, WRONG_PROPS, AGG_RE,
)


def check_cypher(text: str) -> tuple[bool, list[str]]:
    problems = []
    u = text.upper()
    if "MATCH" not in u:
        problems.append("no MATCH")
    if "RETURN" not in u:
        problems.append("no RETURN")
    if "LIMIT" not in u and not AGG_RE.search(text):
        problems.append("no LIMIT on non-aggregate")
    if WRITE_RE.search(text):
        problems.append("write clause")
    if WRONG_PROPS.search(text):
        problems.append("wrong property name")
    if "```" in text:
        problems.append("markdown fence (should be raw Cypher)")
    if "<think>" in text:
        problems.append("<think> block (wrong base model?)")
    return (not problems), problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Adapter dir (e.g. ./lora_adapter_q3_v1)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # Passing the ADAPTER path directly: unsloth reads adapter_config.json,
    # loads the matching base (Qwen3-4B), and attaches the adapter. This makes
    # a base/adapter mismatch impossible.
    print(f"Loading adapter {args.model} (base per adapter_config: "
          f"{LORA_BASE_MODEL}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model,
        max_seq_length = 10240,     # system prompt alone is ~7.8k tokens
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)

    # Same template chain as training — never "gemma".
    for tmpl_name in ("qwen3-instruct", "qwen3", "qwen-3"):
        try:
            tokenizer = get_chat_template(tokenizer, tmpl_name)
            print(f"✓ chat template: {tmpl_name}")
            break
        except Exception:
            continue
    else:
        print("✓ chat template: tokenizer native (ChatML)")
    model.eval()
    print("✓ Loaded\n")

    passed = 0
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}] {q}")
        # EXACTLY the training format: system role + user role.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user",   "content": q},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids      = input_ids,
                attention_mask = torch.ones_like(input_ids),
                max_new_tokens = args.max_new_tokens,
                do_sample      = False,          # match Ollama temperature 0
                pad_token_id   = tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

        print(f"    {response[:300]}")
        ok, problems = check_cypher(response)
        print(f"    {'✓ LOOKS GOOD' if ok else '✗ ' + ', '.join(problems)}\n")
        passed += ok

    print("=" * 50)
    print(f"  {passed}/{len(TEST_QUESTIONS)} questions produced valid-looking Cypher")
    if passed == len(TEST_QUESTIONS):
        print("  ✓ Proceed to eval_runner.py, then benchmark vs the base model")
    elif passed >= len(TEST_QUESTIONS) - 2:
        print("  ✓ Mostly working — run full eval for precise scores")
    else:
        print("  ✗ Check: was the adapter trained with THIS prompt.py "
              "(Modelfile-sourced SYSTEM_PROMPT)?")
    print("=" * 50)


if __name__ == "__main__":
    main()