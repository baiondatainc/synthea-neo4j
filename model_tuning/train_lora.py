#!/usr/bin/env python3
"""
train_lora.py  v6  — aligned with Modelfile.qwen3-4b (2026-07)
─────────────────────────────────────────────────────────────
Fine-tunes Qwen3-4B-Instruct-2507 for RP Text2Cypher.

Why Qwen3-4B-Instruct-2507 over Qwen2.5-Coder-7B:
  - Matches the NEW production Ollama model (Modelfile.qwen3-4b,
    FROM qwen3:4b-instruct-2507-q4_K_M)
  - Non-thinking instruct variant → same plain ChatML template, no <think>
  - Smaller (4B vs 7B) → faster training, faster CPU inference in Ollama

Changes vs v5:
  1. Base model → Qwen3-4B-Instruct-2507 (was Qwen2.5-Coder-7B)
  2. LORA_BASE_MODEL is now actually IMPORTED from prompt.py
     (v5 referenced it without importing — NameError at runtime)
  3. SYSTEM_PROMPT comes from Modelfile.qwen3-4b via prompt.py — the full
     ~7-8k token production prompt, so --max-seq-length default is now 10240
     (was 4096, which cannot fit the real system prompt)
  4. Chat template: prefer the tokenizer's native Qwen3 ChatML; fatal check
     that no <think> block appears in formatted examples (would mean the
     wrong/thinking base model was loaded)
  5. check_pair: LIMIT only required for NON-aggregate queries — the
     Modelfile's own examples ("How many patients?") have no LIMIT and are
     correct; v5 would have dropped them from training data
  6. Deploy instructions point at Modelfile.qwen3-4b / text2cypher-q3-ft

Expected loss:
  Smoke test (50 rows, 3 steps): 2-5  (not converged, just verifies format)
  Full train epoch 1:            1.0-2.0
  Full train epoch 3:            0.2-0.5

Usage:
    # Smoke test first (2 min):
    python scripts/train_lora.py \\
        --train-data data/splits/train.parquet \\
        --val-data   data/splits/val.parquet \\
        --output-dir ./lora_smoke_test \\
        --num-epochs 1 --smoke-test --skip-gguf

    # Full training:
    python scripts/train_lora.py \\
        --train-data data/splits/train.parquet \\
        --val-data   data/splits/val.parquet \\
        --output-dir ./lora_adapter_q3_v1
"""

import unsloth  # noqa: F401 — must be FIRST

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from prompt import (
    SYSTEM_PROMPT,
    RESPONSE_TEMPLATE,
    WRITE_RE,
    WRONG_PROPS,
    AGG_RE,
    LORA_BASE_MODEL,   # ← v5 used this without importing it
)

# DataCollatorForCompletionOnlyLM moved between trl and transformers across versions
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from transformers import DataCollatorForCompletionOnlyLM
    except ImportError:
        # Implement a minimal version inline if neither works
        from transformers import DataCollatorForLanguageModeling

        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            def __init__(self, response_template, tokenizer, *args, **kwargs):
                super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
                self.response_template = response_template
                self.response_token_ids = response_template

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                for i, label in enumerate(batch["labels"]):
                    input_ids = batch["input_ids"][i].tolist()
                    tmpl = self.response_token_ids
                    for j in range(len(input_ids) - len(tmpl) + 1):
                        if input_ids[j:j + len(tmpl)] == tmpl:
                            batch["labels"][i, :j + len(tmpl)] = -100
                            break
                    else:
                        batch["labels"][i] = torch.full_like(label, -100)
                return batch

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def check_pair(question: str, cypher: str) -> tuple[bool, str]:
    if not question or not question.strip(): return False, "empty question"
    if not cypher  or not cypher.strip():   return False, "empty cypher"
    u = cypher.upper()
    if "MATCH"  not in u: return False, "no MATCH"
    if "RETURN" not in u: return False, "no RETURN"
    # LIMIT is only mandatory for non-aggregate result sets (per Modelfile).
    # Aggregate queries like "MATCH (p:Patient) RETURN count(p) AS n" are valid
    # without LIMIT — v5 wrongly dropped these.
    if "LIMIT" not in u and not AGG_RE.search(cypher):
        return False, "no LIMIT (non-aggregate)"
    if WRITE_RE.search(cypher):      return False, "write keyword"
    if WRONG_PROPS.search(cypher):   return False, "wrong_property_name"
    if len(cypher) < 20:             return False, "too short"
    return True, "ok"


def validate_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    from collections import Counter
    ok_mask, reasons = [], []
    for _, r in df.iterrows():
        ok, reason = check_pair(r["question"], r["cypher"])
        ok_mask.append(ok)
        if not ok: reasons.append(reason)
    valid_df = df[pd.Series(ok_mask, index=df.index)].copy()
    dropped  = len(df) - len(valid_df)
    print(f"  {name}: {len(valid_df):,} valid / {len(df):,} total", end="")
    if dropped: print(f"  (dropped {dropped}: {dict(Counter(reasons))})")
    else: print()
    if len(valid_df) == 0:
        print("  ✗ No valid pairs — re-run validate_pairs.py")
        sys.exit(1)
    return valid_df


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING — Qwen3 ChatML (identical markup to Qwen2.5; NO <think> blocks
# in the 2507 instruct variant)
# ─────────────────────────────────────────────────────────────────────────────

def make_formatting_func(tokenizer):
    """
    Qwen3 ChatML format:
      <|im_start|>system
      SYSTEM_PROMPT<|im_end|>
      <|im_start|>user
      QUESTION<|im_end|>
      <|im_start|>assistant
      CYPHER<|im_end|>

    Loss is computed only on the assistant (Cypher) turn via DataCollator.
    """
    def formatting_func(examples: dict) -> list[str]:
        texts = []
        for q, c in zip(examples["question"], examples["cypher"]):
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT.strip()},
                {"role": "user",      "content": str(q).strip()},
                {"role": "assistant", "content": str(c).strip()},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize              = False,
                add_generation_prompt = False,
            )
            texts.append(text)
        return texts
    return formatting_func


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning — Qwen3-4B-Instruct-2507 for RP Text2Cypher"
    )
    parser.add_argument("--train-data",     default="data/splits/train.parquet")
    parser.add_argument("--val-data",       default="data/splits/val.parquet")
    parser.add_argument("--num-epochs",     type=int,   default=3)
    parser.add_argument("--output-dir",     default="./lora_adapter_q3_v1")
    parser.add_argument("--batch-size",     type=int,   default=1,
                        help="Per-device batch. Seq length is ~8-9k tokens now; "
                             "raise to 2 only if VRAM allows.")
    parser.add_argument("--eval-steps",     type=int,   default=50)
    parser.add_argument("--save-steps",     type=int,   default=50)   # was 100
    parser.add_argument("--max-seq-length", type=int,   default=10240,
                        help="Must fit the FULL Modelfile system prompt (~7-8k "
                             "tokens) + question + Cypher. Ollama num_ctx is 9216.")
    parser.add_argument("--learning-rate",  type=float, default=None)
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Run on 50 rows to verify pipeline (2 min)")
    parser.add_argument("--skip-gguf",      action="store_true")
    
    args = parser.parse_args()

    lr = args.learning_rate or settings.LEARNING_RATE

    print("=" * 70)
    print("  LoRA Fine-Tuning — Qwen3-4B-Instruct-2507  (Text2Cypher v6)")
    print("=" * 70)
    print(f"  Base model:     {LORA_BASE_MODEL}")
    print(f"  Output:         {args.output_dir}")
    print(f"  max_seq_length: {args.max_seq_length}")
    print(f"  Smoke test:     {args.smoke_test}")
    print(f"  Timestamp:      {datetime.now().isoformat()}")
    print()

    # ── 1. Load and validate data ─────────────────────────────────────────────
    print("📂 Loading datasets...")
    for p in [args.train_data, args.val_data]:
        if not Path(p).exists():
            print(f"  ✗ Not found: {p}")
            sys.exit(1)

    train_df = pd.read_parquet(args.train_data)
    val_df   = pd.read_parquet(args.val_data)
    for df in [train_df, val_df]:
        df["question"] = df["question"].astype(str)
        df["cypher"]   = df["cypher"].astype(str)

    train_df = validate_dataset(train_df, "train")
    val_df   = validate_dataset(val_df,   "val")

    if args.smoke_test:
        train_df = train_df.sample(n=min(50, len(train_df)), random_state=42)
        val_df   = val_df.sample(n=min(10, len(val_df)),     random_state=42)
        print(f"  Smoke test: {len(train_df)} train / {len(val_df)} val")

    train_dataset = Dataset.from_pandas(
        train_df[["question","cypher"]], preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        val_df[["question","cypher"]], preserve_index=False
    )

    # ── 2. Load Qwen3-4B-Instruct-2507 ────────────────────────────────────────
    print(f"\n📥 Loading {LORA_BASE_MODEL} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = LORA_BASE_MODEL,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )
    # The 2507 instruct tokenizer already ships a plain ChatML template with no
    # <think> handling — keep it. Only remap if unsloth knows a qwen3 template.
    for tmpl_name in ("qwen3-instruct", "qwen3", "qwen-3"):
        try:
            tokenizer = get_chat_template(tokenizer, tmpl_name)
            print(f"   ✓ chat template: {tmpl_name}")
            break
        except Exception:
            continue
    else:
        print("   ✓ chat template: tokenizer native (ChatML)")
    print(f"   ✓ {model.config.model_type} loaded")

    # ── 3. LoRA adapter ───────────────────────────────────────────────────────
    print(f"\n🎯 LoRA  r={settings.LORA_RANK}  α={settings.LORA_ALPHA}  "
          f"rsLoRA={settings.LORA_USE_RSLORA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r               = settings.LORA_RANK,
        lora_alpha      = settings.LORA_ALPHA,
        target_modules  = settings.LORA_TARGET_MODULES,
        lora_dropout    = settings.LORA_DROPOUT,
        bias            = settings.LORA_BIAS,
        use_rslora      = settings.LORA_USE_RSLORA,
        use_gradient_checkpointing = settings.LORA_USE_GRADIENT_CHECKPOINTING,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"   ✓ {trainable:,.0f} / {total:,.0f} trainable ({trainable/total*100:.2f}%)")

    # ── 4. Verify format + token budget ──────────────────────────────────────
    print(f"\n🔍 Verifying format and token budget...")
    fmt_fn = make_formatting_func(tokenizer)
    sample = fmt_fn({
        "question": [train_df.iloc[0]["question"]],
        "cypher":   [train_df.iloc[0]["cypher"]],
    })
    ex        = sample[0]
    tok_count = len(tokenizer.encode(ex))

    print(f"   Tokens per example:  {tok_count}")
    print(f"   max_seq_length:      {args.max_seq_length}")
    print(f"   Headroom:            {args.max_seq_length - tok_count}")
    print(f"   Start: {ex[:80]!r}")
    print(f"   End:   {ex[-50:]!r}")

    if tok_count > args.max_seq_length - 100:
        print(f"   ✗ FATAL: example too long ({tok_count} tokens). "
              f"Increase --max-seq-length.")
        sys.exit(1)

    # Guard: a <think> block means the THINKING qwen3 base/template was loaded
    # instead of the 2507 instruct variant — masking and Ollama serving would
    # both break silently.
    if "<think>" in ex:
        print("   ✗ FATAL: <think> block in formatted example — wrong base model")
        print("     Use Qwen/Qwen3-4B-Instruct-2507 (non-thinking), not qwen3 base.")
        sys.exit(1)

    # Verify response template exists
    if RESPONSE_TEMPLATE not in ex:
        print(f"   ✗ FATAL: response template not found: {RESPONSE_TEMPLATE!r}")
        print(f"   Full example:\n{ex[:500]}")
        sys.exit(1)

    # Verify Cypher is in assistant turn
    cypher_start = train_df.iloc[0]["cypher"][:20]
    if cypher_start not in ex:
        print(f"   ✗ FATAL: Cypher not found in formatted example")
        sys.exit(1)

    print(f"   ✓ Format verified — Qwen3 ChatML with Cypher in assistant turn")

    # ── 5. Response masking collator ─────────────────────────────────────────
    print(f"\n🎯 Response masking — loss computed ONLY on Cypher tokens")
    resp_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    print(f"   Response template IDs: {resp_ids}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template = resp_ids,
        tokenizer         = tokenizer,
    )
    print(f"   ✓ Prompt tokens masked (label=-100)")

    # ── 6. Training config ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_dataset) //
                         (args.batch_size * settings.GRADIENT_ACCUMULATION_STEPS))
    total_steps  = steps_per_epoch * args.num_epochs
    warmup_steps = max(1, round(total_steps * settings.WARMUP_RATIO))

    print(f"\n⚙️  Training")
    print(f"   {len(train_dataset):,} train / {len(val_dataset):,} val")
    print(f"   {args.num_epochs} epochs  ·  {steps_per_epoch} steps/epoch  "
          f"·  {total_steps} total  ·  warmup={warmup_steps}")
    print(f"   LR={lr:.2e}  batch={args.batch_size * settings.GRADIENT_ACCUMULATION_STEPS}")
    if args.smoke_test:
        print(f"   Smoke test: expect loss 2-6 (only {total_steps} steps)")
    else:
        print(f"   Target loss: epoch1 <1.5  epoch3 <0.5")

    sft_config = SFTConfig(
        output_dir                  = settings.CHECKPOINT_DIR,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = settings.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps                = warmup_steps,
        num_train_epochs            = args.num_epochs,
        learning_rate               = lr,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        optim                       = "adamw_8bit",
        weight_decay                = settings.WEIGHT_DECAY,
        lr_scheduler_type           = settings.LR_SCHEDULER_TYPE,
        seed                        = 42,
        report_to                   = "none",
        max_grad_norm               = settings.MAX_GRAD_NORM,
        max_seq_length              = args.max_seq_length,
        packing                     = False,
        per_device_eval_batch_size = 1,     # eval was silently using batch 8
        prediction_loss_only       = True,  # don't retain logits — loss is all we need
        eval_accumulation_steps    = 1,
    )

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    print(f"\n🏋️  Initializing trainer...")
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        formatting_func  = make_formatting_func(tokenizer),
        data_collator    = collator,       # ← response masking: loss on Cypher only
        args             = sft_config,
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    print(f"\n🎓 Training...")
    print("=" * 70)
    stats = trainer.train()
    print("=" * 70)

    loss = stats.metrics.get("train_loss", "?")
    time = stats.metrics.get("train_runtime", 0)
    print(f"   Train loss: {loss:.4f}" if isinstance(loss, float) else f"   Loss: {loss}")
    print(f"   Time:       {time/60:.1f} min")

    if isinstance(loss, float):
        if args.smoke_test and loss < 10:
            print("\n  ✓ Smoke test PASSED — format is correct")
            print(f"    Run full training: remove --smoke-test --skip-gguf flags")
        elif not args.smoke_test and loss < 0.5:
            print("\n  ✓ Excellent. Proceed to eval_runner.py")
        elif not args.smoke_test and loss < 1.0:
            print("\n  ✓ Good. Run eval_runner.py")
        elif not args.smoke_test:
            print(f"\n  ⚠  Loss {loss:.2f} — consider 1-2 more epochs")

    # ── 9. Save adapter ───────────────────────────────────────────────────────
    print(f"\n💾 Saving → {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"   ✓ Saved")

    # ── 10. GGUF ──────────────────────────────────────────────────────────────
    if not args.skip_gguf:
        print(f"\n🔧 Exporting GGUF (q4_k_m)...")
        os.makedirs(settings.GGUF_EXPORT_DIR, exist_ok=True)
        try:
            model.save_pretrained_gguf(
                settings.GGUF_EXPORT_DIR, tokenizer,
                quantization_method="q4_k_m"
            )
            print(f"   ✓ {settings.GGUF_EXPORT_DIR}/model-unsloth.Q4_K_M.gguf")
            print(f"\n   To deploy:")
            print(f"   # Update Modelfile.qwen3-4b FROM line:")
            print(f"   # FROM {settings.GGUF_EXPORT_DIR}/model-unsloth.Q4_K_M.gguf")
            print(f"   ollama create text2cypher-q3-ft -f Modelfile.qwen3-4b")
        except Exception as e:
            print(f"   ⚠  GGUF failed: {e}")

    print(f"\n{'='*70}")
    print(f"  ✅ Done → {args.output_dir}")
    print(f"{'='*70}")
    print(f"\n  python scripts/quick_check.py --model {args.output_dir}")
    print(f"  python scripts/eval_runner.py  --model {args.output_dir} \\")
    print(f"    --eval data/eval/eval.json --uri bolt://localhost:7687\n")


if __name__ == "__main__":
    main()