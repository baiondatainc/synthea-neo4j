#!/usr/bin/env python3
"""
train_lora.py  v7  — clean export chain (2026-07)
─────────────────────────────────────────────────────────────
Fine-tunes Qwen3-4B-Instruct-2507 for RP Text2Cypher and produces a
DEPLOYABLE q4_k_m GGUF in one run.

Changes vs v6 (why v6's GGUF emitted EOS after one token):
  1. load_in_4bit=False — v6's 4-bit load made unsloth silently swap to its
     bnb-4bit mirror, whose tokenizer lacks a pad token. Unsloth then ADDED
     <|PAD_TOKEN|> (vocab resize) and any merge dequantized from nf4.
     Both poisons produced a GGUF whose adapter worked in Python but died
     after one token in Ollama. 16-bit load eliminates both, and trains
     FASTER (no dequant overhead). 4B bf16 fits easily in 23.5 GB.
  2. Pad-token guard — even if a future unsloth version tries, pad is pinned
     to the EXISTING <|endoftext|> token: no vocab resize, ever.
  3. Section 10 rewritten — unsloth's save_pretrained_gguf (the component
     that produced the broken artifact) is replaced with the manual chain
     verified by hand: save_pretrained_merged(16bit) → llama.cpp
     convert_hf_to_gguf → llama-quantize q4_k_m.
  4. num_epochs default 3 → 2 — v6's loss hit 0.017 by epoch 1.2; epoch 3
     was pure overfitting risk on a ~350-pair dataset.
  5. Keeps the v6.1 OOM fixes: per_device_eval_batch_size=1,
     prediction_loss_only=True, eval_accumulation_steps=1, save_steps=50.

Usage:
    # Smoke test first (2 min):
    python model_tuning/train_lora.py \\
        --train-data data/splits/train.parquet \\
        --val-data   data/splits/val.parquet \\
        --output-dir ./lora_smoke_test \\
        --num-epochs 1 --smoke-test --skip-gguf

    # Full training + deployable GGUF (~30 min):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python model_tuning/train_lora.py \\
        --train-data data/splits/train.parquet \\
        --val-data   data/splits/val.parquet \\
        --output-dir ./lora_adapter_q3_v2

    # Deploy:
    #   set Modelfile.qwen3-4b FROM → <GGUF_EXPORT_DIR>/model-q4_k_m.gguf
    #   ollama create text2cypher-ft-candidate -f Modelfile.qwen3-4b
"""

import unsloth  # noqa: F401 — must be FIRST

import argparse
import os
import shutil
import subprocess
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
    LORA_BASE_MODEL,
)

# DataCollatorForCompletionOnlyLM moved between trl and transformers across versions
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from transformers import DataCollatorForCompletionOnlyLM
    except ImportError:
        # Minimal inline fallback (verified: masks everything before the
        # response template; quick_check 5/5 confirms the adapter it trained)
        from transformers import DataCollatorForLanguageModeling

        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            def __init__(self, response_template, tokenizer, *args, **kwargs):
                super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
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

LLAMA_CPP = Path.home() / ".unsloth" / "llama.cpp"


def check_pair(question: str, cypher: str) -> tuple[bool, str]:
    if not question or not question.strip(): return False, "empty question"
    if not cypher  or not cypher.strip():   return False, "empty cypher"
    u = cypher.upper()
    if "MATCH"  not in u: return False, "no MATCH"
    if "RETURN" not in u: return False, "no RETURN"
    # LIMIT only mandatory for non-aggregate result sets (per Modelfile).
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
# FORMATTING — Qwen3 ChatML (2507 instruct: no <think> blocks)
# ─────────────────────────────────────────────────────────────────────────────

def make_formatting_func(tokenizer):
    def formatting_func(examples: dict) -> list[str]:
        texts = []
        for q, c in zip(examples["question"], examples["cypher"]):
            messages = [
                {"role": "system",    "content": SYSTEM_PROMPT.strip()},
                {"role": "user",      "content": str(q).strip()},
                {"role": "assistant", "content": str(c).strip()},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False))
        return texts
    return formatting_func


# ─────────────────────────────────────────────────────────────────────────────
# GGUF EXPORT — manual chain (merge 16bit → convert → quantize)
# ─────────────────────────────────────────────────────────────────────────────

def export_gguf(model, tokenizer, output_dir: str) -> Path | None:
    """Merge adapter into 16-bit weights, convert with llama.cpp, quantize.

    Replaces unsloth's save_pretrained_gguf, which produced a GGUF that
    emitted EOS after one token (broken tokenizer/EOS metadata from the
    4-bit-mirror tokenizer). This chain was verified by hand.
    """
    convert  = LLAMA_CPP / "convert_hf_to_gguf.py"
    quantize = LLAMA_CPP / "llama-quantize"
    if not convert.exists() or not quantize.exists():
        print(f"   ✗ llama.cpp tools not found under {LLAMA_CPP} — "
              f"skipping GGUF. Convert manually later.")
        return None

    merged_dir = Path(output_dir).with_name(Path(output_dir).name + "_merged")
    gguf_dir   = Path(settings.GGUF_EXPORT_DIR)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    bf16 = gguf_dir / "model-bf16.gguf"
    q4   = gguf_dir / "model-q4_k_m.gguf"

    print(f"\n🔧 [1/3] Merging adapter into 16-bit weights → {merged_dir}")
    model.save_pretrained_merged(str(merged_dir), tokenizer,
                                 save_method="merged_16bit")

    print(f"🔧 [2/3] Converting to GGUF bf16 → {bf16}")
    subprocess.run([sys.executable, str(convert), str(merged_dir),
                    "--outfile", str(bf16), "--outtype", "bf16"], check=True)

    print(f"🔧 [3/3] Quantizing to q4_k_m → {q4}")
    subprocess.run([str(quantize), str(bf16), str(q4), "q4_k_m"], check=True)

    bf16.unlink(missing_ok=True)     # reclaim ~8 GB
    shutil.rmtree(merged_dir, ignore_errors=True)
    return q4


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning — Qwen3-4B-Instruct-2507 for RP Text2Cypher (v7)"
    )
    parser.add_argument("--train-data",     default="data/splits/train.parquet")
    parser.add_argument("--val-data",       default="data/splits/val.parquet")
    parser.add_argument("--num-epochs",     type=int,   default=2,
                        help="v6 hit loss 0.017 by epoch 1.2 — 2 is plenty")
    parser.add_argument("--output-dir",     default="./lora_adapter_q3_v2")
    parser.add_argument("--batch-size",     type=int,   default=1)
    parser.add_argument("--eval-steps",     type=int,   default=50)
    parser.add_argument("--save-steps",     type=int,   default=50)
    parser.add_argument("--max-seq-length", type=int,   default=10240,
                        help="Must fit the full Modelfile system prompt "
                             "(~7-8k tokens) + question + Cypher")
    parser.add_argument("--learning-rate",  type=float, default=None)
    parser.add_argument("--smoke-test",     action="store_true")
    parser.add_argument("--skip-gguf",      action="store_true")
    args = parser.parse_args()

    lr = args.learning_rate or settings.LEARNING_RATE

    print("=" * 70)
    print("  LoRA Fine-Tuning — Qwen3-4B-Instruct-2507  (Text2Cypher v7)")
    print("=" * 70)
    print(f"  Base model:     {LORA_BASE_MODEL}  (16-bit load)")
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
        train_df[["question", "cypher"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(
        val_df[["question", "cypher"]], preserve_index=False)

    # ── 2. Load Qwen3-4B-Instruct-2507 in 16-bit ──────────────────────────────
    # load_in_4bit=False is THE v7 fix: the 4-bit path swapped in unsloth's
    # bnb-4bit mirror whose tokenizer had no pad token → <|PAD_TOKEN|> added,
    # vocab resized, merge sourced from nf4 → broken GGUF (EOS after 1 token).
    print(f"\n📥 Loading {LORA_BASE_MODEL} (16-bit) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = LORA_BASE_MODEL,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = False,
    )
    for tmpl_name in ("qwen3-instruct", "qwen3", "qwen-3"):
        try:
            tokenizer = get_chat_template(tokenizer, tmpl_name)
            print(f"   ✓ chat template: {tmpl_name}")
            break
        except Exception:
            continue
    else:
        print("   ✓ chat template: tokenizer native (ChatML)")

    # Pad guard: pin pad to an EXISTING token so no code path can ever resize
    # the vocabulary again.
    if tokenizer.pad_token is None or "<|PAD_TOKEN|>" in str(tokenizer.pad_token):
        tokenizer.pad_token = "<|endoftext|>"
        print("   ✓ pad_token pinned to <|endoftext|> (no vocab resize)")
    print(f"   ✓ {model.config.model_type} loaded  "
          f"(vocab={len(tokenizer):,} — must match base, no added tokens)")

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
    print(f"   ✓ {trainable:,.0f} / {total:,.0f} trainable "
          f"({trainable / total * 100:.2f}%)")

    # ── 4. Verify format + token budget ──────────────────────────────────────
    print(f"\n🔍 Verifying format and token budget...")
    fmt_fn = make_formatting_func(tokenizer)
    ex = fmt_fn({"question": [train_df.iloc[0]["question"]],
                 "cypher":   [train_df.iloc[0]["cypher"]]})[0]
    tok_count = len(tokenizer.encode(ex))

    print(f"   Tokens per example:  {tok_count}")
    print(f"   Headroom:            {args.max_seq_length - tok_count}")
    print(f"   Start: {ex[:80]!r}")
    print(f"   End:   {ex[-50:]!r}")

    if tok_count > args.max_seq_length - 100:
        print(f"   ✗ FATAL: example too long ({tok_count}). "
              f"Increase --max-seq-length."); sys.exit(1)
    if "<think>" in ex:
        print("   ✗ FATAL: <think> block — wrong (thinking) base model loaded")
        sys.exit(1)
    if RESPONSE_TEMPLATE not in ex:
        print(f"   ✗ FATAL: response template not found: {RESPONSE_TEMPLATE!r}")
        sys.exit(1)
    if train_df.iloc[0]["cypher"][:20] not in ex:
        print("   ✗ FATAL: Cypher not found in formatted example"); sys.exit(1)
    print("   ✓ Format verified — Qwen3 ChatML with Cypher in assistant turn")

    # ── 5. Response masking collator ─────────────────────────────────────────
    print(f"\n🎯 Response masking — loss computed ONLY on Cypher tokens")
    resp_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    print(f"   Response template IDs: {resp_ids}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=resp_ids, tokenizer=tokenizer)
    print("   ✓ Prompt tokens masked (label=-100)")

    # ── 6. Training config ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(train_dataset) //
                          (args.batch_size * settings.GRADIENT_ACCUMULATION_STEPS))
    total_steps  = steps_per_epoch * args.num_epochs
    warmup_steps = max(1, round(total_steps * settings.WARMUP_RATIO))

    print(f"\n⚙️  Training")
    print(f"   {len(train_dataset):,} train / {len(val_dataset):,} val")
    print(f"   {args.num_epochs} epochs · {steps_per_epoch} steps/epoch · "
          f"{total_steps} total · warmup={warmup_steps}")
    print(f"   LR={lr:.2e}  effective batch="
          f"{args.batch_size * settings.GRADIENT_ACCUMULATION_STEPS}")

    sft_config = SFTConfig(
        output_dir                  = settings.CHECKPOINT_DIR,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = 1,     # default 8 caused the eval OOM
        prediction_loss_only        = True,  # never retain eval logits
        eval_accumulation_steps     = 1,
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
    )

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    print(f"\n🏋️  Initializing trainer...")
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        formatting_func  = make_formatting_func(tokenizer),
        data_collator    = collator,
        args             = sft_config,
    )

    # ── 8. Train ──────────────────────────────────────────────────────────────
    print(f"\n🎓 Training...")
    print("=" * 70)
    stats = trainer.train()
    print("=" * 70)

    loss = stats.metrics.get("train_loss", "?")
    t = stats.metrics.get("train_runtime", 0)
    print(f"   Train loss: {loss:.4f}" if isinstance(loss, float) else f"   Loss: {loss}")
    print(f"   Time:       {t / 60:.1f} min")

    # ── 9. Save adapter ───────────────────────────────────────────────────────
    print(f"\n💾 Saving adapter → {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("   ✓ Saved")

    # ── 10. Merge + GGUF (manual, verified chain) ─────────────────────────────
    if not args.skip_gguf:
        try:
            q4 = export_gguf(model, tokenizer, args.output_dir)
            if q4:
                print(f"\n   ✓ Deployable GGUF: {q4.resolve()}")
                print(f"\n   To deploy:")
                print(f"   1. Modelfile.qwen3-4b FROM → {q4.resolve()}")
                print(f"   2. ollama create text2cypher-ft-candidate -f Modelfile.qwen3-4b")
                print(f"   3. ollama run text2cypher-ft-candidate \"How many patients?\"")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠  GGUF chain failed at: {e.cmd}")
            print(f"      Adapter is saved — convert manually per the docstring.")

    print(f"\n{'=' * 70}")
    print(f"  ✅ Done → {args.output_dir}")
    print(f"{'=' * 70}")
    print(f"\n  python model_tuning/quick_check.py --model {args.output_dir}")
    print(f"  python model_tuning/eval_runner.py  --model {args.output_dir} \\")
    print(f"    --eval data/eval/eval.json --uri bolt://localhost:7687\n")


if __name__ == "__main__":
    main()