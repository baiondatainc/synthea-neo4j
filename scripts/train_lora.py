#!/usr/bin/env python3
"""
train_lora.py  v5  — aligned with Modelfile.jp (2026-07-07)
─────────────────────────────────────────────────────────────
Fine-tunes Qwen2.5-Coder-7B-Instruct for RP Text2Cypher.

Why Qwen2.5-Coder over Gemma 2 9B:
  - Matches the production Ollama model (text2cypher is built on qwen2.5-coder:7b)
  - Specifically trained on code/query generation
  - Smaller (7B vs 9B) → faster training and inference
  - Better at structured output without hallucinating property names

Why this version will converge (fixes vs v1-v4):
  1. Correct base model — Qwen2.5-Coder, same as Ollama production
  2. Correct property names — exactly from data_catalog.yaml and Modelfile.jp
     (previous versions used wrong names like c.amount, c.agent, iv.amount_collected)
  3. DataCollatorForCompletionOnlyLM — loss computed only on Cypher answer tokens
     (previous versions computed loss on full 2000+ token prompt → loss=20)
  4. System prompt fits in context — 2324 tokens, 1722 headroom for Q+Cypher
  5. Qwen ChatML format — <|im_start|>user / <|im_start|>assistant

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

    # Full training (~25 min on RTX 4500):
    python scripts/train_lora.py \\
        --train-data data/splits/train.parquet \\
        --val-data   data/splits/val.parquet \\
        --output-dir ./lora_adapter_v3
"""

import unsloth  # noqa: F401 — must be FIRST

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# DataCollatorForCompletionOnlyLM moved between trl and transformers across versions
try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    try:
        from transformers import DataCollatorForCompletionOnlyLM
    except ImportError:
        # Implement a minimal version inline if neither works
        from transformers import DataCollatorForLanguageModeling
        import torch
        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            def __init__(self, response_template, tokenizer, *args, **kwargs):
                super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
                self.response_template = response_template
                self.response_token_ids = response_template

            def torch_call(self, examples):
                batch = super().torch_call(examples)
                # Mask prompt tokens — set labels to -100 for everything before response
                for i, label in enumerate(batch["labels"]):
                    # Find response template in input_ids
                    input_ids = batch["input_ids"][i].tolist()
                    tmpl = self.response_token_ids
                    for j in range(len(input_ids) - len(tmpl) + 1):
                        if input_ids[j:j+len(tmpl)] == tmpl:
                            batch["labels"][i, :j+len(tmpl)] = -100
                            break
                    else:
                        # Template not found — mask everything (skip this example)
                        batch["labels"][i] = torch.full_like(label, -100)
                return batch

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL  — Qwen2.5-Coder-7B-Instruct (matches Ollama production model)
# ─────────────────────────────────────────────────────────────────────────────

LORA_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Qwen ChatML response template — loss computed only on tokens after this
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Aligned with Modelfile.jp — uses EXACT same property names.
# Compact version (~2324 tokens) to fit in 4096 context with headroom.
# MUST be identical in train_lora.py, eval_runner.py, demo_ollama.py.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Neo4j Cypher generator for the Radiology Partners (RP) knowledge graph.
Output ONLY raw Cypher. No markdown. No explanations. ALWAYS include LIMIT. Alias every property.

Labels: Patient, Visit, Charge, Transaction, Statement, RCCall, IVRInbound,
        DiallerCall, PhoneBridge, Campaign, Location, InsurancePlan,
        Practice, DiagnosisCode, ProcedureCode, BirdeyeReview

EXACT properties (use ONLY these names):
Patient:      patient_id, source_db, gender, state, city, zip, dob
              payor_cohort, call_tier, propensity_grade
              is_self_pay, is_friction, is_tennessee, is_bai, is_fully_covered
              outstanding_balance, total_charged, total_paid, adj_bad_debt
              has_insurance, carrier_name, plan_name, plan_type
Charge:       charge_id, charge_amount, balance, line_status, dos_aging_bucket
              procedure_modality, procedure_code, is_voided, is_hold
              current_responsible_level, service_date, post_date, source_db
Transaction:  payment_id, transaction_type, paysource, payment_method
              payment_amount, adjustment_amount, adjustment_bucket
              denial_code, denial_note, post_date
RCCall:       rccallId, agent_name, team_name, skill_name, campaign_name
              sla, agent_time, total_time, in_queue, acw_time, disp_name
              rc_attributable, start_date
IVRInbound:   response_id, ivr_type, amount_paid, balance, result_desc
              call_datetime, call_duration, auth_success
DiallerCall:  account, result_desc, patient_balance, service_loc, call_datetime
Statement:    statement_id, statement_level, patient_balance, total_balance
              is_on_hold, is_released, text_successful, email_successful
Location:     location_id, name, city, state, zip, source_db
              birdeye_avg_rating, birdeye_review_count, birdeye_one_star_pct
PhoneBridge:  phone_type, primary_campaign, rc_call_count, campaign_count
InsurancePlan: plan_name, carrier_name, plan_type, plan_number, source_db
Visit:        visit_id, source_db, admit_date, discharge_date
              location_id, primary_insurance_plan
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
(BirdeyeReview)-[:REVIEWS]->(Location)  (Campaign)-[:RUN_BY]->(Practice)

"""


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GATE
# ─────────────────────────────────────────────────────────────────────────────

WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.I)

# Wrong property names that were in old training data — catch and reject
WRONG_PROPS = re.compile(
    r"\b(c\.amount|c\.agent|c\.call_type|c\.duration|c\.successful_calls"
    r"|c\.failed_calls|iv\.amount_collected|rc\.agent\b"
    r"|c\.service_date|c\.successful|l\.code)\b",
    re.I
)

def check_pair(question: str, cypher: str) -> tuple[bool, str]:
    if not question or not question.strip(): return False, "empty question"
    if not cypher  or not cypher.strip():   return False, "empty cypher"
    u = cypher.upper()
    if "MATCH"  not in u: return False, "no MATCH"
    if "RETURN" not in u: return False, "no RETURN"
    if "LIMIT"  not in u: return False, "no LIMIT"
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
# FORMATTING — Qwen ChatML
# ─────────────────────────────────────────────────────────────────────────────

def make_formatting_func(tokenizer):
    """
    Qwen ChatML format:
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
        description="LoRA fine-tuning — Qwen2.5-Coder-7B for RP Text2Cypher"
    )
    parser.add_argument("--train-data",     default="data/splits/train.parquet")
    parser.add_argument("--val-data",       default="data/splits/val.parquet")
    parser.add_argument("--num-epochs",     type=int,   default=3)
    parser.add_argument("--output-dir",     default="./lora_adapter_v3")
    parser.add_argument("--batch-size",     type=int,   default=2)
    parser.add_argument("--eval-steps",     type=int,   default=50)
    parser.add_argument("--save-steps",     type=int,   default=100)
    parser.add_argument("--max-seq-length", type=int,   default=4096)
    parser.add_argument("--learning-rate",  type=float, default=None)
    parser.add_argument("--smoke-test",     action="store_true",
                        help="Run on 50 rows to verify pipeline (2 min)")
    parser.add_argument("--skip-gguf",      action="store_true")
    args = parser.parse_args()

    lr = args.learning_rate or settings.LEARNING_RATE

    print("=" * 70)
    print("  LoRA Fine-Tuning — Qwen2.5-Coder-7B  (Text2Cypher v5)")
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

    # ── 2. Load Qwen2.5-Coder model ───────────────────────────────────────────
    print(f"\n📥 Loading {LORA_BASE_MODEL} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = LORA_BASE_MODEL,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = True,
    )
    # Use Qwen chat template
    tokenizer = get_chat_template(tokenizer, "qwen-2.5")
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
        print(f"   ✗ WARNING: too long ({tok_count}). Increase --max-seq-length.")

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

    print(f"   ✓ Format verified — Qwen ChatML with Cypher in assistant turn")

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
            print(f"   # Update Modelfile.jp FROM line:")
            print(f"   # FROM {settings.GGUF_EXPORT_DIR}/model-unsloth.Q4_K_M.gguf")
            print(f"   ollama create text2cypher-ft -f Modelfile.jp")
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