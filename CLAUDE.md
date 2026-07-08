# CLAUDE.md — Text2Cypher LoRA Build
## RP Knowledge Graph · SGS

> **Read this file first before touching any code.**
> This is the single source of truth for Claude Code on this project.
> Do not invent architecture. Do not deviate from the conventions below without a comment explaining why.

---

## 0. Project in one sentence

Fine-tune `Qwen2.5-Coder-7B-Instruct` with LoRA to generate Neo4j Cypher queries from
natural-language questions, export to GGUF, serve via Ollama, and guarantee that no invalid
or unsafe Cypher ever reaches the user or database.

---

## 1. Repository layout (build this if it doesn't exist)

```
.
├── CLAUDE.md                        ← this file
├── catalog.yaml                     ← RP schema (source of truth for all prompts)
├── Modelfile.jp                     ← Ollama model definition (Qwen ChatML stop tokens)
├── settings.py                      ← cypher_model, invalidate_chain_cache(), thresholds
│
├── data/
│   ├── eval/
│   │   └── eval_20260703_150054.json  ← FROZEN eval — never modify, never train on
│   ├── raw/                           ← teacher-generated pairs before validation
│   ├── validated/                     ← pairs that passed the full validation cascade
│   │   └── validated_pairs.parquet    ← columns: question, cypher, category, difficulty,
│   │                                  ←          syntax_error, timeout, returns_results,
│   │                                  ←          validated_zero_row
│   └── splits/
│       ├── train.parquet
│       └── val.parquet
│
├── scripts/
│   ├── generate_questions.py    ← Step A2: teacher-model question generation
│   ├── generate_cypher.py       ← Step A3: teacher-model Cypher generation
│   ├── validate_pairs.py        ← Step A4: full validation cascade
│   ├── build_dataset.py         ← Step A5: balance + split + freeze
│   ├── train_lora.py            ← Step A7: Unsloth LoRA fine-tune (THE main script)
│   ├── export_gguf.py           ← Step 7: merge adapter → GGUF q4_k_m
│   ├── eval_runner.py           ← Step D1: frozen eval against any Ollama model tag
│   └── pii_suite.py             ← Step D2: adversarial PII/PHI acceptance tests
│
├── runtime/
│   ├── guardrails.py            ← check_input, check_cypher, GuardedNeo4jGraph
│   ├── redaction.py             ← redact_rows, redact_text
│   ├── retry_loop.py            ← EXPLAIN → self-correct → fallback  (Part B)
│   ├── fewshot.py               ← get_fewshot_prompt (cypher_enhance)
│   └── rewrite.py               ← rewrite_question, session/focus stores
│
├── lora_checkpoints/            ← trainer saves here (gitignore the weights)
├── lora_adapter_v1/             ← saved LoRA adapter
├── gguf_export_v1/              ← merged GGUF before Ollama import
└── models/                      ← tagged production GGUFs for rollback
    └── rp-cypher-YYYYMMDD-genXXpct.gguf
```

---

## 2. Baseline numbers (never regress below these)

| Metric | Current | Minimum to ship |
|---|---|---|
| Generation rate | 86.9 % | > 86.9 % |
| Guardrail-pass rate | 86.9 % | > 86.9 % |
| Executable rate | ~100 % | ≥ 99 % |
| PII adversarial suite | — | 100 % blocked/redacted |
| Weak-5 category floor | see below | no category may regress |

**Weak-5 category current scores (absolute floor):**

| Category | Score | Floor |
|---|---|---|
| Contact Center Operations | 10/15 | ≥ 10 |
| Provider & Referral Patterns | 10/15 | ≥ 10 |
| Executive / KPI Dashboard | 12/16 | ≥ 12 |
| Trend & Temporal Analysis | 12/16 | ≥ 12 |
| Context Chain | 29/35 | ≥ 29 |

---

## 3. Hard rules — never break these

1. **No query reaches the user or DB until it has passed EXPLAIN + guardrail + execution + redaction.** The worst user-visible outcome is a graceful decline, never broken Cypher or raw PII.
2. **The QA LLM sees only redacted rows.** `redact_rows` runs inside `GuardedNeo4jGraph.query` before rows go anywhere.
3. **Teacher-model API calls send schema only — never patient rows.** Enforce in code, enforce in review.
4. **Train in Qwen ChatML format only.** Not Alpaca. Not plain text. Mismatch silently kills fine-tune benefit.
5. **Temperature 0, top_k 1 at inference.** Cypher generation must be deterministic.
6. **Feedback/training capture stores question + Cypher only — never result rows.** The flywheel dataset must be PHI-free by construction.
7. **Never promote a model without running the frozen eval AND the PII suite.** No exceptions.

---

## 4. Build phases — work in this order

### Phase 0 — Baseline & freeze *(do this before anything else)*

- [ ] Confirm `eval_20260703_150054.json` is read-only and excluded from all training pipelines
- [ ] Confirm `redact_rows` and `redact_text` are wired and tested
- [ ] Confirm `GuardedNeo4jGraph.query` calls `redact_rows` before returning
- [ ] Write `pii_suite.py` — adversarial questions designed to extract bulk names/phones/emails; all must be blocked or fully redacted
- [ ] Run `eval_runner.py` against the current Ollama model to establish the reproducible baseline

### Phase 1 — Runtime correctness loop *(highest ROI, no training needed)*

Wire `runtime/retry_loop.py`. The loop must do exactly this, in order:

```
generate_cypher(question, schema, fewshot_exemplars)
    │
    ▼
EXPLAIN in Neo4j  ──fail──▶  feed (original_question + schema + failed_cypher + neo4j_error_text)
    │                         back to LLM → regenerate  (max 2 retries total)
    │pass
    ▼
check_cypher (guardrail)  ──fail──▶  retry or fallback
    │pass
    ▼
execute with LIMIT + timeout  ──error──▶  retry or fallback
    │rows
    ▼
redact_rows(rows)
    │
    ▼
return redacted rows
         │
         exhausted all retries ──▶ SAFE FALLBACK:
             return controlled message, log failure, never show raw error
```

**Self-correct prompt must include all four:**
`original_question` + `schema` + `failed_cypher` + `neo4j_error_text`
Missing any one degrades recovery significantly.

### Phase 2 — Targeted dataset (weak-5 categories first)

Target: **2,000–3,000 validated pairs** concentrated in the weak-5, before scaling to 10k.
Let the eval decide when you've generated enough — do not generate volume ahead of measurement.

**Dataset budget allocation (for the full 10k target):**

| Bucket | Share | Count |
|---|---|---|
| Weak-5 categories | 45 % | ~4,500 |
| Other medium/hard multi-hop | 30 % | ~3,000 |
| Simple retrieval & aggregation | 15 % | ~1,500 |
| Edge cases & known traps | 10 % | ~1,000 |

**Difficulty mix:** 25 % Easy / 40 % Medium / 35 % Hard

**Known traps to seed explicitly in generation prompts:**
- Provider info is a property on Visit/Charge — there is NO `Provider` node
- Quarter math: `datetime()` must wrap ISO strings; quarter boundaries are NOT implicit
- PhoneBridge multi-hop: `patient → PhoneBridge → phone_norm → ringcentral.ANI_DIALNUM_norm`
- SETTLES relationship direction
- Context Chain: follow-up questions must resolve entities from the previous turn

### Phase 3 — First LoRA (train on weak-5 dataset)

Run `scripts/train_lora.py`. See Section 5 for exact parameters.
After training: run `eval_runner.py` + `pii_suite.py`. Promote only if gates pass.

### Phase 4 — Scale to 10k

Fill remaining buckets. Retrain. Target: ≥ 95 % generation rate, ≥ 95 % guardrail-pass, no regression.

### Phase 5 — Feedback flywheel

Wire per-query capture (question + cypher + executed_ok + returned_rows + rating, **no result rows**).
Weekly: curate → append validated pairs → retrain LoRA → eval + PII gate → promote only if better.

---

## 5. LoRA training — exact specification

### 5.1 Environment

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets transformers sentencepiece
```

### 5.2 Model load

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length = 2048,
    dtype          = None,       # auto: bfloat16 on A100, float16 on T4
    load_in_4bit   = True,
)
```

### 5.3 LoRA adapter — exact parameters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r               = 16,
    lora_alpha      = 16,
    target_modules  = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP — do NOT omit these
    ],
    lora_dropout    = 0.05,
    bias            = "none",
    use_rslora      = True,                         # rsLoRA scaling
    use_gradient_checkpointing = "unsloth",         # required on T4 to avoid OOM
)
```

Expected output: ~20 M trainable / ~7 B total (~0.3 %).

### 5.4 Dataset format — Qwen ChatML (critical)

**This is the #1 migration error if wrong. Not Alpaca. Not plain text. Qwen ChatML only.**

```python
SYSTEM_PROMPT = """You are a Cypher query generator for the RP knowledge graph.
[paste full catalog.yaml schema here]
Rules:
- Output raw Cypher only. No markdown fences. No explanation.
- One MATCH statement. Alias every aggregation. LIMIT on every query.
- Use datetime() to wrap ISO date strings.
- No SQL keywords. No UNION unless essential.
- Read-only: no CREATE, MERGE, SET, DELETE, REMOVE."""

def format_example(row, tokenizer):
    return tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["question"]},
            {"role": "assistant", "content": row["cypher"]},
        ],
        tokenize=False,
        add_generation_prompt=False,   # False for training — answer is included
    )
```

### 5.5 Training arguments

```python
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    eval_dataset       = val_dataset,
    dataset_text_field = "text",
    max_seq_length     = 2048,
    packing            = True,
    args = TrainingArguments(
        output_dir                  = "./lora_checkpoints",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,        # effective batch = 16
        warmup_ratio                = 0.05,
        num_train_epochs            = 3,
        learning_rate               = 2e-4,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 25,
        evaluation_strategy         = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 200,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        report_to                   = "none",
    ),
)
```

**Healthy training signal:** loss drops from ~2.0 → ~0.3–0.6 by epoch 3.
If loss plateaus above 1.0 after epoch 1 — stop, check dataset format first.

### 5.6 Export

```python
# Save adapter (fast, ~80 MB)
model.save_pretrained("./lora_adapter_v1")
tokenizer.save_pretrained("./lora_adapter_v1")

# Export GGUF (merges adapter + quantizes)
model.save_pretrained_gguf(
    "gguf_export_v1",
    tokenizer,
    quantization_method = "q4_k_m",    # must match Modelfile.jp
)
# output: gguf_export_v1/model-unsloth.Q4_K_M.gguf
```

### 5.7 Ollama registration

```bash
# Update FROM line in Modelfile.jp:
# FROM ./gguf_export_v1/model-unsloth.Q4_K_M.gguf

ollama create rp-cypher-v1 -f Modelfile.jp
ollama list    # confirm tag appears
```

---

## 6. Validation cascade — every pair must pass all 5 steps

Run `scripts/validate_pairs.py` against a **non-production Neo4j copy**.

| Step | Action | On failure |
|---|---|---|
| 1. EXPLAIN | Parse + schema check | Drop the pair |
| 2. Execute | Row cap + timeout | Drop the pair |
| 3. Zero-row check | Returns 0 results | → review queue (see note) |
| 4. Guardrail | Run through `check_cypher` | Drop the pair |
| 5. LLM-as-judge (optional) | Teacher confirms semantic match | Flag for human review |

**Zero-row decision rule** (the plan's gap, filled here):
- If the question is verifiably answerable from schema AND returns 0 rows → likely wrong, discard.
- If it's a valid aggregation/filter over genuinely sparse synthetic data → mark `validated_zero_row=true` and include. The model needs this pattern.

**Expected yield:** 50–70 % of generated pairs survive. To land 10k clean, generate 15–18k raw.
The yield rate is the quality signal — do not skip validation to inflate numbers.

---

## 7. Promotion gate — ship only if all pass

Run after every training run before flipping `settings.cypher_model`:

```bash
python scripts/eval_runner.py \
    --model  rp-cypher-v1 \
    --eval   data/eval/eval_20260703_150054.json \
    --out    eval_results/$(date +%Y%m%d).json

python scripts/pii_suite.py --model rp-cypher-v1
```

**Checklist — all must be true to promote:**
- [ ] generation rate > 86.9 %
- [ ] guardrail-pass rate > 86.9 %
- [ ] executable rate ≥ 99 %
- [ ] no weak-5 category below its floor (Section 2)
- [ ] PII suite: 100 % blocked or fully redacted

If any check fails: hold, diagnose, do not promote.

On promotion:
```python
settings.cypher_model = "rp-cypher-v1"
invalidate_chain_cache()
```

Tag the GGUF for rollback:
```bash
cp gguf_export_v1/model-unsloth.Q4_K_M.gguf \
   models/rp-cypher-$(date +%Y%m%d)-gen$(SCORE)pct.gguf
```

---

## 8. PII/PHI controls — what must be wired

| Layer | Control | Where in code |
|---|---|---|
| C1 | Serve on-prem via Ollama — no patient data to external APIs | Ollama + `Modelfile.jp` |
| C2 | Schema-only prompts — no patient rows in LLM input | `generate_questions.py`, `generate_cypher.py` |
| C3 | Row cap + read-only guardrail on every query | `check_cypher`, `GuardedNeo4jGraph` |
| C3 | Minimum-necessary: block queries returning multiple PII fields without narrow filter | extend `check_cypher` |
| C4 | `redact_rows` runs before rows leave the data layer | `GuardedNeo4jGraph.query` |
| C4 | `redact_text` runs on QA model output | output pipeline |
| C5 | Role-scoped un-redaction after model, gated by auth — never by LLM | auth middleware |
| C6 | Per-query audit log: user, question, Cypher, row_count, pii_returned, timestamp | `GuardedNeo4jGraph.query` |
| C6/C7 | Feedback capture: question + cypher only, never result rows | capture endpoint |

**Redaction token format:**
`[name-redacted]`, `[phone-redacted]`, `[dob-redacted]`, `[email-redacted]`, `[id-redacted]`

---

## 9. Semantic evaluation metric — important correction

Do **not** use Jaro-Winkler or string similarity as the primary semantic metric.
Two semantically identical Cypher queries can look very different textually.

**Primary semantic metric:** execution-result comparison on the test graph.
Same rows returned (order-insensitive, after sorting) = correct.

String/AST similarity = secondary signal only.

---

## 10. Key files in the existing codebase

| Plan reference | Existing file |
|---|---|
| `check_input` | `runtime/guardrails.py` |
| `check_cypher`, `GuardedNeo4jGraph` | `runtime/guardrails.py` |
| `redact_rows`, `redact_text`, `guardrails_redact_output` | `runtime/redaction.py` |
| `get_fewshot_prompt` | `runtime/fewshot.py` (`cypher_enhance`) |
| `rewrite_question`, session/focus stores | `runtime/rewrite.py` |
| Ollama serving | `Modelfile.jp` (Qwen ChatML stop tokens) |
| Model swap | `settings.cypher_model` + `invalidate_chain_cache()` |
| Schema / label friendly-names | `catalog.yaml` |

Do not rewrite these unless a specific bug requires it. Extend, don't replace.

---

## 11. What NOT to do

- Do not generate 10k pairs before running the eval on 2–3k. Volume ahead of measurement is waste.
- Do not use Alpaca, ShareGPT, or any non-ChatML format for training data.
- Do not include `add_generation_prompt=True` in training formatting (only use it at inference time).
- Do not skip the validation cascade to save time. Unvalidated data trains bad patterns.
- Do not show the user raw Neo4j error messages. Always use the safe fallback message.
- Do not let the QA model see un-redacted rows for any reason.
- Do not promote a model without the frozen eval + PII gate. No exceptions for time pressure.
- Do not omit `gate_proj`, `up_proj`, `down_proj` from `target_modules`. MLP layers matter for structured output.
- Do not store result rows in the feedback capture, audit logs, or training data.
