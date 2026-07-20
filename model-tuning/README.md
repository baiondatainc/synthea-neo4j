# Text2Cypher LoRA Training Pipeline
## Gemma 2 9B Fine-tuning for RP Knowledge Graph

This directory contains scripts to train, validate, and evaluate a fine-tuned Gemma model for generating Cypher queries from natural language questions.

---

## 📋 Quick Start

### 1. **Prepare Training Data**

You need validated question→Cypher pairs. The pipeline expects:
- Input: `data/raw/generated_pairs.parquet` (from your generation process)
- Output: `data/validated/validated_pairs.parquet` + `data/splits/{train,val}.parquet`

```bash
# Validate all pairs against guardrails and Neo4j schema
python scripts/validate_pairs.py \
    --input data/raw/generated_pairs.parquet \
    --output data/validated/validated_pairs.parquet \
    --uri neo4j://localhost:7687 \
    --user neo4j \
    --password <password> \
    --build-splits

# Or just build splits from existing validated data
python scripts/build_dataset.py \
    --input data/validated/validated_pairs.parquet \
    --output data/splits/ \
    --train-split 0.8 \
    --val-split 0.1
```

### 2. **Train LoRA Adapter**

```bash
python scripts/train_lora.py \
    --train-data data/splits/train.parquet \
    --val-data data/splits/val.parquet \
    --num-epochs 3 \
    --output-dir ./lora_adapter_v2 \
    --batch-size 2
```

**Expected output:**
- `lora_adapter_v2/` — adapter weights (peft format)
- `gguf_export_v1/model-unsloth.Q4_K_M.gguf` — merged + quantized for Ollama

### 3. **Evaluate Against Frozen Eval Set**

```bash
python scripts/eval_runner.py \
    --model ./lora_adapter_v2 \
    --eval data/eval/eval.json \
    --out eval_results/v2.json \
    --uri neo4j://localhost:7687 \
    --user neo4j \
    --password <password>
```

**Success criteria** (from `CLAUDE.md`):
- Generation rate ≥ 86.9%
- Guardrail-pass rate ≥ 86.9%
- Executable rate ≥ 99%
- No weak-5 category regresses

---

## 📁 Directory Structure

```
scripts/
├── train_lora.py           ← Main training script
├── validate_pairs.py       ← Validation cascade (5 steps)
├── build_dataset.py        ← Create train/val/test splits
├── eval_runner.py          ← Frozen eval metrics
└── README.md               ← This file

data/
├── raw/                    ← Generated (un-validated) pairs
├── validated/              ← Pairs after validation cascade
│   └── validated_pairs.parquet
├── splits/                 ← Train/val/test splits
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── eval/                   ← Frozen eval set (NEVER train on this)
│   └── eval.json
└── synthea/                ← Synthetic data sources (optional)

lora_checkpoints/           ← Trainer saves checkpoints here
lora_adapter_v1/            ← Saved adapter (reusable)
gguf_export_v1/             ← GGUF export for Ollama
models/                     ← Production GGUF snapshots (for rollback)
```

---

## 🔄 Full Pipeline Workflow

### Phase 0: Setup (One-time)
```bash
# Ensure eval set is frozen and never trained on
# Verify guardrails.py and redaction.py are wired
# Check settings.py LoRA config
```

### Phase 1: Generate → Validate → Split (Iterative)

**Step 1: Generate question→Cypher pairs** (using teacher model, schema-only)
```bash
python scripts/generate_questions.py --output data/raw/questions.parquet
python scripts/generate_cypher.py --input data/raw/questions.parquet --output data/raw/pairs.parquet
```
*(These scripts are generated separately; templates provided in CLAUDE.md)*

**Step 2: Validate against guardrails** (5-step cascade)
```bash
python scripts/validate_pairs.py \
    --input data/raw/pairs.parquet \
    --output data/validated/validated_pairs.parquet \
    --build-splits
```

**Expected yield:** 50–70% (quality signal — unvalidated data is ~50% valid)

**Step 3: Build splits**
```bash
python scripts/build_dataset.py \
    --input data/validated/validated_pairs.parquet \
    --output data/splits/ \
    --show-budget
```

### Phase 2: Train & Evaluate

**Step 4: Train LoRA**
```bash
python scripts/train_lora.py \
    --train-data data/splits/train.parquet \
    --val-data data/splits/val.parquet \
    --num-epochs 3 \
    --output-dir ./lora_adapter_v2
```

**Expected training signal:**
- Loss: ~2.0 → 0.3–0.6 by epoch 3
- No divergence (loss exploding) after epoch 1

**Step 5: Evaluate & Gate**
```bash
# Run frozen eval
python scripts/eval_runner.py \
    --model ./lora_adapter_v2 \
    --eval data/eval/eval.json \
    --out eval_results/v2.json

# Run PII suite (TODO: implement pii_suite.py)
python scripts/pii_suite.py --model ./lora_adapter_v2
```

**Promotion gate — all must pass:**
- [ ] Generation rate > 86.9%
- [ ] Guardrail-pass rate > 86.9%
- [ ] Executable rate ≥ 99%
- [ ] No weak-5 category regressed
- [ ] PII suite: 100% blocked or redacted

### Phase 3: Deploy

If gates pass:
```bash
# Tag model for rollback
cp gguf_export_v1/model-unsloth.Q4_K_M.gguf \
   models/rp-cypher-$(date +%Y%m%d)-gen$(SCORE)pct.gguf

# Register with Ollama
ollama create rp-cypher-v2 -f Modelfile.jp

# Promote in settings
# Edit settings.py: CYPHER_MODEL_TAG = "rp-cypher-v2"
# Clear cache
python -c "from settings import invalidate_chain_cache; invalidate_chain_cache()"
```

---

## 🎯 LoRA Training Configuration

See `settings.py` for all hyperparameters. Key defaults for Gemma 2 9B:

```python
# Model
LORA_BASE_MODEL = "google/gemma-2-9b-it"
LORA_MODEL_PATH = "../text2cypher-gemma-2-9b-it-finetuned-2024v1"  # If local

# Adapter
LORA_RANK = 16                      # LoRA rank
LORA_ALPHA = 32                     # Scaling
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP
]
LORA_DROPOUT = 0.05
LORA_USE_RSLORA = True              # rsLoRA scaling

# Training
TRAIN_BATCH_SIZE = 2                # Per GPU (smaller for Gemma 9B)
GRADIENT_ACCUMULATION_STEPS = 8     # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"

# Inference
GENERATION_TEMPERATURE = 0.0        # Deterministic
GENERATION_TOP_K = 1
MAX_SEQ_LENGTH = 2048
```

---

## 📊 Data Format

### Input: Validated Pairs (Parquet)

```python
{
    "question": str,              # Natural language question
    "cypher": str,                # Generated Cypher query
    "category": str,              # Optional: e.g., "Provider & Referral Patterns"
    "difficulty": str,            # Optional: "easy", "medium", "hard"
    "validated_zero_row": bool,   # Optional: True if returns 0 rows intentionally
}
```

### Training Format: Gemma Chat

Internally, each pair is formatted as:

```
<bos><start_of_turn>user
[question]<end_of_turn>
<start_of_turn>model
[cypher]<end_of_turn><eos>
```

(Handled by `format_gemma_chat()` in `train_lora.py`)

---

## ✅ Validation Cascade (5 Steps)

Every pair must pass all steps:

| Step | Action | On Fail |
|------|--------|---------|
| 1. EXPLAIN | Parse + schema validation | Drop |
| 2. Execute | Row cap + timeout | Drop |
| 3. Zero-row | Check if intentional | Flag |
| 4. Guardrail | `check_cypher()` safety | Drop |
| 5. Semantic | LLM-as-judge (optional) | Flag |

Run with:
```bash
python scripts/validate_pairs.py --input data/raw/pairs.parquet --output data/validated/validated_pairs.parquet
```

Expected yield: 50–70%

---

## 📈 Metrics

### Generation Rate
% of eval questions that produced valid Cypher (passed EXPLAIN).

### Guardrail-Pass Rate
% of generated queries that passed `check_cypher()` (read-only, schema, row cap).

### Executable Rate
% of guardrail-clean queries that ran without timeout/error.

### Returns-Rows Rate
% of executable queries that returned ≥1 row (proxy for semantic correctness).

### Per-Category Breakdown
Scores for weak-5 categories (must not regress):
- Contact Center Operations (floor: 10/15)
- Provider & Referral Patterns (floor: 10/15)
- Executive / KPI Dashboard (floor: 12/16)
- Trend & Temporal Analysis (floor: 12/16)
- Context Chain (floor: 29/35)

---

## 🔒 PII/PHI Safety

The training pipeline enforces:

1. **Schema-only prompts** — no patient data sent to LLM
2. **Validated-only dataset** — training pairs must pass guardrail
3. **Input redaction** — questions checked for PII before LLM
4. **Output guardrails** — Cypher must be read-only, schema-bounded, row-capped
5. **Feedback capture** — question + Cypher only, never result rows

Verify these controls are wired:
- `runtime/guardrails.py` — `check_input()`, `check_cypher()`
- `runtime/redaction.py` — `redact_rows()`, `redact_text()`
- `settings.py` — `PII_PATTERNS`, row limits

---

## 🚨 Common Issues

### Training loss doesn't decrease
- **Check:** Dataset format (must be Gemma chat format, not Alpaca)
- **Check:** Learning rate (2e-4 is standard; try 1e-4 if unstable)
- **Check:** Batch size (smaller if OOM: try 1)

### Generated Cypher is identical across questions
- **Check:** Temperature 0 is correct (deterministic output)
- **Check:** Model actually loaded the adapter
- **Check:** Model was properly saved after training

### Eval scores regressed after training
- **Likely:** Overfitting on small dataset
- **Fix:** Increase validation set, reduce epochs, use dropout

### Guardrail-pass rate dropped
- **Likely:** Training data included unvalidated pairs
- **Fix:** Re-run validation cascade; rebuild splits from scratch

### GGUF export fails
- **Check:** `unsloth[gguf]` installed
- **Check:** Disk space (GGUF ≈ 4-5GB)
- **Workaround:** Skip GGUF export; use adapter directly with Ollama

---

## 🔗 Integration: From Adapter to Production

### Option 1: Ollama + Adapter (Recommended)

```bash
# 1. Export GGUF from adapter (done by train_lora.py)
# Output: gguf_export_v1/model-unsloth.Q4_K_M.gguf

# 2. Create Ollama model
cat > Modelfile.jp << 'EOF'
FROM ./gguf_export_v1/model-unsloth.Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER top_k 1

SYSTEM """You are a Cypher query generator for the RP knowledge graph.
Generate only valid Cypher. No markdown. Always LIMIT 100."""
EOF

ollama create rp-cypher-v2 -f Modelfile.jp

# 3. Serve
ollama run rp-cypher-v2
```

### Option 2: Direct Model Loading (For Fine-grained Control)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# Load LoRA adapter
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora_adapter_v2")

# Generate
model.eval()
inputs = tokenizer("MATCH (p:Patient) RETURN p LIMIT 10", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0)
```

---

## 📚 References

- **CLAUDE.md** — Complete architecture & rules (read first)
- **text2cypher_train_and_implement_plan.md** — Full strategy & risk analysis
- **catalog.yaml** — RP Knowledge Graph schema (source of truth)
- **settings.py** — Guardrail thresholds and model config

---

## 🤝 Feedback Loop (Phase 5)

After production deployment, capture per-query feedback:

```python
# Capture (PHI-free)
{
    "question": "...",          # User question
    "generated_cypher": "...",  # What model generated
    "executed_ok": true,        # Did it run?
    "returned_rows": 42,        # How many?
    "rating": "good",           # User feedback: good/bad/unclear
}
```

**Weekly:**
```bash
# 1. Curate high-confidence corrections
# 2. Append to validated_pairs.parquet
# 3. Rebuild splits
# 4. Retrain LoRA
python scripts/train_lora.py ...

# 5. Evaluate + gate
python scripts/eval_runner.py ...

# 6. Promote only if better
# settings.py: CYPHER_MODEL_TAG = "..."
```

---

## ⚡ Tips for Best Results

1. **Start small:** Generate & validate 2–3k pairs first. Let eval drive volume.
2. **Weak-5 focus:** Allocate 45% to Contact Center, Provider, KPI, Temporal, Context Chain categories.
3. **Validate early:** Validation is where accuracy comes from, not raw volume.
4. **Deterministic eval:** Use same frozen eval set for all comparisons.
5. **Guard against regression:** No promotion without eval + PII gate.
6. **Feedback is gold:** User-validated corrections directly improve the model.

---

## 📞 Troubleshooting

For issues, check:
1. `test-context.log` — runtime traces
2. `lora_checkpoints/` — trainer logs (search for errors)
3. `eval_results/` — evaluation metrics over time
4. `settings.py` — verify configuration matches your setup

---

Last updated: 2026-07-06
