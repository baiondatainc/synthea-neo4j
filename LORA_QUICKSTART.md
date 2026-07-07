# 🚀 LoRA Training Quick Start Guide
## Text2Cypher + Gemma 2 9B for RP Knowledge Graph

This guide walks you through training the model with existing code and data.

---

## ✅ What's Included (Already Created)

Your scripts directory now has:

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `train_lora.py` | **Main LoRA trainer** | `data/splits/train.parquet` | `lora_adapter_v2/` + GGUF |
| `validate_pairs.py` | Validates raw pairs (5-step cascade) | `data/raw/pairs.parquet` | `data/validated/validated_pairs.parquet` |
| `build_dataset.py` | Creates train/val/test splits | `data/validated/validated_pairs.parquet` | `data/splits/{train,val,test}.parquet` |
| `eval_runner.py` | Evaluates on frozen eval set | `data/eval/eval.json` | `eval_results/v1.json` + metrics |
| `infer.py` | Quick inference + execution | Question string | Cypher + results |

---

## 📊 Your Current Data Structure

```
data/
├── eval/          ← Empty (add your frozen eval set here)
├── raw/           ← Empty (add generated pairs here)
├── validated/     ← Will be populated by validate_pairs.py
├── splits/        ← Will be populated by build_dataset.py
└── synthea/       ← Existing synthetic data (optional)
```

---

## 🎯 Three Training Scenarios

### Scenario 1: You Have Raw Question→Cypher Pairs

If you've already generated question→Cypher pairs (e.g., using a teacher model):

```bash
# Step 1: Validate all pairs
python scripts/validate_pairs.py \
    --input data/raw/your_generated_pairs.parquet \
    --output data/validated/validated_pairs.parquet \
    --uri neo4j://YOUR_URI \
    --user neo4j \
    --password YOUR_PASSWORD \
    --build-splits

# Step 2: Train LoRA
python scripts/train_lora.py \
    --train-data data/splits/train.parquet \
    --val-data data/splits/val.parquet \
    --num-epochs 3 \
    --output-dir ./lora_adapter_v1

# Step 3: Evaluate
python scripts/eval_runner.py \
    --model ./lora_adapter_v1 \
    --eval data/eval/eval.json \
    --out eval_results/v1.json
```

**Expected time:**
- Validation: 10-30 min (depends on pair count)
- Training: 2-4 hours (A100) or 6-12 hours (T4)
- Evaluation: 10-20 min

---

### Scenario 2: You Need to Generate Pairs First

If you need to generate question→Cypher pairs from schema:

```bash
# Step 0: Generate pairs using your teacher model
# (You'll need to implement this; template in CLAUDE.md)
# Expected output: data/raw/generated_pairs.parquet

# Then follow Scenario 1 from Step 1 onwards
```

---

### Scenario 3: You Have Small Dataset and Want Quick Iteration

```bash
# Skip validation for testing (NOT recommended for production)
python scripts/build_dataset.py \
    --input data/raw/your_pairs.parquet \
    --output data/splits/

# Train with smaller epochs
python scripts/train_lora.py \
    --train-data data/splits/train.parquet \
    --val-data data/splits/val.parquet \
    --num-epochs 1 \
    --batch-size 4

# Quick eval
python scripts/eval_runner.py \
    --model ./lora_adapter_v1 \
    --eval data/eval/eval.json \
    --limit 50
```

---

## 🔑 Key Configuration (settings.py)

Already updated for Gemma 2 9B:

```python
LORA_BASE_MODEL = "google/gemma-2-9b-it"
LORA_MODEL_PATH = "../text2cypher-gemma-2-9b-it-finetuned-2024v1"

TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3

GENERATION_TEMPERATURE = 0  # Deterministic
CYPHER_ROW_LIMIT = 100
CYPHER_TIMEOUT_SECONDS = 15
```

Change these as needed for your environment (GPU VRAM, time constraints, etc.)

---

## 🎓 Expected Training Curve

With proper data, you should see:

```
Epoch 1, Step 10:    Loss = 1.8
Epoch 1, Step 50:    Loss = 1.2
Epoch 1, Step 100:   Loss = 0.9
Epoch 2, Step 100:   Loss = 0.6
Epoch 2, Step 200:   Loss = 0.5
Epoch 3, Step 100:   Loss = 0.4  ← Good convergence
```

**Warning signs:**
- Loss stays above 1.0 after epoch 1 → Bad dataset format
- Loss explodes (NaN) → Learning rate too high, try 1e-4
- Loss doesn't move → Model frozen, check adapter loading

---

## 📈 Success Criteria (From CLAUDE.md)

After training, run evaluation and check:

```bash
python scripts/eval_runner.py --model ./lora_adapter_v1 --eval data/eval/eval.json
```

**You've succeeded if:**
- ✅ Generation rate ≥ 86.9% (produced valid Cypher)
- ✅ Guardrail-pass rate ≥ 86.9% (passed safety checks)
- ✅ Executable rate ≥ 99% (ran without error)
- ✅ No weak-5 category regressed (see CLAUDE.md)

If metrics are lower, check:
1. Training data quality (validation yield ≥ 50%)
2. Epoch count (3 is usually sufficient)
3. Learning rate (try 1e-4 if unstable)

---

## 🛠️ Testing Your Trained Model

Quick test before production deployment:

```bash
# Test generation
python scripts/infer.py \
    --model ./lora_adapter_v1 \
    --question "What is the average cost of visits by location?" \
    --execute \
    --uri neo4j://localhost:7687 \
    --user neo4j \
    --password YOUR_PASSWORD

# Should output:
# ✅ Generated Cypher
# ✅ Guardrail check passed
# ✅ Execution successful with N rows
```

---

## 📦 Deploying to Production

Once evaluation passes:

### Option A: Use Ollama (Recommended)

```bash
# 1. GGUF export (done by train_lora.py)
# Output: gguf_export_v1/model-unsloth.Q4_K_M.gguf

# 2. Create Ollama model
cat > Modelfile.jp << 'EOF'
FROM ./gguf_export_v1/model-unsloth.Q4_K_M.gguf

PARAMETER temperature 0
PARAMETER top_k 1

SYSTEM """You are a Cypher expert. Generate only valid queries. Always LIMIT 100."""
EOF

ollama create rp-cypher-prod -f Modelfile.jp

# 3. Verify
ollama list | grep rp-cypher-prod

# 4. Update settings
# settings.py: CYPHER_MODEL_TAG = "rp-cypher-prod"
```

### Option B: Direct Model Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it")
model = PeftModel.from_pretrained(base, "./lora_adapter_v1")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

# Use model.generate() in your inference pipeline
```

---

## 🐛 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| **OOM error during training** | Batch size too large | Reduce `TRAIN_BATCH_SIZE` to 1 |
| **Loss doesn't decrease** | Wrong data format (not Gemma ChatML) | Re-validate, check `format_gemma_chat()` |
| **Eval score lower than baseline** | Overfitting on small dataset | Validate more pairs, reduce epochs |
| **Guardrail-pass dropped** | Unvalidated training data included | Re-run validation cascade |
| **Model generates same query for different questions** | Temperature is 0 (correct but confusing in testing) | Expected behavior—add noise only at inference if needed |
| **GGUF export fails** | Missing unsloth[gguf] or disk space | Skip or check `pip list \| grep unsloth` |

---

## 📚 Next Steps

1. **Prepare data:**
   ```bash
   # Get your question→Cypher pairs into data/raw/
   # Ensure Neo4j is running and accessible
   ```

2. **Validate:**
   ```bash
   python scripts/validate_pairs.py --build-splits
   ```

3. **Train:**
   ```bash
   python scripts/train_lora.py
   ```

4. **Evaluate:**
   ```bash
   python scripts/eval_runner.py --model ./lora_adapter_v1
   ```

5. **Deploy:**
   ```bash
   ollama create rp-cypher-prod -f Modelfile.jp
   ```

---

## 🎯 Optimization Tips

- **Faster training:** Use smaller dataset (1-2k pairs) first, validate yield quality
- **Better accuracy:** Allocate 45% of pairs to weak-5 categories (see CLAUDE.md)
- **Lower VRAM:** Reduce batch size or max_seq_length (from 2048 to 1024)
- **Faster eval:** Use `--limit N` to sample the evaluation set

---

## 📞 Reference Files

- **CLAUDE.md** — Complete architecture (read if stuck)
- **text2cypher_train_and_implement_plan.md** — Full strategy & risks
- **settings.py** — All tunable parameters
- **catalog.yaml** — RP schema (source of truth)
- **runtime/guardrails.py** — Safety checks applied to all queries

---

## ⚡ One-Liner Quick Test

```bash
# Assuming data/splits/ exists and Neo4j is running:
python scripts/train_lora.py --num-epochs 1 && python scripts/eval_runner.py --model ./lora_adapter_v1 --limit 50
```

---

**Status:** ✅ All scripts created and configured for Gemma 2 9B + your existing codebase

**Last updated:** 2026-07-06
