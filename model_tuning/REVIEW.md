# Code Review: `synthea-neo4j/scripts/`

## Overview

Fifteen Python scripts (~4,500 LOC) forming a Text2Cypher LoRA training pipeline: sample a live Neo4j graph → generate question/Cypher pairs → validate against the DB → fine-tune Qwen2.5-Coder-7B → export to Ollama → evaluate. The pipeline is functional but has several correctness, security, and hygiene issues that should be addressed before further growth.

## Critical Issues

### 1. Chat template mismatch will silently corrupt inference — HIGH
[train_lora.py:313](train_lora.py#L313) trains with the **Qwen 2.5** chat template (correct — the base model is qwen2.5-coder:7b, as documented on [train_lora.py:8](train_lora.py#L8)). But both [infer.py:69](infer.py#L69) and [quick_check.py:81](quick_check.py#L81) call `get_chat_template(tokenizer, "gemma")`. Applying the wrong template to the adapter will produce degraded / malformed output without any error. Both files should use `"qwen-2.5"`.

### 2. Hardcoded Neo4j password in source — HIGH (security)
[neo4j_catalog.py:13](neo4j_catalog.py#L13) contains `NEO4J_PASSWORD = "rp_strong_pass_2025"`. If this repo is ever pushed to a shared remote, the credential is exposed. Move to `os.environ["NEO4J_PASSWORD"]` and add CLI flags matching the other scripts.

### 3. ~20 GB of build artifacts under `scripts/` — HIGH
- [gguf_export_v3/](gguf_export_v3/) — ~15 GB of merged HF safetensors (4× shards).
- [gguf_export_v3_gguf/](gguf_export_v3_gguf/) — 4.6 GB Q4_K_M GGUF.
- [unsloth_compiled_cache/](unsloth_compiled_cache/) — compiled kernels, environment-specific.
- [__pycache__/](__pycache__/).

These are build outputs, not source. `.gitignore` them and regenerate on demand.

### 4. `create_eval_set.py` is a 0-byte file
[create_eval_set.py](create_eval_set.py) — delete it, or actually implement it. Committed empty files are a maintenance smell.

## Structural / Duplication Issues

- **System prompt is duplicated across 5 files** ([demo_inference.py:49-90](demo_inference.py#L49-L90), [demo_ollama.py](demo_ollama.py), [train_lora.py:112-160](train_lora.py#L112-L160), [eval_runner.py](eval_runner.py), [infer.py:32-54](infer.py#L32-L54)). Any schema change requires editing all five and getting them exactly right. Move to a shared `prompts.py` (or generate from `catalog.yaml`).
- **Schema constants (`VALID_LABELS`, `VALID_RELS`) duplicated** across [diagnose.py](diagnose.py), [validate_pairs.py](validate_pairs.py), and [build_dataset.py](build_dataset.py). Extract to a shared module.
- **`demo_ollama.py` largely overlaps `demo_inference.py`.** [demo_ollama.py](demo_ollama.py) is a thin subset of [demo_inference.py](demo_inference.py) (Ollama mode only). Fold it into one entry point with `--mode ollama`.
- **[settings.py](settings.py) is 31 lines and reads like a leftover patch** — old/new constants that appear identical. Either finish the migration or delete the file.
- **[generate_pairs_from_neo4j.py](generate_pairs_from_neo4j.py) is 1,362 lines** with ~1,100 lines being category question templates. This should be a YAML/JSON config consumed by a small generator, not code. Adding a new category currently means touching a monolith.

## Code Quality Notes

- **[eval_runner.py:43-62](eval_runner.py#L43-L62) `_call_guardrail`** accepts two different return shapes (object *and* tuple). Fragile — enforce one contract at the boundary.
- **[diagnose_guardrail.py:14](diagnose_guardrail.py#L14)** hardcodes `eval_results/latest.json` and has no `argparse`. Add `--input` for consistency with sibling scripts.
- **[diagnose.py:108](diagnose.py#L108)** truncates exception messages to 80 chars — loses debugging context. Log full trace to a file, print truncated to stdout.
- **[export_to_ollama.py:33-59](export_to_ollama.py#L33-L59)** has nested try/except fallback paths for GGUF export. The `ollama create` subprocess doesn't capture stderr — silent failures likely.
- **[build_dataset.py](build_dataset.py) lines ~120-125** concatenates pandas frames in a loop for stratified splits — inefficient. Use `groupby` + `train_test_split` per group.
- **[validate_pairs.py](validate_pairs.py) `validate_pair` vs `validate_schema_only`** return different shapes (`(bool, dict)` vs `(DataFrame, stats)`). Normalize.

## Test Coverage

None visible in this directory. For a pipeline that produces training data + a model, at minimum I'd expect:
- Unit tests for the guardrail (currently only exercised via full eval runs).
- A fixture-based test for `validate_pairs` covering schema violations, EXPLAIN failures, execution errors, and zero-row cases.
- A regression pair or two locked into `create_eval_set.py` so the empty file finally earns its name.

## Positive Notes

- Clear cascading validation model in [validate_pairs.py](validate_pairs.py) — schema → EXPLAIN → execute → guardrail is the right shape.
- Row redaction before returning DB output ([eval_runner.py](eval_runner.py), [infer.py](infer.py)) is defensive and appropriate for PHI-adjacent data.
- `WRONG_PROPS` regex in [train_lora.py:170-175](train_lora.py#L170-L175) catches known bad property names in training data — good pre-flight check.
- `EXPLAIN`-before-execute pattern is used consistently.

## Priority for Fixes

1. Fix the Gemma→Qwen chat template bug in `infer.py` and `quick_check.py` (correctness).
2. Remove the hardcoded password from `neo4j_catalog.py` (security).
3. `.gitignore` the three artifact directories (~20 GB) and `__pycache__/`.
4. Delete or implement `create_eval_set.py`.
5. Consolidate the duplicated system prompt into one module.
6. Then: merge `demo_ollama.py` into `demo_inference.py`, move question templates out of code, add a small test suite.
