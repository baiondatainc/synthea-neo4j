# Neo4j schema migration notes

This document captures the required changes to align the application with the current insurance-policy graph in `data_catalog.yaml`.

## Scope

The graph has moved from the legacy healthcare model to a policy model based on nodes such as:

- Policy
- Policyholder
- Claim
- Vehicle
- Product
- DistributionChannel
- HealthMember
- Insurer
- PreAuthorization
- Reinsurer

The core relationship vocabulary now includes:

- UNDER_POLICY
- HAS_PRODUCT
- OWNED_BY
- COVERED_BY
- HAS_PRE_AUTH
- REINSURED_BY
- FILED_BY
- INSURED_UNDER
- HAS_SERVICE_LINE
- SOLD_VIA
- SPONSORED_BY
- HAS_EVENT
- ASSESSED_BY

## High-priority files to keep aligned

### Core allowlists and runtime config

- `settings.py`
  - update `SAFE_NODES`
  - update `SAFE_RELATIONSHIPS`

### Dataset and validation logic

- `model_tuning/validate_pairs.py`
  - valid labels and valid relationships must match the catalog

- `model_tuning/build_dataset.py`
  - fallback known labels and relation names must match the catalog

- `model_tuning/infer.py`
  - system prompt and schema summary must describe the correct policy graph

### Runtime query surfaces

- `main.py`
  - graph stats queries
  - financial summary query
  - vectorization fallback / commands

- `api/websocket_server.py`
  - dashboard count queries
  - sample question set

- `qa/summarizer.py`
  - summary intent detection
  - ID field mapping
  - fetchers for Policy / Claim / Policyholder

## Important migration rules

1. Do not trust legacy healthcare labels such as `Patient`, `Visit`, `Charge`, `Location`, or `Practice` when the catalog says otherwise.
2. Keep catalogs authoritative; regenerate from Neo4j whenever the graph changes.
3. Any model prompt or schema validation should describe the policy graph, not the old healthcare graph.
4. Any aggregate stats or QA summary must use the current node and relationship names or fail safely.
5. Vector embedding and semantic retrieval features should be rebuilt against the new graph model instead of reusing the old `Patient` pipeline.

## Remaining caution areas

These files still contain old-domain vocabulary and should be reviewed in a follow-up pass if the project needs complete semantic parity beyond the core constraints already aligned here:

- `qa/aggregate_summarizer.py`
- `qa/cypher_autofix.py`
- `qa/hybrid_retriever.py`
- `semantic/embeddings.py`
- `semantic/clustering.py`
- `api/openai_compat.py`
- `metadata/catalog.py`
- various test fixtures and sample prompts in `tests/` and `docs/`

## Verification status

The edited schema-alignment files were syntax-checked successfully with:

```bash
python -m compileall main.py settings.py api/websocket_server.py qa/summarizer.py model_tuning/build_dataset.py model_tuning/infer.py model_tuning/validate_pairs.py
```

This completed without syntax errors.
