# Cypher Generation Eval

Runs the 200 questions in [`docs/questions-eval.md`](../../../docs/questions-eval.md) through the Cypher LLM and records three signals per question:

| Signal | Meaning | Gates the test? |
|---|---|---|
| `cypher_generated` | LLM returned a non-empty Cypher string | **yes** — assertion fails if empty |
| `guardrail_ok` | Query passes `guardrails/cypher.py` (read-only, schema valid, no SQL) | no — recorded only |
| `executable` | Neo4j accepts `EXPLAIN <cypher>` | no — recorded only |

The pass bar is intentionally low so we can see *what's broken* in the report instead of having every test crash on the same issue.

## Run

```bash
# All 200 (needs LLM + Ollama; Neo4j optional)
uv run pytest tests/eval -v

# First 20, fast iteration
uv run pytest tests/eval --eval-limit 20 -v

# Only easy questions
uv run pytest tests/eval --eval-difficulty E -v

# Skip Neo4j EXPLAIN check (still generates + guardrails)
uv run pytest tests/eval --eval-no-execute -v
```

Requires `.env` with `NEO4J_URI`/`NEO4J_PASSWORD` (used by `config.get_settings()`), `CYPHER_MODEL`, and `OLLAMA_BASE_URL`. If Neo4j is unreachable the eval still runs — `executable_rate` is reported as "not tested".

## Reports

Each run writes to `tests/eval/reports/`:

- `eval_YYYYMMDD_HHMMSS.json` — full per-question records (cypher, guardrail reason, EXPLAIN error)
- `eval_YYYYMMDD_HHMMSS.md` — summary tables by difficulty and category, plus a failures section
- `latest.json` / `latest.md` — symlinks to the most recent run

The Markdown report calls out three failure types:
1. **Generation failures** — LLM produced empty/error output
2. **Guardrail blocks** — generated something but the guardrail rejected it (often: SQL syntax, unknown labels)
3. **Not executable** — guardrail passed but Neo4j EXPLAIN failed (often: missing properties, type mismatches)

## Override the questions file

```bash
EVAL_QUESTIONS_PATH=/some/other/file.md uv run pytest tests/eval
```

## Notes

- The eval reuses `qa.chain.get_cypher_llm()` and `qa.chain.CYPHER_GENERATION_PROMPT`, so it tracks production behavior. If you change the prompt or model, this eval reflects it on the next run.
- We bypass the answer LLM and the GraphCypherQAChain plumbing — `cypher_llm.invoke(prompt)` directly. This is ~10× faster than the full chain and isolates Cypher quality from QA-formatting quality.
- EXPLAIN validates query plan (syntax + schema reference) without executing, so it's a cheap "would Neo4j accept this?" check. It does **not** catch runtime errors like type mismatches in a comparison, missing indexes, or empty result sets — by design.
