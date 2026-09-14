# IA text2cypher regression suite

Two layers:

| layer | tool | what it proves | questions |
|---|---|---|---|
| API | pytest (`test_text2cypher.py`) | the generated Cypher returns the **correct answer** | all 64 |
| UI  | Playwright (`e2e/smoke.spec.ts`) | the chat interface works end-to-end | 10 + 1 multi-turn |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env && edit          # API URL, Neo4j creds, UI selectors
export $(grep -v '^#' .env | xargs)
```

## 1. Validate the questions file first (no chat needed)
```bash
pytest --no-api
```
Fails if any ground-truth Cypher does not execute - fix `questions.yaml`, not the model.

## 2. Full API run
```bash
pytest -v                    # everything
pytest --category fraud      # subset
pytest --ids Q01,Q34         # specific
pytest -n 4                  # parallel
```
Each run writes `results/run-<stamp>.jsonl` (every generated Cypher, rows, answer)
and `results/run-<stamp>.md` (pass/fail by category + per-question table).

## 3. UI smoke
```bash
cd e2e && npm i && npx playwright install chromium
npx playwright test
npx playwright show-report
```

## Adapting to your backend
* `harness.ask_chat()` - one function, one POST. Change the payload/fields there.
  If your API returns only `cypher`, the harness executes it on Neo4j itself.
* Response field names: `T2C_FIELD_CYPHER`, `T2C_FIELD_ROWS`, `T2C_FIELD_ANSWER`.
* UI selectors: `CHAT_INPUT_SELECTOR`, `CHAT_SEND_SELECTOR`, `CHAT_RESPONSE_SELECTOR`.

## Before the first real run
1. `values:` in `questions.yaml` - align enum strings (`active`, `open`, `broker`...) with
   the synthetic generator vocabulary / profiled value sets.
2. `fraud_prob_high` and `loss_ratio_full` - percent vs ratio depends on how you load them.
3. Q16 / Q64 are date-relative: generate synthetic dates relative to the run date.
4. Q34 needs policy `CTM5DQ3K` to exist - fix the generator seed.
5. Q31 must return 0 - if it doesn't, the ingestion (not the chat) is broken.
