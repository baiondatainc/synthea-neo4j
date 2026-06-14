# HealthGraph AI — v2 design

Context memory, guardrails, hybrid retrieval, and answer caching.

Status: design — not yet implemented. Pairs with [healthgraph_architecture-v2.html](healthgraph_architecture-v2.html).

---

## 1. Module layout

```
synthea-neo4j/
  memory/
    __init__.py
    session.py            # Redis session store (LangChain RedisChatMessageHistory wrapper)
    focus.py              # Focus-entity tracker (which IDs the conversation is "on")
    rewriter.py           # Elliptical follow-up rewriter (LLM-driven)
  qa/
    chain.py              # (existing) — refactor: routes through memory + guardrails
    contextual_chain.py   # NEW — assembles transcript + focus → standalone question
    hybrid_retriever.py   # NEW — vector + graph traversal path
    router.py             # NEW — picks text2cypher vs hybrid
  guardrails/
    __init__.py
    input.py              # Topic + prompt-injection filter
    cypher.py             # Read-only check, row cap, timeout, schema validation
    output.py             # PHI/PII redaction (names, phones, emails, SSN-like)
  cache/
    __init__.py
    answer_cache.py       # Redis-backed (question_hash, context_hash) → response
  semantic/
    __init__.py
    embeddings.py         # Offline: vectorize node profiles → Neo4j vector index
    clustering.py         # Offline: behaviour cohorts (graph algorithms / k-means)
  metadata/
    catalog.py            # Friendly names for properties/coded values
    data_dictionary.yaml  # Hand-curated reference
  api/
    openai_compat.py      # (existing) — minor change: pass conversation_id through
    websocket_server.py   # (existing) — unchanged
```

---

## 2. Phasing

The phases are ordered so each one ships value standalone — you don't have to finish
the hybrid retriever to get memory + guardrails into production.

### Phase A — guardrails (1 week)

Lowest risk, highest immediate value. Drops into the existing `qa.chain` with one
wrapper per check. No new infra.

- `guardrails/cypher.py` — parse the LLM's Cypher, reject if it contains
  `CREATE|DELETE|SET|MERGE|DROP|REMOVE|CALL apoc.*write`. Inject `LIMIT 100` if
  missing. Wrap the Neo4j call with `session.run(timeout=15)`.
- `guardrails/cypher.py` — load `db.schema.visualization()` once at startup;
  reject queries that reference labels/properties not in the snapshot.
- `guardrails/output.py` — Presidio (or regex pass for v0) to mask names from
  `Patient.first_name`/`last_name`, phone numbers, emails before the answer
  hits the SSE stream.
- `guardrails/input.py` — small classifier prompt or rule list:
  *"Is the user asking about the RP knowledge graph?"* → bool. Reject if not.

**Acceptance:** existing `test_chain.py` still passes; new tests cover write
attempts, off-schema labels, PHI in answers, off-topic input.

### Phase B — memory (1 week)

`LangChain RedisChatMessageHistory` keyed by `conversation_id`. Pulled from the
`/v1/chat/completions` request — LibreChat sends `conversation_id` in the
metadata. Falls back to a hash of the first user message if absent.

- `memory/session.py` — wraps `RedisChatMessageHistory` with TTL (`SESSION_TTL=24h`)
  and a helper to dump the last N turns as a transcript.
- `memory/focus.py` — after each successful Cypher run, persist the result row
  IDs (Patient IDs, Practice codes, etc.) under `focus:{conversation_id}`.
- `memory/rewriter.py` — pre-LLM step:

  ```
  Transcript:
    user: which patients have the highest balance in Tennessee?
    asst: [10 patient IDs returned]
    user: which of those leave a balance after IVR pay?
  Focus IDs: [SAPA:1000001, SAPA:1000002, ...]

  → Rewrite the last user message into a standalone question that includes
    the focus IDs.
  ```

  Outputs: `"Which patients from {SAPA:1000001, ...} have a non-zero balance
  after their IVR pay events?"`

**Acceptance:** follow-ups like *"narrow that to Atlanta"* resolve correctly;
restart `rp-agent` and a paused conversation still picks up its focus.

### Phase C — answer cache (3 days)

After memory is in place, we can hash `(rewritten_question, schema_version)` and
skip everything when there's a hit.

- `cache/answer_cache.py` — `get(key)` / `set(key, payload, ttl=1h)`. Payload is
  the full `{text, cypher, results, chart_spec}` so we can re-stream it without
  re-running.
- Add a `cache_hit` field to `/health` and a `?nocache=1` query param.

**Acceptance:** identical second question returns ≤50ms with the same chart;
cache TTL respected; cache misses on the *original* question still happen if
the rewriter produces something new (correct, by design).

### Phase D — hybrid retriever (2–3 weeks)

The heavy lift. Splits in two:

**D.1 — offline vectorization (`semantic/embeddings.py`)**

- Build a per-node profile string: e.g. for Patient, concat name + cohort +
  payor + state + recent procedures + balance bucket.
- Embed with `sentence-transformers/all-MiniLM-L6-v2` (or whatever the team
  standardizes on).
- Write embeddings into a Neo4j vector index: `CREATE VECTOR INDEX patient_emb`.
- Re-run nightly.

**D.2 — runtime retriever (`qa/hybrid_retriever.py`)**

- For semantic questions (decided by `qa/router.py`):
  1. Embed user question.
  2. Vector search top-K entry nodes.
  3. Expand: 1–2 hop traversal around each entry node.
  4. Build a context bundle (subgraph triples + a flattened text).
  5. Hand to the answer LLM with both the bundle and a *generated* Cypher
     (text2cypher still runs in parallel for evidence).

**D.3 — `qa/router.py`**

Cheap classifier: regex/heuristic first (counts/aggregates/"how many" → cypher;
"which patients like..."/"similar to" → hybrid). Fall back to a small prompt
classifier for ambiguous queries.

**Acceptance:** "find patients similar to {focus IDs}" returns reasonable
neighbours; aggregate questions are unchanged.

### Phase E — metadata catalog (parallel to A/B)

Quick win that improves all subsequent phases. Hand-write
`metadata/data_dictionary.yaml`:

```yaml
labels:
  Patient:
    description: "An RP patient with rollup financials and contact info"
    properties:
      payor_cohort:
        values:
          bai: "BCBS Affiliated, Inc."
          sapa: "Self-pay"
          ...
relationships:
  HAS_CHARGE:
    description: "..."
```

`metadata/catalog.py` injects this into the text2cypher prompt and post-processes
results to swap codes for friendly names.

---

## 3. Key contracts

### Memory protocol

```python
# memory/session.py
class SessionStore:
    def get_history(self, conv_id: str) -> list[BaseMessage]: ...
    def append(self, conv_id: str, msg: BaseMessage) -> None: ...
    def get_focus(self, conv_id: str) -> list[str]: ...
    def set_focus(self, conv_id: str, ids: list[str]) -> None: ...
```

### Guardrail protocol

Every guardrail is `def check(payload) -> GuardrailResult` where
`GuardrailResult` has `ok: bool`, `payload: T` (possibly mutated — e.g. LIMIT
injected), `reason: str`. The chain calls them in a list and bails on the first
`ok=False`.

### Conversation ID propagation

LibreChat passes `conversation_id` in the request body via the `metadata` field
on the OpenAI-compatible endpoint. We thread it from
`openai_compat.chat_completions` → `qa.chain.stream_qa_response(question,
conversation_id)`. Falls back to `hash(first_user_message)` if missing.

---

## 4. Redis keys

| Pattern | TTL | Purpose |
|---|---|---|
| `chat:{conv_id}` | 24h | LangChain message history list |
| `focus:{conv_id}` | 24h | JSON list of focus entity IDs |
| `cache:{q_hash}:{schema_ver}` | 1h | Cached answer payload |
| `schema:snapshot` | manual | Allowed labels/props, refreshed on ingestion |
| `embed:patient:{id}` | n/a | Optional Redis vector cache (else Neo4j-only) |

Single Redis instance is fine for v2 — split out later if hot keys emerge.

---

## 5. Configuration additions

Add to `config.py`:

```python
redis_url: str = Field("redis://redis:6379/0", env="REDIS_URL")
session_ttl_seconds: int = Field(86400, env="SESSION_TTL")
cache_ttl_seconds: int = Field(3600, env="CACHE_TTL")
cypher_row_limit: int = Field(100, env="CYPHER_ROW_LIMIT")
cypher_timeout_seconds: int = Field(15, env="CYPHER_TIMEOUT")
guardrails_enabled: bool = Field(True, env="GUARDRAILS_ENABLED")
hybrid_retriever_enabled: bool = Field(False, env="HYBRID_RETRIEVER_ENABLED")
```

And to the compose `agent` service env block plus `dockers/.env.example`.

Add `redis` service to `docker-compose.yml`:

```yaml
redis:
  image: redis:7.4-alpine
  container_name: rp-redis
  restart: unless-stopped
  ports:
    - "6379:6379"
  command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
  volumes:
    - redis_data:/data
```

---

## 6. Open questions / decisions for next session

- **PHI scope** — do we mask the patient's own data (anyone authenticated can
  see their own row), or always mask? Affects how we key the redaction by
  caller identity. If LibreChat doesn't pass user identity to the agent, we
  default to always-mask.
- **Embedding model** — local (sentence-transformers) vs OpenAI/Anthropic
  hosted. Local is free + private but adds GPU dependency.
- **Schema version** — bumping schema version invalidates the cache. Tie this
  to the ingestion timestamp from `ingest/ingestion.py`? Or a manual constant?
- **Guardrail bypass** — admin role can bypass the row cap for ad-hoc
  reports? Or always enforced and reports use a separate code path?
- **Conversation ID** — verify LibreChat actually sends one. If not, we either
  patch LibreChat or hash on first user message + IP (less reliable).

---

## 7. What to build first

Recommendation: **Phase A (guardrails) + Phase E (catalog)** in week 1 — both
are pure additions that improve every existing query path without new infra.
Then Phase B (memory) once Redis is in the stack. Phase C (cache) lands the
day after memory. Phase D (hybrid retriever) is its own multi-week project —
ship it behind `HYBRID_RETRIEVER_ENABLED=false` until the offline build is
proven.
