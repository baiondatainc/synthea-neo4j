# Context awareness — how it's built

Conversational memory + follow-up rewriting (Phase B of the [v2 design](v2_design.md)).
Lets the user ask follow-ups like *"narrow them to Atlanta"* without restating
the prior question, while keeping the agent stateless from the request's point
of view.

---

## The runtime flow

```
LibreChat sends OpenAI-style request
  │
  ▼
api/openai_compat.py   ── extract conversation_id from body/headers ──┐
  │                                                                    │
  ▼                                                                    │
qa/chain.py: stream_qa_response(question, conversation_id)             │
  │                                                                    │
  ▼                                                                    │
[1] Input guardrail (Phase A — block injection / off-topic)            │
  │                                                                    │
  ▼                                                                    │
[2] Pull transcript + focus IDs from Redis  ◄── keyed by conversation_id ┘
  │
  ▼
[3] If question is elliptical → rewriter (LLM) → standalone question
  │
  ▼
[4] Cache lookup with the *rewritten* question (Phase C)
  │   on hit → replay bundle, also update memory
  ▼
[5] text2cypher → cypher guardrail → Neo4j → qa_llm → stream answer
  │
  ▼
[6] On success: append (user_q, answer) to transcript +
                extract & store focus IDs from results
```

---

## The four pieces

### 1. Identifying a conversation
[`api/openai_compat.py`](../api/openai_compat.py)

`_extract_conversation_id(body, request)` checks five locations in order:

| Location | Notes |
|---|---|
| `body.metadata.conversation_id` | LibreChat custom metadata bag |
| `body.conversation_id` / `conversationId` / `chat_id` | Common variants |
| `body.user` | Standard OpenAI session field |
| `X-Conversation-Id` header | For ad-hoc clients |
| `hash(first_user_message)` | Fallback — stable because LibreChat resends the full transcript every turn |

The hash fallback is why memory works even when the upstream UI doesn't send a real id — the same conversation always produces the same key.

### 2. Session store
[`memory/session.py`](../memory/session.py)

A thin wrapper around `langchain_community.chat_message_histories.RedisChatMessageHistory`. Stores a JSON-serialized list of `HumanMessage`/`AIMessage` under `chat:{conversation_id}` with `SESSION_TTL` (default 24h).

Why a wrapper instead of using LangChain's class directly:

- **Fallback** — if Redis is unreachable, a process-local dict keeps the agent serving (no persistence, but no crash either)
- **`.transcript(conv_id, max_turns=6)`** renders the last N turns as a prompt-ready string
- **TTL refresh** on every read/write keeps active conversations alive without manual touch

### 3. Focus tracker
[`memory/focus.py`](../memory/focus.py)

After every successful query, `extract_entity_ids(results)` scans Neo4j result rows for known ID-bearing columns (`patientId`, `practiceId`, `locationId`, …), dedupes, caps at 50, and persists the list to `focus:{conversation_id}`.

This is what makes *"narrow them to Atlanta"* work — we know who *them* refers to because we captured the IDs from the previous answer.

Replace-not-append: each successful turn **replaces** the focus set rather than growing it. The conversation should always be "on" the most recent answer's entities, not the union of everything ever discussed.

### 4. The rewriter
[`memory/rewriter.py`](../memory/rewriter.py)

LLM calls are the expensive part, so we avoid them whenever possible with a three-stage gate:

```python
def rewrite_question(question, transcript, focus_ids):
    if not transcript:                       return question  # first turn → pass through
    if not focus_ids:                        return question  # nothing to anchor to
    if not _is_likely_followup(question):    return question  # heuristic skip
    return llm_rewrite(...)                                   # only here do we pay
```

`_is_likely_followup` uses two regexes:

| Group | Patterns |
|---|---|
| **Demonstratives** | `those`, `them`, `that`, `it`, `they`, `theirs`, `same` |
| **Follow-up verbs** | `narrow`, `drill`, `filter`, `exclude`, `instead`, `only`, `but also`, `then` |

Plus a length heuristic: questions shorter than ~35 chars ending in `?` are almost always elliptical (*"what about TN?"*).

When the LLM does run, the prompt is tight: recent transcript + focus IDs + new question + an explicit *"if standalone, return unchanged"* instruction.

---

## How memory + cache compose

Because the answer cache is keyed on the **rewritten** question (not the original), context-aware caching falls out for free:

> **Turn 1** — *"Who's the top patient in TN?"* → nothing to rewrite → LLM runs → answer cached under `hash("who's the top patient in tn")` + focus IDs persisted.
>
> **Turn 2** — *"how about their procedures?"* → rewriter expands to *"What procedures did patients ACRB:1000023, NRAA:1000021, … have?"* → cache key uses the **expanded** form → first time misses (LLM runs), repeat hits in <50ms.

So a user replaying the same follow-up sequence twice gets the second one straight out of Redis.

---

## Where it's wired

The chain integration in [`qa/chain.py`](../qa/chain.py) is small:

```python
# top of stream_qa_response
if conversation_id and settings.memory_enabled:
    session_store = get_session_store()
    focus_store = get_focus_store()
    transcript = session_store.transcript(conversation_id)
    focus_ids = focus_store.get(conversation_id)
    rewritten = rewrite_question(question, transcript, focus_ids)
    if rewritten != question:
        yield {"type": "rewrite", "data": rewritten}
        question = rewritten

# … chain runs on `question` …

# bottom of stream_qa_response, on success
if conversation_id:
    session_store.append_user(conversation_id, original_question)
    session_store.append_assistant(conversation_id, answer_text)
    focus_store.set(conversation_id, extract_entity_ids(results))
```

We persist the **original** user question to the transcript (not the rewritten one) — so the next rewrite sees what the user actually typed, which is what the LLM needs to disambiguate the *next* follow-up.

---

## Design choices, in one breath

- **Redis instead of in-process** — survives restarts, shared across workers, TTL semantics for free.
- **Two stores instead of one** — focus has a different lifecycle than transcript (replace vs append).
- **Heuristic gate before LLM** — saves a call on every first-turn or self-contained question (the common case).
- **LangChain `RedisChatMessageHistory`** — consistent serialization with the rest of the stack; gives a clean swap path if we later move to a vector-search-over-history layer.
- **Fail-soft** — Redis down → in-process fallback; rewriter LLM errors → original question is used.

---

## Inspect it live

```bash
# Watch the keys as you chat
docker exec -it rp-redis redis-cli MONITOR | grep -E 'chat:|focus:'

# Or directly
docker exec -it rp-redis redis-cli
> KEYS chat:*
> LRANGE chat:<conv_id> 0 -1
> GET focus:<conv_id>
> TTL chat:<conv_id>
```

---

# Next step — how to improvise it

Eight upgrades, roughly in order of bang-for-buck. None of them require infra changes; each is a localized addition.

### 1. Replace the regex follow-up detector with a tiny classifier

Today's `_is_likely_followup` uses two regexes. It catches most cases but misses paraphrases (*"can you do the same but for charges?"* — `do the same` doesn't match `same` because of word boundaries quirks) and false-positives on legitimate words (*"let's narrow our scope to imaging"* fires `narrow` even when standalone).

A 200-example fine-tune on a small instruction-tuned model, or a cheap zero-shot prompt against the QA LLM (`"Is this a follow-up that needs prior context to answer? yes/no"`), would push the gate from ~80% to ~95% accurate. Cost is one fast LLM call per turn instead of zero — still much cheaper than always rewriting.

### 2. Track multiple focus dimensions

`focus:{conv_id}` is one list. But conversations move through different dimensions: patient IDs, practice codes, locations, date windows, payor cohorts. Today they all collide in one list — *"narrow them to high-balance"* applied to a list that mixes patients and practices gives the LLM mixed signals.

Bucket the focus state into named dimensions:

```python
focus = {
    "patients":  ["ACRB:1000023", "NRAA:1000021"],
    "practices": ["ACRB"],
    "states":    ["TN"],
    "window":    {"from": "2024-01", "to": "2025-12"},
}
```

The rewriter then renders only the buckets it needs. Patients aren't accidentally substituted into a question about practices.

### 3. Summarize the transcript when it grows

Right now `.transcript(max_turns=6)` truncates by turn count. For long sessions this either drops important context (truncating too aggressively) or grows the prompt unnecessarily (keeping all turns).

A rolling summary — every N turns, a small LLM call produces a one-paragraph summary of everything before the most recent 2–3 turns, stored under `summary:{conv_id}`. The transcript handed to the rewriter is then `summary + last 3 turns` instead of raw last 6. Keeps prompt cost flat as the conversation grows.

### 4. Confidence-aware rewriting

When the rewriter's input is ambiguous (*"can you also include the other ones?"* — which other ones?), it currently guesses. Better: have the rewriter return a structured response

```json
{"action": "rewrite", "question": "..."}
{"action": "ask", "clarification": "Which 'other ones' — the patients from Atlanta or the procedures from last month?"}
{"action": "passthrough"}
```

Emit a `clarification` event instead of guessing wrong. UI surfaces a question back to the user. This costs one extra LLM call when triggered, but kills a class of "the agent answered confidently but about the wrong thing" failures.

### 5. Semantic cache key

Today `make_cache_key(question, schema_version)` lowercases and collapses whitespace. *"How many patients in TN?"* and *"Number of patients in Tennessee"* are different keys.

Add an embedding-based fallback: on cache miss, embed the question via the same `sentence-transformers` model already loaded for Phase D, do an ANN lookup against past cached questions (keys stored in a small Redis sorted-set + values in `cache:answer:*`), and if the cosine similarity > 0.92 reuse the bundle. Half-the-cost of running the LLM for paraphrases.

### 6. Vector memory — retrieve relevant turns, not just recent ones

For long sessions, the most useful prior turn might be 12 messages ago, not the last 2. Embed every persisted turn into Redis (or Neo4j vector index — we already have one from Phase D), and let the rewriter retrieve the top-K *semantically relevant* turns rather than the top-K *recent* turns.

The Patient Journey UI's "find similar to this patient" logic translates directly: embed user-question → ANN search over `embed:msg:{conv_id}:*` → assemble retrieved turns into the rewriter prompt.

### 7. Cross-conversation persona memory

Today everything lives under `{conv_id}`. But the user is the same person across conversations. Store a per-user profile:

```
user:{user_id} → {"roles": ["RCM analyst"], "preferred_practices": ["ACRB"], "common_filters": ["self_pay"]}
```

Updated occasionally by an offline job that mines patterns in their question history. The rewriter and even the cypher prompt can use this as soft context — *"how many of them this quarter?"* defaults to "RCM analyst's usual practices" when the conversation otherwise can't disambiguate.

### 8. Audit and debug surfaces

A few small additions that pay back fast in operability:

- **`GET /sessions/{conv_id}`** — returns the current transcript, focus state, and Redis TTLs as JSON. Replaces `redis-cli` for routine debugging.
- **Rewrite log** — append every `(original, rewritten, focus_ids)` tuple to a Redis stream `rewrites:log` with a 7-day cap. Lets you grep "rewriter said the wrong thing" cases later.
- **Cache-hit telemetry** — count hits, misses, and stale invalidations per cache_key. Surface as `/cache/stats`. Tells you which questions are warm-cache candidates without guessing.
- **Memory dashboard tab** — add a "Memory" tab to `metadata-ui` mirroring the Patient Journey layout: pick a conversation, see its transcript, focus IDs, recent rewrites, cache hits.

---

## Suggested order

| When | Do |
|---|---|
| Now | (8) audit endpoints — high value, ~1 day |
| This week | (1) classifier-based detector, (2) multi-dimensional focus |
| Next sprint | (3) rolling summary, (4) confidence-aware rewriter |
| Bigger lift | (5) semantic cache key + (6) vector memory (combine — they share the embedding pipeline) |
| Long-term | (7) cross-conversation persona, once you have stable user identity from LibreChat |
