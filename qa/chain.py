"""
LangChain RAG pipeline with streaming and result extraction for charting.

Architecture:
  memory rewrite → input guardrail → cypher_llm (text2cypher) → autofix →
  cypher guardrail (read-only, schema, row cap) → Neo4j → output redaction →
  qa_llm → streamed answer

Caching:
  - The Neo4j graph (connection + introspected schema) is built once per
    process and reused across requests. Call `invalidate_chain_cache()`
    after a schema change or ingestion to rebuild.
  - The Cypher LLM ChatOllama client is similarly cached.

Timing:
  - Per-request timing logs every major phase. Set LOG_LEVEL=INFO to see.

Changes vs previous version (query logging fixes):
  - EVERY exit path now writes a query_log row:
      * input guardrail block   → status="blocked"
      * answer cache hit        → status="cache_hit"
      * hybrid success          → status="hybrid"
      * hybrid crash            → status="error"
      * empty cypher            → status="blocked"
      * cypher guardrail block  → status="blocked"
      * Neo4j execution error   → status="error"  (and NOW RETURNS — see below)
      * success                 → status="ok"
  - BUGFIX: the generic Neo4j execution exception handler previously logged an
    error row but did not return, so the pipeline fell through to the QA phase
    with results=[], streamed an answer over zero rows, and Phase F logged a
    SECOND row with status="ok". It now yields an error event and returns.
  - `rewritten` is computed once after Phase B and passed to every log call
    (previously only 2 of 6 call sites included it).
  - If the QA answer generation fails but the fallback text is used, the final
    "ok" row carries detail="qa_fallback: <error>" so it is distinguishable.
"""
import os
import time
import logging
import asyncio
from typing import AsyncGenerator, Any

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama

from config import get_settings
from graph.schema_text import GRAPH_SCHEMA
from qa.llm import get_llm
from metadata.catalog import get_catalog
from guardrails import check_input, check_cypher, redact_rows, redact_text
from memory import (
    get_session_store,
    get_focus_store,
    extract_entity_ids,
    rewrite_question,
)
from cache import get_answer_cache, make_cache_key
from qa.router import route, Path
from qa.hybrid_retriever import stream_hybrid_response
from qa.cypher_autofix import autofix_cypher
from qa.querylog import log_query

logger = logging.getLogger(__name__)


# ── Module-level caches ───────────────────────────────────────────────────────

_GRAPH_CACHE: "Neo4jGraph | None" = None
_CYPHER_LLM_CACHE = None

# The text2cypher model already contains the full graph schema in its Modelfile
# SYSTEM prompt. We therefore do NOT re-inject GRAPH_SCHEMA into the LangChain
# cypher prompt. If you ever switch cypher_model to a generic model that lacks a
# baked-in schema, set _INJECT_SCHEMA = True to restore full-schema injection.
_INJECT_SCHEMA = False
_SCHEMA_PLACEHOLDER = (
    "The full graph schema (labels, properties, relationships, direction rules, "
    "and examples) is provided in the Cypher model's system prompt. Generate "
    "Cypher directly from the question."
)

# Cap the number of result rows fed into the QA answer prompt (charts still get
# the full result set). Keeps the QA prompt small on large result sets.
_QA_CONTEXT_ROW_CAP = 100


def invalidate_chain_cache() -> None:
    """Clear the graph and LLM caches. Call after schema/ingestion changes."""
    global _GRAPH_CACHE, _CYPHER_LLM_CACHE
    _GRAPH_CACHE = None
    _CYPHER_LLM_CACHE = None
    logger.info("Chain cache invalidated — graph + LLM will rebuild on next request")


# ── Streaming callback (retained for compatibility; unused by the direct path) ─

class StreamingCallback(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queue.put({"type": "token", "data": token})

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        await self.queue.put({"type": "end", "data": ""})

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        await self.queue.put({"type": "error", "data": str(error)})

    async def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        pass


# ── Guarded Neo4j Graph ───────────────────────────────────────────────────────

class GuardrailBlocked(Exception):
    """Raised when the Cypher guardrail rejects a query."""


class GuardedNeo4jGraph(Neo4jGraph):
    """Drop-in Neo4jGraph that autofixes, runs the guardrail, and redacts PII rows."""

    def query(self, query: str, params: dict | None = None) -> list[dict]:
        settings = get_settings()

        if settings.guardrails_enabled:
            original_query = query
            query = autofix_cypher(query, logger)   # strips/extracts fences + fixes patterns
            if query != original_query:
                logger.info("autofix applied — query patched before guardrail")

            if not query.strip():
                # Model produced no query (e.g. a bare fence). Fail clearly.
                raise GuardrailBlocked("Empty Cypher after autofix — the model produced no query.")

            check = check_cypher(query)
            if not check.ok:
                logger.warning(f"Cypher guardrail blocked: {check.reason}\nQuery: {query[:300]}")
                raise GuardrailBlocked(check.reason)
            query = check.payload

        rows = super().query(query, params)

        if settings.guardrails_enabled and settings.guardrails_redact_output:
            rows = redact_rows(rows)
        return rows


# ── Neo4j Graph factory (cached) ──────────────────────────────────────────────

def get_neo4j_graph() -> Neo4jGraph:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE

    t0 = time.monotonic()
    settings = get_settings()
    catalog = get_catalog()

    graph_obj = GuardedNeo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        enhanced_schema=False,
    )
    t_connect = time.monotonic()

    # refresh_schema() still runs so the driver has live label/rel metadata for
    # its own internal use, but we override graph_obj.schema (the text that gets
    # injected into CYPHER_GENERATION_PROMPT) to avoid double-sending the schema.
    graph_obj.refresh_schema()
    t_refresh = time.monotonic()

    if _INJECT_SCHEMA:
        graph_obj.schema = f"{GRAPH_SCHEMA}\n\n{catalog.schema_addendum()}"
    else:
        # Model carries the schema in its Modelfile SYSTEM prompt — don't resend.
        graph_obj.schema = _SCHEMA_PLACEHOLDER
    t_build = time.monotonic()

    schema_chars = len(graph_obj.schema)
    logger.info(
        f"Neo4j graph initialised — "
        f"connect={int((t_connect - t0) * 1000)}ms, "
        f"refresh_schema={int((t_refresh - t_connect) * 1000)}ms, "
        f"build_text={int((t_build - t_refresh) * 1000)}ms, "
        f"inject_schema={_INJECT_SCHEMA}, "
        f"schema={schema_chars} chars (~{schema_chars // 4} tokens)"
    )

    if os.getenv("DUMP_SCHEMA", "").lower() in ("1", "true", "yes"):
        try:
            with open("/tmp/schema_dump.txt", "w") as f:
                f.write(graph_obj.schema)
            logger.info("Schema dumped to /tmp/schema_dump.txt")
        except Exception as e:
            logger.warning(f"Could not dump schema: {e}")

    _GRAPH_CACHE = graph_obj
    return graph_obj


# ── Cypher specialist LLM (cached) ────────────────────────────────────────────

def get_cypher_llm():
    global _CYPHER_LLM_CACHE
    if _CYPHER_LLM_CACHE is not None:
        return _CYPHER_LLM_CACHE

    settings = get_settings()
    try:
        llm = ChatOllama(
            model=settings.cypher_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=256,
            # 8192 (was 4096): the Modelfile SYSTEM prompt (schema + ~55 few-shot
            # examples) is ~6k tokens. At 4096 it was truncated. NOTE: runtime
            # num_ctx OVERRIDES the Modelfile PARAMETER, so this is authoritative.
            num_ctx=8192,
            keep_alive="24h",
        )
        logger.info(
            f"Cypher LLM initialised — model={settings.cypher_model!r}, "
            f"num_ctx=8192, num_predict=256, keep_alive=24h"
        )
        _CYPHER_LLM_CACHE = llm
        return llm
    except Exception as e:
        logger.warning(
            f"Cypher model {settings.cypher_model!r} not available ({e}). "
            "Falling back to default LLM for Cypher generation."
        )
        return get_llm(streaming=False)


# ── Prompts ───────────────────────────────────────────────────────────────────

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""Schema:
{schema}

Question: {question}

Cypher:""",
)

QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a helpful healthcare data analyst.

Given these graph query results, provide a clear plain-English answer.
Include: a direct answer, key numbers/patterns, and any notable insights.
If a value is marked [name-redacted], [phone-redacted], etc., refer to the
entity by its identifier or position instead of inventing names.

Question: {question}
Graph Results: {context}

Answer:""",
)


# ── Build chain (retained for compatibility; the direct path does not use it) ──

def build_chain(streaming_callback: StreamingCallback = None) -> GraphCypherQAChain:
    t0 = time.monotonic()

    graph_obj = get_neo4j_graph()
    t_graph = time.monotonic()

    cypher_llm = get_cypher_llm()
    t_cypher_llm = time.monotonic()

    qa_llm = get_llm(streaming=bool(streaming_callback))
    if streaming_callback:
        qa_llm.callbacks = [streaming_callback]
    t_qa_llm = time.monotonic()

    chain = GraphCypherQAChain.from_llm(
        llm=qa_llm,
        graph=graph_obj,
        cypher_llm=cypher_llm,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=QA_GENERATION_PROMPT,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        input_key="query",
    )
    t_done = time.monotonic()

    logger.debug(
        f"Chain built — "
        f"graph={int((t_graph - t0) * 1000)}ms, "
        f"cypher_llm={int((t_cypher_llm - t_graph) * 1000)}ms, "
        f"qa_llm={int((t_qa_llm - t_cypher_llm) * 1000)}ms, "
        f"wire={int((t_done - t_qa_llm) * 1000)}ms, "
        f"total={int((t_done - t0) * 1000)}ms"
    )
    return chain


# ── Extract results from intermediate steps (retained for compatibility) ──────

def extract_from_steps(intermediate_steps: list) -> tuple[str, list]:
    cypher = ""
    results = []

    for step in intermediate_steps:
        if isinstance(step, dict):
            if "query" in step and not cypher:
                cypher = step["query"]
            if "context" in step and not results:
                ctx = step["context"]
                if isinstance(ctx, list):
                    results = ctx
        elif isinstance(step, (list, tuple)):
            for sub in step:
                if isinstance(sub, dict):
                    if "query" in sub and not cypher:
                        cypher = sub["query"]
                    if "context" in sub and not results:
                        ctx = sub["context"]
                        if isinstance(ctx, list):
                            results = ctx

    return cypher, results


# ── Direct Cypher generation (replaces GraphCypherQAChain's broken extraction) ─

async def _generate_cypher(question: str, graph_obj, cypher_llm) -> str:
    """Call the cypher LLM directly and return cleaned Cypher.

    Bypasses GraphCypherQAChain.extract_cypher(), which mangled our model's
    fenced output. autofix_cypher() extracts the fenced body and fixes common
    pattern mistakes. Returns "" if the model produced no query.
    """
    cypher_prompt = CYPHER_GENERATION_PROMPT.format(
        schema=graph_obj.schema, question=question
    )
    raw = await cypher_llm.ainvoke(cypher_prompt)
    raw_text = raw.content if hasattr(raw, "content") else str(raw)
    logger.info(f"Raw cypher LLM output: {raw_text[:160]!r}")
    cleaned = autofix_cypher(raw_text, logger)
    logger.info(f"Cypher after autofix: {cleaned[:160]!r}")
    return cleaned


# ── Streaming generator ───────────────────────────────────────────────────────

async def stream_qa_response(
    question: str,
    conversation_id: str | None = None,
    use_cache: bool = True,
) -> AsyncGenerator[dict, None]:
    """Runs the full RAG pipeline and yields events as they complete.

    Yield types:
      {"type": "token",      "data": "<word>"}
      {"type": "rewrite",    "data": "<rewritten question>"}
      {"type": "cypher",     "data": "<cypher>", "results": [...]}
      {"type": "cache_hit",  "data": "<key>"}
      {"type": "end",        "data": ""}
      {"type": "error",      "data": "<message>"}
      {"type": "blocked",    "data": "<reason>"}

    Query logging: every exit path writes exactly ONE query_log row.
    """
    settings = get_settings()
    t_start = time.monotonic()
    timings: dict[str, float] = {}

    def _elapsed_ms() -> int:
        return int((time.monotonic() - t_start) * 1000)

    # Stores initialised early so Phase B (rewrite) and Phase F (persist) share
    # them and the raw user turn is always available to write.
    original_question = question
    session_store = get_session_store() if conversation_id else None
    focus_store = get_focus_store() if conversation_id else None

    # ── Phase B: memory + follow-up rewriting (BEFORE the guardrail) ──────
    t_mem = time.monotonic()
    if conversation_id and settings.memory_enabled and session_store and focus_store:
        transcript = session_store.transcript(conversation_id)
        focus_ids = focus_store.get(conversation_id)
        logger.info(
            f"Phase B: transcript_chars={len(transcript) if transcript else 0}, "
            f"focus_ids={len(focus_ids)}, question={question!r}"
        )
        if transcript:
            rewritten_q = rewrite_question(question, transcript, focus_ids)
            if rewritten_q and rewritten_q != question:
                logger.info(f"Phase B rewrite (pre-guard): {question!r} → {rewritten_q!r}")
                yield {"type": "rewrite", "data": rewritten_q}
                question = rewritten_q
        else:
            logger.info("Phase B: no transcript yet — skipping rewrite (first turn)")
    timings["memory"] = time.monotonic() - t_mem

    # Computed once, passed to EVERY log_query call below.
    rewritten = question if question != original_question else None

    # ── Phase A: input guardrail (on the possibly-rewritten question) ─────
    t_a = time.monotonic()
    if settings.guardrails_enabled:
        gate = check_input(question)
        if not gate.ok:
            logger.info(f"Input guardrail blocked: {gate.reason}")
            log_query(
                question=original_question,
                conversation_id=conversation_id,
                rewritten=rewritten,
                status="blocked",
                detail=f"input_guardrail: {gate.reason}",
                total_ms=_elapsed_ms(),
                model=settings.cypher_model,
            )
            yield {"type": "blocked", "data": gate.reason}
            yield {"type": "end", "data": ""}
            return
        question = gate.payload
    timings["guardrail"] = time.monotonic() - t_a

    # ── Phase C: answer cache lookup (after rewrite, before LLM) ──────────
    t_cache = time.monotonic()
    cache = None
    cache_key = None
    if use_cache and settings.cache_enabled:
        cache = get_answer_cache()
        cache_key = make_cache_key(question, settings.schema_version)
        cached = cache.get(cache_key)
        if cached:
            timings["cache_lookup"] = time.monotonic() - t_cache
            timings["total"] = time.monotonic() - t_start
            logger.info(
                f"Answer cache HIT — key={cache_key}, "
                f"total={int(timings['total'] * 1000)}ms"
            )
            log_query(
                question=original_question,
                conversation_id=conversation_id,
                rewritten=rewritten,
                cypher=cached.get("cypher") or None,
                status="cache_hit",
                detail=f"key={cache_key}",
                total_ms=int(timings["total"] * 1000),
                row_count=len(cached.get("results") or []),
                answer_chars=len(cached.get("answer") or ""),
                model=settings.cypher_model,
            )
            yield {"type": "cache_hit", "data": cache_key}
            yield {"type": "token", "data": cached.get("answer", "")}
            yield {
                "type": "cypher",
                "data": cached.get("cypher", ""),
                "results": cached.get("results", []),
            }
            yield {"type": "end", "data": ""}
            if conversation_id and session_store and focus_store:
                try:
                    session_store.append_user(conversation_id, original_question)
                    if cached.get("answer"):
                        session_store.append_assistant(conversation_id, cached["answer"])
                    if cached.get("results"):
                        new_focus = extract_entity_ids(cached["results"])
                        if new_focus:
                            focus_store.set(conversation_id, new_focus)
                except Exception as e:
                    logger.warning(f"Memory persist on cache hit failed: {e}")
            return
    timings["cache_lookup"] = time.monotonic() - t_cache

    # ── Phase D: route — cypher (exact/aggregate) or hybrid (similarity) ──
    if settings.hybrid_retriever_enabled and route(question) == Path.HYBRID:
        logger.info(f"Router → HYBRID for: {question[:80]}")
        answer_tokens: list[str] = []
        hybrid_cypher = ""
        hybrid_results: list[dict] = []
        hybrid_error: str | None = None
        t_hybrid = time.monotonic()
        try:
            async for ev in stream_hybrid_response(question):
                if ev["type"] == "token":
                    answer_tokens.append(ev["data"])
                elif ev["type"] == "cypher":
                    hybrid_cypher = ev.get("data", "")
                    hybrid_results = ev.get("results", [])
                elif ev["type"] == "error":
                    hybrid_error = ev.get("data", "unknown hybrid error")
                yield ev
                if ev["type"] in ("end", "error"):
                    break
        except Exception as e:
            logger.error(f"Hybrid path crashed: {e}", exc_info=True)
            log_query(
                question=original_question,
                conversation_id=conversation_id,
                rewritten=rewritten,
                cypher=hybrid_cypher or None,
                status="error",
                detail=f"hybrid: {e}",
                total_ms=_elapsed_ms(),
                model=settings.cypher_model,
            )
            yield {"type": "error", "data": str(e)}
            return

        timings["hybrid"] = time.monotonic() - t_hybrid
        timings["total"] = time.monotonic() - t_start
        logger.info(
            f"Pipeline complete (hybrid) — "
            f"guardrail={int(timings['guardrail'] * 1000)}ms, "
            f"memory={int(timings['memory'] * 1000)}ms, "
            f"cache={int(timings['cache_lookup'] * 1000)}ms, "
            f"hybrid={int(timings['hybrid'] * 1000)}ms, "
            f"total={int(timings['total'] * 1000)}ms"
        )

        answer_text = "".join(answer_tokens).strip()
        log_query(
            question=original_question,
            conversation_id=conversation_id,
            rewritten=rewritten,
            cypher=hybrid_cypher or None,
            exec_ms=int(timings["hybrid"] * 1000),
            total_ms=int(timings["total"] * 1000),
            row_count=len(hybrid_results),
            status="error" if hybrid_error else "hybrid",
            detail=f"hybrid: {hybrid_error}" if hybrid_error else None,
            answer_chars=len(answer_text),
            model=settings.cypher_model,
        )

        try:
            if conversation_id and session_store and focus_store:
                session_store.append_user(conversation_id, original_question)
                if answer_text:
                    session_store.append_assistant(conversation_id, answer_text)
                new_focus = extract_entity_ids(hybrid_results)
                if new_focus:
                    focus_store.set(conversation_id, new_focus)
            if (cache is not None and cache_key and answer_text
                    and hybrid_results and not hybrid_error):
                cache.set(cache_key, {
                    "question": question,
                    "cypher":   hybrid_cypher,
                    "results":  hybrid_results,
                    "answer":   answer_text,
                })
                logger.info(f"Answer cache SET (hybrid): {cache_key}")
        except Exception as e:
            logger.warning(f"Hybrid post-success persist failed: {e}")
        return

    # ── Phase E: Cypher path (DIRECT — bypasses GraphCypherQAChain) ───────
    # GraphCypherQAChain's internal extract_cypher() truncated our model's
    # ```cypher fenced output to just the opening ```. We instead generate,
    # clean (autofix extracts the fenced body), execute (guardrail inside), and
    # stream the answer ourselves — full control, no broken extraction layer.
    t_build = time.monotonic()
    graph_obj = get_neo4j_graph()
    cypher_llm = get_cypher_llm()
    qa_llm = get_llm(streaming=True)
    timings["build_chain"] = time.monotonic() - t_build

    cypher: str = ""
    cypher_gen_ms: int | None = None
    exec_ms: int | None = None
    results: list[dict] = []
    answer_tokens = []
    qa_error: str | None = None
    t_chain_start = time.monotonic()
    t_gen = time.monotonic()

    # 1) Generate Cypher (deterministic, non-streaming).
    try:
        cypher = await _generate_cypher(question, graph_obj, cypher_llm)
    except Exception as e:
        logger.error(f"Cypher generation failed: {e}", exc_info=True)
        log_query(
            question=original_question,
            conversation_id=conversation_id,
            rewritten=rewritten,
            status="error",
            detail=f"cypher_gen: {e}",
            total_ms=_elapsed_ms(),
            model=settings.cypher_model,
        )
        yield {"type": "error", "data": f"Failed to generate a query: {e}"}
        return
    cypher_gen_ms = int((time.monotonic() - t_gen) * 1000)

    if not cypher.strip():
        log_query(
            question=original_question,
            conversation_id=conversation_id,
            rewritten=rewritten,
            cypher_gen_ms=cypher_gen_ms,
            status="blocked",
            detail="empty cypher after autofix",
            total_ms=_elapsed_ms(),
            model=settings.cypher_model,
        )
        yield {"type": "blocked", "data": "The model did not produce a query. Try rephrasing the question."}
        yield {"type": "end", "data": ""}
        return

    # 2) Execute (autofix re-runs idempotently + guardrail + redaction inside).
    #    graph_obj.query is synchronous (Neo4j sync driver) → run off the loop.
    t_exec = time.monotonic()
    try:
        results = await asyncio.to_thread(graph_obj.query, cypher)
        exec_ms = int((time.monotonic() - t_exec) * 1000)
        logger.info(f"Cypher executed — {len(results)} rows")
    except GuardrailBlocked as e:
        log_query(
            question=original_question,
            conversation_id=conversation_id,
            rewritten=rewritten,
            cypher=cypher,
            cypher_gen_ms=cypher_gen_ms,
            status="blocked",
            detail=f"cypher_guardrail: {e}",
            total_ms=_elapsed_ms(),
            model=settings.cypher_model,
        )
        yield {"type": "blocked", "data": f"The generated query was rejected: {e}"}
        yield {"type": "end", "data": ""}
        return
    except Exception as e:
        # BUGFIX: previously this logged and then FELL THROUGH to the QA phase
        # with results=[], producing an answer over zero rows AND a second
        # query_log row with status="ok". Now: log once, tell the client, stop.
        logger.error(f"Cypher execution failed: {e}", exc_info=True)
        log_query(
            question=original_question,
            conversation_id=conversation_id,
            rewritten=rewritten,
            cypher=cypher,
            cypher_gen_ms=cypher_gen_ms,
            exec_ms=int((time.monotonic() - t_exec) * 1000),
            status="error",
            detail=f"cypher_exec: {str(e)[:500]}",
            total_ms=_elapsed_ms(),
            model=settings.cypher_model,
        )
        yield {"type": "error", "data": f"Query execution failed: {e}"}
        return

    # 3) Generate the natural-language answer, streaming tokens as they arrive.
    try:
        context_rows = results[:_QA_CONTEXT_ROW_CAP]
        qa_prompt = QA_GENERATION_PROMPT.format(
            question=question, context=str(context_rows)
        )
        async for chunk in qa_llm.astream(qa_prompt):
            tok = chunk.content if hasattr(chunk, "content") else str(chunk)
            if tok:
                answer_tokens.append(tok)
                yield {"type": "token", "data": tok}
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        qa_error = str(e)[:300]
        if not answer_tokens:
            fallback = f"The query ran and returned {len(results)} row(s)."
            answer_tokens.append(fallback)
            yield {"type": "token", "data": fallback}

    # 4) Emit cypher + results (for charting) and close the stream.
    yield {"type": "cypher", "data": cypher, "results": results}
    yield {"type": "end", "data": ""}

    timings["chain_execute"] = time.monotonic() - t_chain_start

    # ── Phase F: persist to memory + cache on success ─────────────────────
    t_persist = time.monotonic()
    try:
        answer_text = "".join(answer_tokens).strip()

        # Memory — always write the turn so follow-ups can resolve it.
        if conversation_id and session_store and focus_store:
            session_store.append_user(conversation_id, original_question)
            if answer_text:
                session_store.append_assistant(conversation_id, answer_text)
                logger.info(f"Phase F: transcript written for conv_id={conversation_id!r}")
            if results:
                new_focus = extract_entity_ids(results)
                if new_focus:
                    focus_store.set(conversation_id, new_focus)
                    logger.debug(f"Phase F: focus updated — {len(new_focus)} IDs")

        # Answer cache — only cache turns with a real answer + valid Cypher,
        # and never cache a QA-fallback answer.
        if cache is not None and cache_key and answer_text and cypher and not qa_error:
            cache.set(cache_key, {
                "question": question,
                "cypher":   cypher,
                "results":  results,
                "answer":   answer_text,
            })
            logger.info(f"Answer cache SET: {cache_key}")
        else:
            logger.debug(
                f"Cache SET skipped — "
                f"cache={cache is not None}, key={bool(cache_key)}, "
                f"answer={bool(answer_text)} ({len(answer_text)} chars), "
                f"cypher={bool(cypher)}, qa_error={bool(qa_error)}"
            )

    except Exception as e:
        logger.warning(f"Post-persist failed: {e}", exc_info=True)

    timings["persist"] = time.monotonic() - t_persist
    timings["total"] = time.monotonic() - t_start

    log_query(
        question=original_question,
        conversation_id=conversation_id,
        rewritten=rewritten,
        cypher=cypher,
        cypher_gen_ms=cypher_gen_ms,
        exec_ms=exec_ms,
        total_ms=int(timings["total"] * 1000),
        row_count=len(results),
        status="ok",
        detail=f"qa_fallback: {qa_error}" if qa_error else None,
        answer_chars=len("".join(answer_tokens)),
        model=settings.cypher_model,
    )
    logger.info(
        f"Pipeline complete (cypher) — "
        f"guardrail={int(timings['guardrail'] * 1000)}ms, "
        f"memory={int(timings['memory'] * 1000)}ms, "
        f"cache_lookup={int(timings['cache_lookup'] * 1000)}ms, "
        f"build_chain={int(timings['build_chain'] * 1000)}ms, "
        f"chain_execute={int(timings['chain_execute'] * 1000)}ms, "
        f"persist={int(timings['persist'] * 1000)}ms, "
        f"total={int(timings['total'] * 1000)}ms"
    )