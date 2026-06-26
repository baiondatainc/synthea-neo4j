"""
LangChain RAG pipeline with streaming and result extraction for charting.

Architecture:
  input guardrail → cypher_llm (text2cypher) → cypher guardrail (read-only,
  schema, row cap) → Neo4j → output redaction → qa_llm → streamed answer

Caching:
  - The Neo4j graph (connection + introspected schema) is built once per
    process and reused across requests. Call `invalidate_chain_cache()`
    after a schema change or ingestion to rebuild.
  - The Cypher LLM ChatOllama client is similarly cached.

Timing:
  - Per-request timing logs every major phase. Set LOG_LEVEL=INFO to see.

Changes vs previous version:
  - Phase B: session_store / focus_store initialised unconditionally so they
    are always available for Phase F persist (fixes None-check fragility).
  - Phase F: answer_text prefers chain_result["answer"] then falls back to
    buffered tokens; both paths now log clearly so you can see which fired.
  - Phase F: cache SET guard logs each missing field individually.
  - Minor: removed redundant re import (unused).
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

logger = logging.getLogger(__name__)


# ── Module-level caches ───────────────────────────────────────────────────────

_GRAPH_CACHE: "Neo4jGraph | None" = None
_CYPHER_LLM_CACHE = None


def invalidate_chain_cache() -> None:
    """Clear the graph and LLM caches. Call after schema/ingestion changes."""
    global _GRAPH_CACHE, _CYPHER_LLM_CACHE
    _GRAPH_CACHE = None
    _CYPHER_LLM_CACHE = None
    logger.info("Chain cache invalidated — graph + LLM will rebuild on next request")


# ── Streaming callback ────────────────────────────────────────────────────────

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
    """Drop-in Neo4jGraph that runs the Cypher guardrail and redacts PII rows."""

    def query(self, query: str, params: dict | None = None) -> list[dict]:
        settings = get_settings()

        if settings.guardrails_enabled:
            check = check_cypher(query)
            if not check.ok:
                logger.warning(f"Cypher guardrail blocked: {check.reason}\nQuery: {query[:300]}")
                raise GuardrailBlocked(check.reason)
            query = check.payload

        try:
            rows = super().query(query, params)
        except Exception:
            raise

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

    graph_obj.refresh_schema()
    t_refresh = time.monotonic()

    graph_obj.schema = f"{GRAPH_SCHEMA}\n\n{catalog.schema_addendum()}"
    t_build = time.monotonic()

    schema_chars = len(graph_obj.schema)
    logger.info(
        f"Neo4j graph initialised — "
        f"connect={int((t_connect - t0) * 1000)}ms, "
        f"refresh_schema={int((t_refresh - t_connect) * 1000)}ms, "
        f"build_text={int((t_build - t_refresh) * 1000)}ms, "
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
            num_ctx=4096,
            keep_alive="24h",
        )
        logger.info(
            f"Cypher LLM initialised — model={settings.cypher_model!r}, "
            f"num_ctx=4096, num_predict=256, keep_alive=24h"
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


# ── Build chain ───────────────────────────────────────────────────────────────

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


# ── Extract results from intermediate steps ───────────────────────────────────

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
    """
    settings = get_settings()
    t_start = time.monotonic()
    timings: dict[str, float] = {}

    # ── Phase A: input guardrail ──────────────────────────────────────────
    if settings.guardrails_enabled:
        gate = check_input(question)
        if not gate.ok:
            logger.info(f"Input guardrail blocked: {gate.reason}")
            yield {"type": "blocked", "data": gate.reason}
            yield {"type": "end", "data": ""}
            return
        question = gate.payload
    timings["guardrail"] = time.monotonic() - t_start

    # ── Phase B: memory + follow-up rewriting ─────────────────────────────
    # Initialise stores unconditionally so Phase F can always persist,
    # regardless of whether memory_enabled is True.
    t_mem = time.monotonic()
    original_question = question
    session_store = get_session_store() if conversation_id else None
    focus_store = get_focus_store() if conversation_id else None

    if conversation_id and settings.memory_enabled and session_store and focus_store:
        transcript = session_store.transcript(conversation_id)
        focus_ids = focus_store.get(conversation_id)
        logger.debug(
            f"memory: transcript_turns={transcript.count(chr(10)) + 1 if transcript else 0}, "
            f"focus_ids={len(focus_ids)}, question={question!r}"
        )
        rewritten = rewrite_question(question, transcript, focus_ids)
        if rewritten != question:
            logger.info(f"memory rewrite: {question!r} → {rewritten!r}")
            yield {"type": "rewrite", "data": rewritten}
            question = rewritten
    timings["memory"] = time.monotonic() - t_mem

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
            yield {"type": "cache_hit", "data": cache_key}
            yield {"type": "token", "data": cached.get("answer", "")}
            yield {
                "type": "cypher",
                "data": cached.get("cypher", ""),
                "results": cached.get("results", []),
            }
            yield {"type": "end", "data": ""}
            # Persist memory on cache hit so follow-ups keep working.
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
        t_hybrid = time.monotonic()
        try:
            async for ev in stream_hybrid_response(question):
                if ev["type"] == "token":
                    answer_tokens.append(ev["data"])
                elif ev["type"] == "cypher":
                    hybrid_cypher = ev.get("data", "")
                    hybrid_results = ev.get("results", [])
                yield ev
                if ev["type"] in ("end", "error"):
                    break
        except Exception as e:
            logger.error(f"Hybrid path crashed: {e}", exc_info=True)
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

        try:
            answer_text = "".join(answer_tokens).strip()
            if conversation_id and session_store and focus_store:
                session_store.append_user(conversation_id, original_question)
                if answer_text:
                    session_store.append_assistant(conversation_id, answer_text)
                new_focus = extract_entity_ids(hybrid_results)
                if new_focus:
                    focus_store.set(conversation_id, new_focus)
            if cache is not None and cache_key and answer_text and hybrid_results:
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

    # ── Phase E: Cypher path ──────────────────────────────────────────────
    t_build = time.monotonic()
    queue: asyncio.Queue = asyncio.Queue()
    callback = StreamingCallback(queue)
    chain = build_chain(streaming_callback=callback)
    timings["build_chain"] = time.monotonic() - t_build

    chain_result: dict = {}
    answer_tokens: list[str] = []
    t_chain_start = time.monotonic()

    async def run_chain():
        try:
            result = await chain.ainvoke({"query": question})
            chain_result["steps"] = result.get("intermediate_steps", [])
            chain_result["answer"] = result.get("result", "") or ""
            cypher, results = extract_from_steps(chain_result["steps"])

            logger.info(f"Extracted cypher: {cypher[:80] if cypher else 'none'}")
            logger.info(f"Extracted results: {len(results)} rows")

            # Emit the full answer as a single token if streaming didn't
            # deliver it (callback runs in an executor thread).
            if chain_result["answer"] and not answer_tokens:
                await queue.put({"type": "token", "data": chain_result["answer"]})

            await queue.put({"type": "cypher", "data": cypher, "results": results})
            await queue.put({"type": "end", "data": ""})

        except GuardrailBlocked as e:
            logger.info(f"Cypher guardrail rejected the LLM-generated query: {e}")
            chain_result["finished"] = True
            await queue.put({
                "type": "blocked",
                "data": f"The generated Cypher was rejected by the guardrail: {e}. Try rephrasing.",
            })
            await queue.put({"type": "end", "data": ""})

        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)
            error_msg = str(e)
            if "SyntaxError" in error_msg or "GqlError" in error_msg:
                error_msg = (
                    "The query generator produced invalid Cypher. "
                    "Try rephrasing your question.\n\n"
                    f"Details: {error_msg[:300]}"
                )
            await queue.put({"type": "error", "data": error_msg})

    task = asyncio.create_task(run_chain())

    while True:
        item = await queue.get()

        terminal = chain_result.get("steps") or chain_result.get("finished")
        if item["type"] == "end" and not terminal:
            continue

        if item["type"] == "token":
            answer_tokens.append(item["data"])

        yield item

        if item["type"] in ("end", "error"):
            break

    await task
    timings["chain_execute"] = time.monotonic() - t_chain_start

    # ── Phase F: persist to memory + cache on success ─────────────────────
    # Memory persist runs whenever we have a conversation, even with 0 rows,
    # so the transcript is available for follow-up rewriting on the next turn.
    # Cache SET only runs when we have a non-empty answer AND valid Cypher.
    t_persist = time.monotonic()
    try:
        cypher, results = extract_from_steps(chain_result.get("steps", []))

        # Prefer chain_result["answer"] (set by ainvoke, always populated
        # on success). Fall back to streamed tokens only if it's empty —
        # which happens when the QA LLM doesn't return "result" in its
        # output dict (rare, but guard against it).
        answer_text = (chain_result.get("answer") or "").strip()
        if not answer_text:
            answer_text = "".join(answer_tokens).strip()
            if answer_text:
                logger.debug("Phase F: used streamed tokens as answer_text fallback")
        else:
            logger.debug(f"Phase F: answer_text from chain_result ({len(answer_text)} chars)")

        # Memory — always write the turn so follow-ups can resolve it.
        if conversation_id and session_store and focus_store:
            session_store.append_user(conversation_id, original_question)
            if answer_text:
                session_store.append_assistant(conversation_id, answer_text)
                logger.debug(f"Phase F: transcript written for conv_id={conversation_id!r}")
            # Focus IDs only update when we actually got rows back.
            if results:
                new_focus = extract_entity_ids(results)
                if new_focus:
                    focus_store.set(conversation_id, new_focus)
                    logger.debug(f"Phase F: focus updated — {len(new_focus)} IDs")

        # Answer cache — only cache turns with a real answer + valid Cypher.
        if cache is not None and cache_key and answer_text and cypher:
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
                f"cypher={bool(cypher)}"
            )

    except Exception as e:
        logger.warning(f"Post-persist failed: {e}", exc_info=True)

    timings["persist"] = time.monotonic() - t_persist
    timings["total"] = time.monotonic() - t_start

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