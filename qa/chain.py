"""
LangChain RAG pipeline with streaming and result extraction for charting.

Architecture:
  input guardrail → aggregate summary → single node summary →
  memory rewrite → cache lookup → hybrid route → cypher path →
  autofix → cypher guardrail → Neo4j → redaction → qa_llm → streamed answer

Pipeline phases:
  A   — input guardrail
  A2  — aggregate node summary shortcut (all patients, all locations, etc.)
  A3  — single node summary shortcut (patient 12345, location XYZ, etc.)
  B   — memory + follow-up rewriting
  C   — answer cache lookup
  D   — hybrid retriever route
  E   — Cypher QA chain
  F   — memory + cache persist
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
from guardrails import check_input, check_cypher, redact_rows
from qa.cypher_autofix import autofix_cypher
from qa.summarizer import detect_summary_request, summarize_node
from qa.aggregate_summarizer import detect_aggregate_summary, summarize_all_nodes
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

_GRAPH_CACHE: "Neo4jGraph | None" = None
_CYPHER_LLM_CACHE = None


def invalidate_chain_cache() -> None:
    global _GRAPH_CACHE, _CYPHER_LLM_CACHE
    _GRAPH_CACHE = None
    _CYPHER_LLM_CACHE = None
    logger.info("Chain cache invalidated — graph + LLM will rebuild on next request")


# ─────────────────────────────────────────────────────────────────────────────
# Streaming callback
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Guarded Neo4j graph — autofix + guardrail + redaction
# ─────────────────────────────────────────────────────────────────────────────

class GuardrailBlocked(Exception):
    pass


class GuardedNeo4jGraph(Neo4jGraph):
    def query(self, query: str, params: dict | None = None) -> list[dict]:
        settings = get_settings()

        # ── Auto-fix common model errors BEFORE guardrail sees the query ──
        if settings.guardrails_enabled:
            original_query = query
            query = autofix_cypher(query, logger)
            if query != original_query:
                logger.info("autofix applied — query patched before guardrail")

        # ── Guardrail: read-only, schema, row cap ─────────────────────────
        if settings.guardrails_enabled:
            check = check_cypher(query)
            if not check.ok:
                logger.warning(
                    f"Cypher guardrail blocked: {check.reason}\nQuery: {query[:300]}"
                )
                raise GuardrailBlocked(check.reason)
            query = check.payload

        try:
            rows = super().query(query, params)
        except Exception:
            raise

        if settings.guardrails_enabled and settings.guardrails_redact_output:
            rows = redact_rows(rows)

        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Graph + LLM singletons
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Chain builder + step extractor
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Memory persist helper — shared by ALL paths
# ─────────────────────────────────────────────────────────────────────────────

def _persist_memory(
    conversation_id: str | None,
    session_store,
    focus_store,
    original_question: str,
    answer_text: str,
    results: list,
    label: str = "unknown",
) -> None:
    """Write user + assistant turns to Redis. Safe to call from any path."""
    if not (conversation_id and session_store and focus_store):
        logger.info(
            f"Phase F [{label}]: memory SKIPPED — "
            f"conv_id={bool(conversation_id)} "
            f"session={bool(session_store)} "
            f"focus={bool(focus_store)}"
        )
        return
    try:
        session_store.append_user(conversation_id, original_question)
        if answer_text:
            session_store.append_assistant(conversation_id, answer_text)
        if results:
            new_focus = extract_entity_ids(results)
            if new_focus:
                focus_store.set(conversation_id, new_focus)
                logger.info(f"Phase F [{label}]: focus updated — {len(new_focus)} IDs")
        logger.info(
            f"Phase F [{label}]: transcript written — "
            f"conv_id={conversation_id!r} answer={len(answer_text)}chars"
        )
    except Exception as e:
        logger.warning(f"Phase F [{label}]: memory persist failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main streaming entry point
# ─────────────────────────────────────────────────────────────────────────────

async def stream_qa_response(
    question: str,
    conversation_id: str | None = None,
    use_cache: bool = True,
) -> AsyncGenerator[dict, None]:
    settings = get_settings()
    t_start = time.monotonic()
    timings: dict[str, float] = {}

    # ── Phase A: input guardrail ──────────────────────────────────────────
    t_a = time.monotonic()
    if settings.guardrails_enabled:
        gate = check_input(question)
        if not gate.ok:
            logger.info(f"Input guardrail blocked: {gate.reason}")
            yield {"type": "blocked", "data": gate.reason}
            yield {"type": "end", "data": ""}
            return
        question = gate.payload
    timings["guardrail"] = time.monotonic() - t_a

    # Initialise memory stores early — needed by summary paths too
    original_question = question
    session_store = get_session_store() if conversation_id else None
    focus_store   = get_focus_store()   if conversation_id else None

    # ── Phase A2: aggregate node summary ─────────────────────────────────
    # e.g. "summarize all patients", "financial overview", "practice summary"
    agg_intent = detect_aggregate_summary(question)
    if agg_intent:
        logger.info(f"Phase A2: aggregate summary → label={agg_intent['label']}")
        answer_tokens: list[str] = []
        async for chunk in summarize_all_nodes(agg_intent["label"]):
            if chunk["type"] == "token":
                answer_tokens.append(chunk["data"])
            yield chunk
            if chunk["type"] in ("end", "error"):
                break
        _persist_memory(
            conversation_id, session_store, focus_store,
            original_question, "".join(answer_tokens), [],
            label="aggregate_summary",
        )
        timings["total"] = time.monotonic() - t_start
        logger.info(f"Pipeline complete (aggregate_summary) — total={int(timings['total']*1000)}ms")
        return

    # ── Phase A3: single node summary ────────────────────────────────────
    # e.g. "summarize patient 12345", "overview of location SMED Radiology"
    summary_intent = detect_summary_request(question)
    if summary_intent:
        logger.info(
            f"Phase A3: single summary → "
            f"label={summary_intent['label']} id={summary_intent['id']!r}"
        )
        answer_tokens: list[str] = []
        async for chunk in summarize_node(summary_intent["label"], summary_intent["id"]):
            if chunk["type"] == "token":
                answer_tokens.append(chunk["data"])
            yield chunk
            if chunk["type"] in ("end", "error"):
                break
        _persist_memory(
            conversation_id, session_store, focus_store,
            original_question, "".join(answer_tokens), [],
            label="single_summary",
        )
        timings["total"] = time.monotonic() - t_start
        logger.info(f"Pipeline complete (single_summary) — total={int(timings['total']*1000)}ms")
        return

    # ── Phase B: memory + follow-up rewriting ─────────────────────────────
    t_mem = time.monotonic()
    logger.info(
        f"Phase B: conv_id={conversation_id!r} "
        f"memory_enabled={settings.memory_enabled} "
        f"session_store={'ok' if session_store else 'None'} "
        f"focus_store={'ok' if focus_store else 'None'}"
    )

    if conversation_id and settings.memory_enabled and session_store and focus_store:
        transcript = session_store.transcript(conversation_id)
        focus_ids  = focus_store.get(conversation_id)
        logger.info(
            f"Phase B: transcript_chars={len(transcript)} "
            f"focus_ids={len(focus_ids)} "
            f"question={question!r:.80}"
        )
        rewritten = rewrite_question(question, transcript, focus_ids)
        if rewritten != question:
            logger.info(f"Phase B rewrite: {question!r} → {rewritten!r}")
            yield {"type": "rewrite", "data": rewritten}
            question = rewritten
        else:
            logger.info("Phase B: no rewrite (question is standalone or no prior context)")
    timings["memory"] = time.monotonic() - t_mem

    # ── Phase C: answer cache lookup ──────────────────────────────────────
    t_cache   = time.monotonic()
    cache     = None
    cache_key = None
    if use_cache and settings.cache_enabled:
        cache     = get_answer_cache()
        cache_key = make_cache_key(question, settings.schema_version)
        cached    = cache.get(cache_key)
        if cached:
            timings["cache_lookup"] = time.monotonic() - t_cache
            timings["total"]        = time.monotonic() - t_start
            logger.info(
                f"Answer cache HIT — key={cache_key}, "
                f"total={int(timings['total'] * 1000)}ms"
            )
            yield {"type": "cache_hit", "data": cache_key}
            yield {"type": "token",     "data": cached.get("answer", "")}
            yield {"type": "cypher",    "data": cached.get("cypher", ""), "results": cached.get("results", [])}
            yield {"type": "end",       "data": ""}
            _persist_memory(
                conversation_id, session_store, focus_store,
                original_question, cached.get("answer", ""),
                cached.get("results", []), label="cache_hit",
            )
            return
    timings["cache_lookup"] = time.monotonic() - t_cache

    # ── Phase D: hybrid route ─────────────────────────────────────────────
    if settings.hybrid_retriever_enabled and route(question) == Path.HYBRID:
        logger.info(f"Router → HYBRID for: {question[:80]}")
        answer_tokens: list[str] = []
        hybrid_cypher  = ""
        hybrid_results: list[dict] = []
        t_hybrid = time.monotonic()
        try:
            async for ev in stream_hybrid_response(question):
                if ev["type"] == "token":
                    answer_tokens.append(ev["data"])
                elif ev["type"] == "cypher":
                    hybrid_cypher  = ev.get("data", "")
                    hybrid_results = ev.get("results", [])
                yield ev
                if ev["type"] in ("end", "error"):
                    break
        except Exception as e:
            logger.error(f"Hybrid path crashed: {e}", exc_info=True)
            yield {"type": "error", "data": str(e)}
            return

        timings["hybrid"] = time.monotonic() - t_hybrid
        timings["total"]  = time.monotonic() - t_start
        answer_text = "".join(answer_tokens).strip()
        _persist_memory(
            conversation_id, session_store, focus_store,
            original_question, answer_text, hybrid_results, label="hybrid",
        )
        if cache is not None and cache_key and answer_text and hybrid_results:
            cache.set(cache_key, {
                "question": question,
                "cypher":   hybrid_cypher,
                "results":  hybrid_results,
                "answer":   answer_text,
            })
            logger.info(f"Answer cache SET (hybrid): {cache_key}")
        logger.info(f"Pipeline complete (hybrid) — total={int(timings['total']*1000)}ms")
        return

    # ── Phase E: Cypher QA chain ──────────────────────────────────────────
    t_build  = time.monotonic()
    queue    = asyncio.Queue()
    callback = StreamingCallback(queue)
    chain    = build_chain(streaming_callback=callback)
    timings["build_chain"] = time.monotonic() - t_build

    chain_result: dict       = {}
    answer_tokens: list[str] = []
    t_chain_start = time.monotonic()

    async def run_chain():
        try:
            result = await chain.ainvoke({"query": question})
            chain_result["steps"]  = result.get("intermediate_steps", [])
            chain_result["answer"] = result.get("result", "") or ""
            cypher, results = extract_from_steps(chain_result["steps"])
            logger.info(f"Extracted cypher: {cypher[:80] if cypher else 'none'}")
            logger.info(f"Extracted results: {len(results)} rows")
            if chain_result["answer"] and not answer_tokens:
                await queue.put({"type": "token", "data": chain_result["answer"]})
            await queue.put({"type": "cypher", "data": cypher, "results": results})
            await queue.put({"type": "end",    "data": ""})
        except GuardrailBlocked as e:
            logger.info(f"Cypher guardrail rejected: {e}")
            chain_result["finished"] = True
            await queue.put({"type": "blocked", "data": f"The generated Cypher was rejected: {e}. Try rephrasing."})
            await queue.put({"type": "end",     "data": ""})
        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)
            error_msg = str(e)
            if "SyntaxError" in error_msg or "GqlError" in error_msg:
                error_msg = (
                    "The query generator produced invalid Cypher. "
                    f"Try rephrasing.\n\nDetails: {error_msg[:300]}"
                )
            await queue.put({"type": "error", "data": error_msg})

    task = asyncio.create_task(run_chain())

    while True:
        item     = await queue.get()
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

    # ── Phase F: persist to memory + cache ───────────────────────────────
    t_persist = time.monotonic()
    logger.info(
        f"Phase F: conv_id={conversation_id!r} "
        f"session_store={'ok' if session_store else 'None'} "
        f"steps={len(chain_result.get('steps', []))} "
        f"answer_tokens={len(answer_tokens)}"
    )
    try:
        cypher, results = extract_from_steps(chain_result.get("steps", []))
        answer_text = (chain_result.get("answer") or "").strip()
        if not answer_text:
            answer_text = "".join(answer_tokens).strip()

        _persist_memory(
            conversation_id, session_store, focus_store,
            original_question, answer_text, results, label="cypher",
        )

        if cache is not None and cache_key and answer_text and cypher:
            cache.set(cache_key, {
                "question": question,
                "cypher":   cypher,
                "results":  results,
                "answer":   answer_text,
            })
            logger.info(f"Answer cache SET: {cache_key}")
        else:
            logger.info(
                f"Phase F: cache SET skipped — "
                f"cache={cache is not None} key={bool(cache_key)} "
                f"answer={bool(answer_text)}({len(answer_text)}chars) "
                f"cypher={bool(cypher)}"
            )
    except Exception as e:
        logger.warning(f"Phase F persist failed: {e}", exc_info=True)

    timings["persist"] = time.monotonic() - t_persist
    timings["total"]   = time.monotonic() - t_start

    logger.info(
        f"Pipeline complete (cypher) — "
        f"guardrail={int(timings.get('guardrail', 0) * 1000)}ms, "
        f"memory={int(timings.get('memory', 0) * 1000)}ms, "
        f"cache_lookup={int(timings.get('cache_lookup', 0) * 1000)}ms, "
        f"build_chain={int(timings.get('build_chain', 0) * 1000)}ms, "
        f"chain_execute={int(timings.get('chain_execute', 0) * 1000)}ms, "
        f"persist={int(timings.get('persist', 0) * 1000)}ms, "
        f"total={int(timings['total'] * 1000)}ms"
    )