"""
LangChain RAG pipeline with streaming and result extraction for charting.

Architecture:
  input guardrail → cypher_llm (text2cypher) → cypher guardrail (read-only,
  schema, row cap) → Neo4j → output redaction → qa_llm → streamed answer
"""
import re
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
    """Drop-in Neo4jGraph that:
      - runs the Cypher guardrail (read-only, schema, row cap) before execution
      - applies session-level timeout
      - redacts PII columns from result rows before they are returned to the chain
    """

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


# ── Neo4j Graph factory ───────────────────────────────────────────────────────

def get_neo4j_graph() -> Neo4jGraph:
    settings = get_settings()
    catalog = get_catalog()
    graph = GuardedNeo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        enhanced_schema=False,
    )
    graph.refresh_schema()
    # Augment the schema text with friendly descriptions from the data
    # dictionary so the Cypher-generation prompt has human context.
    graph.schema = f"{GRAPH_SCHEMA}\n\n{catalog.schema_addendum()}"
    return graph


# ── Cypher specialist LLM ─────────────────────────────────────────────────────

def get_cypher_llm():
    """Returns the Neo4j text2cypher fine-tuned model running locally via Ollama.

    Falls back to the default configured LLM if text2cypher is not yet
    available so the app keeps working while you set up the specialist model.
    """
    settings = get_settings()

    try:
        llm = ChatOllama(
            model="text2cypher",
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=512,
        )
        logger.info("Cypher LLM: using Neo4j text2cypher specialist model")
        return llm
    except Exception as e:
        logger.warning(
            f"text2cypher not available ({e}). "
            "Falling back to default LLM for Cypher generation."
        )
        return get_llm(streaming=False)


# ── Prompts ───────────────────────────────────────────────────────────────────

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You generate Neo4j CYPHER statements. This is NOT SQL.

CYPHER syntax (NOT SQL):
- Begin every query with MATCH, not FROM or SELECT
- WRONG (SQL):    SELECT count(p) FROM Patient AS p
- CORRECT (Cypher): MATCH (p:Patient) RETURN count(p) AS patient_count

Use only the labels, relationship types, and properties listed in the schema.
Do not invent labels or properties not in the schema.

Schema:
{schema}

STRICT RULES:
- Output ONLY raw Cypher — no markdown, no backticks, no SQL, no explanation
- Every query must start with MATCH (or OPTIONAL MATCH / CALL / WITH / UNWIND)
- Never use SQL keywords: FROM, SELECT, JOIN, GROUP BY, AS in FROM-clause
- GROUP BY does not exist in Cypher — grouping is implicit when you mix
  grouping keys with aggregation in RETURN
- Never use aggregation directly in ORDER BY
  WRONG:    ORDER BY COUNT(p) DESC
  CORRECT:  RETURN p.payor_cohort AS cohort, count(p) AS cnt ORDER BY cnt DESC
- Always alias dotted properties (RETURN p.state AS state)
- Use only READ operations — never CREATE, DELETE, MERGE, SET, REMOVE, DROP

FEW-SHOT EXAMPLES (RP schema):

Q: How many patients are there?
MATCH (p:Patient)
RETURN count(p) AS patient_count

Q: How many patients per practice?
MATCH (p:Patient)-[:REGISTERED_AT]->(pr:Practice)
RETURN pr.code AS practice, count(p) AS patient_count
ORDER BY patient_count DESC

Q: Which patients have the highest outstanding balance?
MATCH (p:Patient)
WHERE p.outstanding_balance IS NOT NULL
RETURN p.patientId AS patient_id, p.outstanding_balance AS balance,
       p.payor_cohort AS cohort
ORDER BY balance DESC
LIMIT 10

Q: What is the total bad debt by state?
MATCH (p:Patient)
WHERE p.state IS NOT NULL
RETURN p.state AS state, sum(p.adj_bad_debt) AS total_bad_debt
ORDER BY total_bad_debt DESC

Q: How many self-pay patients per practice?
MATCH (p:Patient)-[:REGISTERED_AT]->(pr:Practice)
WHERE p.is_self_pay = true
RETURN pr.code AS practice, count(p) AS self_pay_count
ORDER BY self_pay_count DESC

Q: What are the most common procedure modalities?
MATCH (c:Charge)
WHERE c.procedure_modality IS NOT NULL AND c.procedure_modality <> ""
RETURN c.procedure_modality AS modality, count(c) AS charge_count
ORDER BY charge_count DESC
LIMIT 10

Q: Which insurance carriers cover the most visits?
MATCH (v:Visit)-[:UNDER_PLAN]->(i:InsurancePlan)
WHERE i.carrier_name IS NOT NULL
RETURN i.carrier_name AS carrier, count(v) AS visits
ORDER BY visits DESC
LIMIT 10

Q: Show me the gender distribution of patients
MATCH (p:Patient)
WHERE p.gender IS NOT NULL
RETURN p.gender AS gender, count(p) AS count
ORDER BY count DESC

Q: How much was collected via IVR pay-by-phone?
MATCH (p:Patient)-[:CALLED_IVR]->(i:IVRInbound)
WHERE i.amount_paid IS NOT NULL
RETURN sum(i.amount_paid) AS total_ivr_paid

Q: Which locations have the lowest Birdeye ratings?
MATCH (b:BirdeyeReview)-[:REVIEWS]->(l:Location)
RETURN l.name AS location, avg(b.rating) AS avg_rating, count(b) AS review_count
ORDER BY avg_rating ASC
LIMIT 10

Q: Trend of visits by year
MATCH (v:Visit)
WHERE v.admit_date IS NOT NULL
RETURN substring(toString(v.admit_date), 0, 4) AS year, count(v) AS visit_count
ORDER BY year ASC

The question is:
{question}""",
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
    graph = get_neo4j_graph()

    cypher_llm = get_cypher_llm()

    qa_llm = get_llm(streaming=bool(streaming_callback))
    if streaming_callback:
        qa_llm.callbacks = [streaming_callback]

    chain = GraphCypherQAChain.from_llm(
        llm=qa_llm,
        graph=graph,
        cypher_llm=cypher_llm,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        qa_prompt=QA_GENERATION_PROMPT,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
        input_key="query",
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

    `conversation_id` enables Phase B memory: prior-turn transcript + focus
    entities are used to rewrite elliptical follow-ups; after a successful
    answer, the user/assistant pair and result IDs are persisted.

    `use_cache` enables Phase C answer cache: identical (rewritten) questions
    skip the LLM/Neo4j and replay the stored bundle. Set False to force a
    fresh run.

    Yield types:
      {"type": "token",      "data": "<word>"}                        — QA answer token
      {"type": "rewrite",    "data": "<rewritten question>"}          — only on follow-up rewrite
      {"type": "cypher",     "data": "<cypher>", "results": [...]}    — query + redacted results
      {"type": "cache_hit",  "data": "<key>"}                         — served from cache
      {"type": "end",        "data": ""}                              — pipeline complete
      {"type": "error",      "data": "<message>"}                     — something went wrong
      {"type": "blocked",    "data": "<reason>"}                      — guardrail rejected
    """
    settings = get_settings()

    # ── Phase A: input guardrail ──────────────────────────────────────────
    if settings.guardrails_enabled:
        gate = check_input(question)
        if not gate.ok:
            logger.info(f"Input guardrail blocked: {gate.reason}")
            yield {"type": "blocked", "data": gate.reason}
            yield {"type": "end", "data": ""}
            return
        question = gate.payload

    # ── Phase B: memory + follow-up rewriting ─────────────────────────────
    original_question = question
    session_store = None
    focus_store = None
    if conversation_id and settings.memory_enabled:
        session_store = get_session_store()
        focus_store = get_focus_store()
        transcript = session_store.transcript(conversation_id)
        focus_ids = focus_store.get(conversation_id)
        rewritten = rewrite_question(question, transcript, focus_ids)
        if rewritten != question:
            yield {"type": "rewrite", "data": rewritten}
            question = rewritten

    # ── Phase C: answer cache lookup (after rewrite, before LLM) ──────────
    cache = None
    cache_key = None
    if use_cache and settings.cache_enabled:
        cache = get_answer_cache()
        cache_key = make_cache_key(question, settings.schema_version)
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"Answer cache HIT: {cache_key}")
            yield {"type": "cache_hit", "data": cache_key}
            yield {"type": "token", "data": cached.get("answer", "")}
            yield {
                "type": "cypher",
                "data": cached.get("cypher", ""),
                "results": cached.get("results", []),
            }
            yield {"type": "end", "data": ""}
            # Still update memory on cache hit so follow-ups continue to work.
            if conversation_id and session_store is not None and focus_store is not None:
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

    # ── Phase D: route — cypher (exact/aggregate) or hybrid (similarity) ──
    if settings.hybrid_retriever_enabled and route(question) == Path.HYBRID:
        logger.info(f"Router → HYBRID for: {question[:80]}")
        answer_tokens: list[str] = []
        hybrid_cypher = ""
        hybrid_results: list[dict] = []
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

        # Persist memory + cache for the hybrid path the same way the
        # cypher path does at the bottom of this function.
        try:
            answer_text = "".join(answer_tokens).strip()
            if conversation_id and session_store is not None and focus_store is not None:
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

    queue: asyncio.Queue = asyncio.Queue()
    callback = StreamingCallback(queue)
    chain = build_chain(streaming_callback=callback)

    chain_result: dict = {}
    answer_tokens: list[str] = []  # buffer for persisting the final answer

    async def run_chain():
        try:
            result = await chain.ainvoke({"query": question})
            chain_result["steps"] = result.get("intermediate_steps", [])
            cypher, results = extract_from_steps(chain_result["steps"])

            logger.info(f"Extracted cypher: {cypher[:80] if cypher else 'none'}")
            logger.info(f"Extracted results: {len(results)} rows")

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

        # StreamingCallback fires "end" when the QA LLM finishes token streaming.
        # Skip that intermediate "end" — wait for run_chain's terminal "end"
        # which arrives either with steps populated (success path) or with
        # chain_result["finished"] set (guardrail-blocked path).
        terminal = chain_result.get("steps") or chain_result.get("finished")
        if item["type"] == "end" and not terminal:
            continue

        if item["type"] == "token":
            answer_tokens.append(item["data"])

        yield item

        if item["type"] in ("end", "error"):
            break

    await task

    # ── Phase B + C: persist to memory + cache on success ─────────────────
    if chain_result.get("steps"):
        try:
            cypher, results = extract_from_steps(chain_result["steps"])
            answer_text = "".join(answer_tokens).strip()

            # Memory (Phase B)
            if conversation_id and session_store is not None and focus_store is not None:
                session_store.append_user(conversation_id, original_question)
                if answer_text:
                    session_store.append_assistant(conversation_id, answer_text)
                new_focus = extract_entity_ids(results)
                if new_focus:
                    focus_store.set(conversation_id, new_focus)

            # Answer cache (Phase C) — only cache full successful turns.
            if cache is not None and cache_key and answer_text and cypher:
                cache.set(cache_key, {
                    "question": question,
                    "cypher":   cypher,
                    "results":  results,
                    "answer":   answer_text,
                })
                logger.info(f"Answer cache SET: {cache_key}")
        except Exception as e:
            logger.warning(f"Post-success persist failed: {e}")
