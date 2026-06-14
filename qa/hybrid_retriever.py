"""
Runtime hybrid retriever — vector search + 1-hop graph expansion.

Pipeline:
  1. Embed the user question with sentence-transformers
  2. Vector search top-K Patient nodes via Neo4j vector index
  3. 1-hop expansion: pull each candidate's recent visits, charges,
     and behaviour-cohort signals
  4. Build a compact context bundle (already redacted) and stream the
     answer through qa_llm

This is the path the router picks for similarity / behaviour-style
questions ("patients like SAPA:1001", "find similar to this cohort").
For aggregates, the router stays on the Cypher path.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate

from config import get_settings
from graph.connection import Neo4jConnection
from guardrails import redact_rows, redact_text
from qa.llm import get_llm
from semantic.embeddings import embed, PATIENT_INDEX
from semantic.clustering import assign_cohort

logger = logging.getLogger(__name__)


VECTOR_SEARCH_CYPHER = f"""
CALL db.index.vector.queryNodes('{PATIENT_INDEX}', $k, $embedding)
YIELD node AS p, score
RETURN
  coalesce(p.patientId, p.`patientId:ID(Patient)`) AS patient_id,
  p.state                AS state,
  p.city                 AS city,
  p.payor_cohort         AS payor_cohort,
  p.call_tier            AS call_tier,
  p.carrier_name         AS carrier_name,
  p.outstanding_balance  AS outstanding_balance,
  p.adj_bad_debt         AS adj_bad_debt,
  p.is_self_pay          AS is_self_pay,
  p.is_catastrophe       AS is_catastrophe,
  p.is_friction          AS is_friction,
  p.is_clean             AS is_clean,
  p.is_fully_covered     AS is_fully_covered,
  p.has_any_calls        AS has_any_calls,
  score
ORDER BY score DESC
"""


EXPAND_CYPHER = """
MATCH (p:Patient {patientId: $pid})
OPTIONAL MATCH (p)-[:HAD_VISIT]->(v:Visit)
OPTIONAL MATCH (p)-[:HAS_CHARGE]->(c:Charge)
WITH p,
     count(DISTINCT v) AS visit_count,
     count(DISTINCT c) AS charge_count,
     collect(DISTINCT c.procedure_modality)[..5] AS modalities
RETURN visit_count, charge_count,
       [m IN modalities WHERE m IS NOT NULL AND m <> ""] AS modalities
"""


HYBRID_QA_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a helpful healthcare data analyst.

The user asked a similarity / behaviour-style question. Use ONLY the
candidate patients listed below (retrieved by vector similarity from the
RP knowledge graph) to answer. Refer to patients by their patient_id —
do not invent names. If a field is marked [name-redacted] or similar,
preserve that exactly.

Mention 3-5 representative candidates with their distinguishing
attributes (cohort, balance bucket, behaviour flags). Keep the answer
short and concrete.

Question: {question}

Candidates:
{context}

Answer:""",
)


# ── Streaming callback (kept local so the cypher chain's callback can
# stay untouched) ────────────────────────────────────────────────────────────

class _HybridStream(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.queue.put({"type": "token", "data": token})

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        await self.queue.put({"type": "_llm_end", "data": ""})

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        await self.queue.put({"type": "error", "data": str(error)})


# ── Subgraph builders ────────────────────────────────────────────────────────

def _expand(pid: str) -> dict:
    rows = Neo4jConnection.run_query(EXPAND_CYPHER, {"pid": pid})
    if not rows:
        return {"visit_count": 0, "charge_count": 0, "modalities": []}
    return rows[0]


def _format_candidate(row: dict, expansion: dict) -> str:
    """One-line summary of a candidate patient for the qa_llm prompt."""
    bits = [f"patient_id={row['patient_id']}"]
    if row.get("state"):
        bits.append(f"state={row['state']}")
    bits.append(f"cohort={assign_cohort(row)}")
    if row.get("payor_cohort"):
        bits.append(f"payor={row['payor_cohort']}")
    if row.get("call_tier"):
        bits.append(f"call_tier={row['call_tier']}")
    if row.get("carrier_name"):
        bits.append(f"carrier={row['carrier_name']}")
    bal = row.get("outstanding_balance")
    if bal is not None:
        bits.append(f"balance={bal}")
    bd = row.get("adj_bad_debt")
    if bd:
        bits.append(f"bad_debt={bd}")
    if expansion.get("visit_count"):
        bits.append(f"visits={expansion['visit_count']}")
    if expansion.get("charge_count"):
        bits.append(f"charges={expansion['charge_count']}")
    if expansion.get("modalities"):
        bits.append(f"modalities={','.join(expansion['modalities'])}")
    score = row.get("score")
    if score is not None:
        bits.append(f"similarity={float(score):.3f}")
    return "  - " + " | ".join(bits)


def retrieve(question: str, k: int = 10) -> list[dict]:
    """Vector search top-K patients for a question. Already-redacted rows."""
    settings = get_settings()
    vec = embed(question)
    candidates = Neo4jConnection.run_query(
        VECTOR_SEARCH_CYPHER,
        {"k": k, "embedding": vec},
    )
    # Expand + attach
    enriched = []
    for row in candidates:
        pid = row.get("patient_id")
        if not pid:
            continue
        expansion = _expand(pid)
        row["visit_count"] = expansion.get("visit_count")
        row["charge_count"] = expansion.get("charge_count")
        row["modalities"] = expansion.get("modalities")
        row["cohort"] = assign_cohort(row)
        enriched.append(row)
    if settings.guardrails_enabled and settings.guardrails_redact_output:
        enriched = redact_rows(enriched)
    return enriched


# ── Streaming entry point (mirrors qa.chain.stream_qa_response shape) ────────

async def stream_hybrid_response(
    question: str,
    k: int = 10,
) -> AsyncGenerator[dict, None]:
    """Yield events the same way the cypher path does so callers can swap."""
    try:
        candidates = retrieve(question, k=k)
    except Exception as e:
        logger.error(f"Hybrid retrieve failed: {e}", exc_info=True)
        yield {"type": "error", "data": f"Hybrid retrieval failed: {e}"}
        yield {"type": "end", "data": ""}
        return

    if not candidates:
        yield {
            "type": "token",
            "data": (
                "No candidate patients matched. The vector index may be empty — "
                "run `python main.py vectorize` to embed patients first."
            ),
        }
        yield {"type": "cypher", "data": "[hybrid] vector search returned 0 rows", "results": []}
        yield {"type": "end", "data": ""}
        return

    # Build the candidate context block
    candidate_lines = []
    for row in candidates:
        expansion = {
            "visit_count": row.get("visit_count"),
            "charge_count": row.get("charge_count"),
            "modalities": row.get("modalities") or [],
        }
        candidate_lines.append(_format_candidate(row, expansion))
    context_text = "\n".join(candidate_lines)

    # Stream the qa_llm answer
    queue: asyncio.Queue = asyncio.Queue()
    callback = _HybridStream(queue)
    qa_llm = get_llm(streaming=True)
    qa_llm.callbacks = [callback]
    prompt = HYBRID_QA_PROMPT.format(question=question, context=context_text)

    async def run_llm():
        try:
            await qa_llm.ainvoke(prompt)
        except Exception as e:
            logger.error(f"Hybrid LLM failed: {e}", exc_info=True)
            await queue.put({"type": "error", "data": str(e)})
        await queue.put({"type": "_llm_end", "data": ""})

    task = asyncio.create_task(run_llm())

    while True:
        item = await queue.get()
        if item["type"] == "_llm_end":
            break
        if item["type"] == "error":
            yield item
            yield {"type": "end", "data": ""}
            await task
            return
        yield item

    await task

    # Emit a synthetic "cypher" event so downstream artifact/charting code
    # still has structured data to consume.
    yield {
        "type": "cypher",
        "data": f"[hybrid] vector search on {PATIENT_INDEX} k={k} + 1-hop expansion",
        "results": candidates,
    }
    yield {"type": "end", "data": ""}
