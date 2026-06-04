"""
LangChain RAG pipeline:
1. User question → LLM generates Cypher
2. Cypher runs against Neo4j Aura
3. Results + question → LLM generates natural language answer
4. Stream tokens back via callback
"""
import logging
import asyncio
from typing import AsyncGenerator, Any

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from config import get_settings
from graph.schema_text import GRAPH_SCHEMA
from qa.llm import get_llm

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


# ── Neo4j Graph ───────────────────────────────────────────────────────────────

def get_neo4j_graph() -> Neo4jGraph:
    settings = get_settings()
    graph = Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        enhanced_schema=False,
    )
    graph.refresh_schema()
    graph.schema = GRAPH_SCHEMA   # override with curated schema
    return graph


# ── Prompts ───────────────────────────────────────────────────────────────────
# GraphCypherQAChain requires EXACTLY these input_variables

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are a Neo4j Cypher expert for a healthcare knowledge graph.
Use ONLY the schema below. Do not invent labels or relationship types.

Schema:
{schema}

Rules:
- Output ONLY the raw Cypher query — no markdown, no backticks, no explanation
- Always add LIMIT 25 unless the query is a pure count/aggregation
- Use toLower() for .description text matching
- Return human-readable property names, not just IDs

Question: {question}
Cypher Query:""",
)

QA_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a helpful healthcare data analyst.

Given these graph query results, provide a clear plain-English answer.
Include: a direct answer, key numbers/patterns, and any notable insights.

Question: {question}
Graph Results: {context}

Answer:""",
)


# ── Build chain ───────────────────────────────────────────────────────────────

def build_chain(streaming_callback: StreamingCallback = None) -> GraphCypherQAChain:
    graph = get_neo4j_graph()
    cypher_llm = get_llm(streaming=False)
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


# ── Streaming generator ───────────────────────────────────────────────────────

async def stream_qa_response(question: str) -> AsyncGenerator[dict, None]:
    """
    Yields:
      {"type": "cypher",  "data": "<cypher>"}
      {"type": "token",   "data": "<word>"}
      {"type": "end",     "data": ""}
      {"type": "error",   "data": "<msg>"}
    """
    queue: asyncio.Queue = asyncio.Queue()
    callback = StreamingCallback(queue)
    chain = build_chain(streaming_callback=callback)
    cypher_bucket: list[str] = []

    async def run_chain():
        try:
            result = await chain.ainvoke({"query": question})

            # Extract Cypher from intermediate_steps
            cypher = ""
            for step in result.get("intermediate_steps", []):
                if isinstance(step, dict) and "query" in step:
                    cypher = step["query"]
                    break
                if isinstance(step, (list, tuple)):
                    for sub in step:
                        if isinstance(sub, dict) and "query" in sub:
                            cypher = sub["query"]
                            break

            cypher_bucket.append(cypher)
            await queue.put({"type": "cypher", "data": cypher})

        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)
            await queue.put({"type": "error", "data": str(e)})

    task = asyncio.create_task(run_chain())

    while True:
        item = await queue.get()
        yield item
        if item["type"] in ("end", "error"):
            break

    await task
