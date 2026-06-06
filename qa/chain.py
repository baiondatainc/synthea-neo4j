"""
LangChain RAG pipeline with streaming and result extraction for charting.

Architecture:
  cypher_llm → text2cypher (Neo4j fine-tuned, Ollama) — generates Cypher only
  qa_llm     → your configured model (Ollama/Anthropic/OpenAI) — explains results

Fixes applied:
  1. import re moved to top level — not inside a function
  2. get_cypher_llm() — falls back gracefully if text2cypher not installed yet
  3. validate_cypher() — catches GROUP BY, missing MATCH, bare aggregation in ORDER BY
  4. run_chain() — clean Neo4j error messages surfaced to user
  5. CYPHER_GENERATION_PROMPT — matches text2cypher fine-tune format + few-shot examples
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
    graph.schema = GRAPH_SCHEMA
    return graph


# ── Cypher specialist LLM ─────────────────────────────────────────────────────

def get_cypher_llm():
    """
    Returns the Neo4j text2cypher fine-tuned model running locally via Ollama.

    One-time setup:
      1. ollama pull gemma2:9b
      2. git clone https://huggingface.co/neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1
      3. Create Modelfile (see project README)
      4. ollama create text2cypher -f Modelfile
      5. ollama run text2cypher "test"   <- verify it responds

    Falls back to the default configured LLM if text2cypher is not yet available
    so the app keeps working while you set up the specialist model.
    """
    settings = get_settings()

    try:
        llm = ChatOllama(
            model="text2cypher",      # Neo4j fine-tuned model registered in Ollama
            base_url=settings.ollama_base_url,
            temperature=0,            # deterministic — Cypher must be exact
            num_predict=512,          # Cypher is short — cap tokens early
        )
        logger.info("Cypher LLM: using Neo4j text2cypher specialist model")
        return llm
    except Exception as e:
        logger.warning(
            f"text2cypher not available ({e}). "
            "Falling back to default LLM for Cypher generation. "
            "To fix: ollama create text2cypher -f Modelfile"
        )
        return get_llm(streaming=False)


# ── Prompts ───────────────────────────────────────────────────────────────────

# Prompt format matches the neo4j/text2cypher-2024v1 fine-tune training template.
# Few-shot examples cover the most common Synthea healthcare queries.
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""Generate Cypher statement to query a graph database.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not include any text except the generated Cypher statement.

STRICT CYPHER RULES:
- GROUP BY does not exist in Cypher — never use it
- Never use aggregation (COUNT, SUM, AVG) directly in ORDER BY
- Always alias aggregations before using in ORDER BY
  WRONG:  ORDER BY COUNT(p) DESC
  CORRECT: RETURN p.gender AS gender, count(p) AS cnt ORDER BY cnt DESC
- Always alias dotted properties
  WRONG:  RETURN p.gender
  CORRECT: RETURN p.gender AS gender
- Output ONLY raw Cypher — no markdown, no backticks, no explanation

FEW-SHOT EXAMPLES:

Q: What is the gender distribution of patients?
MATCH (p:Patient)
RETURN p.gender AS gender, count(p) AS count
ORDER BY count DESC

Q: What are the top 10 most common conditions?
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)
RETURN c.description AS condition, count(p) AS patient_count
ORDER BY patient_count DESC
LIMIT 10

Q: Which medications are most frequently prescribed?
MATCH (p:Patient)-[:PRESCRIBED]->(m:Medication)
RETURN m.description AS medication, count(p) AS frequency
ORDER BY frequency DESC
LIMIT 25

Q: How many patients per encounter class?
MATCH (p:Patient)-[:HAD_ENCOUNTER]->(e:Encounter)
RETURN e.encounterclass AS encounter_class, count(p) AS count
ORDER BY count DESC

Q: Show me the breakdown of encounter types
MATCH (e:Encounter)
RETURN e.encounterclass AS type, count(e) AS count
ORDER BY count DESC

Q: Show me the trend of encounters by year
MATCH (e:Encounter)
WHERE e.start IS NOT NULL
RETURN substring(e.start, 0, 4) AS year, count(e) AS encounter_count
ORDER BY year ASC

Q: Show patients with diabetes
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)
WHERE toLower(c.description) CONTAINS 'diabetes'
RETURN p.id AS patient_id, p.gender AS gender, c.description AS condition
LIMIT 25

Q: Which providers treat the most patients?
MATCH (e:Encounter)-[:SEEN_BY]->(prov:Provider)
RETURN prov.name AS provider, count(e) AS encounter_count
ORDER BY encounter_count DESC
LIMIT 10

Q: What are the most common allergies?
MATCH (p:Patient)-[:ALLERGIC_TO]->(a:Allergen)
RETURN a.description AS allergen, count(p) AS patient_count
ORDER BY patient_count DESC
LIMIT 15

Q: Show me patients prescribed medication for a condition they have
MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)
MATCH (p)-[:PRESCRIBED]->(m:Medication)
RETURN p.id AS patient_id, c.description AS condition, m.description AS medication
LIMIT 25

The question is:
{question}""",
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


# ── Cypher validator ──────────────────────────────────────────────────────────

def validate_cypher(cypher: str) -> str | None:
    """
    Lightweight pre-flight check — catches common LLM mistakes before Neo4j runs the query.
    Returns an error string if the query looks wrong, None if it looks OK.

    Does NOT execute the query — that is Neo4j's job.
    Logs warnings rather than blocking so valid edge cases still pass through.
    """
    if not cypher or not cypher.strip():
        return "Empty Cypher query generated"

    upper = cypher.upper()

    # GROUP BY does not exist in Cypher
    if "GROUP BY" in upper:
        return "Generated Cypher contains GROUP BY — not valid in Cypher"

    # Must have at least one of these keywords
    if not any(kw in upper for kw in ["MATCH", "RETURN", "CALL", "CREATE", "MERGE"]):
        return "Query missing MATCH or RETURN clause"

    # Bare aggregation in ORDER BY without aliasing
    # e.g. ORDER BY COUNT(p) should be ORDER BY cnt
    if re.search(r'ORDER\s+BY\s+\w*COUNT\s*\(', cypher, re.IGNORECASE):
        before_order = upper.split("ORDER")[0]
        if "WITH" not in upper and " AS " not in before_order:
            return "Aggregation used directly in ORDER BY — alias it first with AS"

    return None  # looks OK


# ── Build chain ───────────────────────────────────────────────────────────────

def build_chain(streaming_callback: StreamingCallback = None) -> GraphCypherQAChain:
    graph = get_neo4j_graph()

    # Specialist: text2cypher fine-tuned model — Cypher generation only
    # No streaming needed here — Cypher output is short and must be complete
    cypher_llm = get_cypher_llm()

    # Generalist: your configured model — explains results to the user
    # Streaming enabled so tokens appear progressively in LibreChat
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
    """
    Pulls the generated Cypher query and raw Neo4j results out of
    LangChain's intermediate_steps structure.

    LangChain stores steps as a mixed list of dicts and tuples depending
    on the chain version — both formats handled here.

    Returns: (cypher_string, results_list)
    """
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

async def stream_qa_response(question: str) -> AsyncGenerator[dict, None]:
    """
    Runs the full RAG pipeline and yields events as they complete.

    Yield types:
      {"type": "token",  "data": "<word>"}           — QA answer token
      {"type": "cypher", "data": "<cypher>",
                         "results": [...]}            — query + raw results
      {"type": "end",    "data": ""}                  — pipeline complete
      {"type": "error",  "data": "<message>"}         — something went wrong
    """
    queue: asyncio.Queue = asyncio.Queue()
    callback = StreamingCallback(queue)
    chain = build_chain(streaming_callback=callback)

    # Written by run_chain(), read by the while loop below
    chain_result: dict = {}

    async def run_chain():
        try:
            result = await chain.ainvoke({"query": question})
            chain_result["steps"] = result.get("intermediate_steps", [])
            cypher, results = extract_from_steps(chain_result["steps"])

            logger.info(f"Extracted cypher: {cypher[:80] if cypher else 'none'}")
            logger.info(f"Extracted results: {len(results)} rows, sample: {results[:2]}")

            # Validate — log warning but don't block if Neo4j already ran it
            validation_error = validate_cypher(cypher)
            if validation_error:
                logger.warning(
                    f"Cypher validation warning: {validation_error}\n"
                    f"Query was: {cypher}"
                )

            # Push cypher+results BEFORE end so the consumer has them
            await queue.put({"type": "cypher", "data": cypher, "results": results})
            await queue.put({"type": "end", "data": ""})

        except Exception as e:
            logger.error(f"Chain error: {e}", exc_info=True)

            # Give the user a readable message — not a raw Python traceback
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
        # We ignore that one and wait for run_chain's own "end" which also
        # carries the cypher + results payload.
        if item["type"] == "end" and not chain_result.get("steps"):
            continue

        yield item

        if item["type"] in ("end", "error"):
            break

    await task