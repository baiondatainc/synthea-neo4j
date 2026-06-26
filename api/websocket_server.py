"""
FastAPI server for streaming QA over the RP Knowledge Graph.
SGS — HealthGraph AI for RP.

WebSocket protocol (JSON messages):
  Client → Server: {"question": "Which patients have the highest balance?"}
  Server → Client: {"type": "cypher",  "data": "MATCH (p:Patient)...", "results": [...]}
  Server → Client: {"type": "token",   "data": "Based on"}
  Server → Client: {"type": "end",     "data": ""}
  Server → Client: {"type": "error",   "data": "error message"}

Also exposes OpenAI-compatible /v1/chat/completions for LibreChat.

Routes
------
  GET  /health               — Neo4j connectivity check
  GET  /stats                — node + relationship counts
  GET  /sample-questions     — example questions for the UI
  POST /ask                  — one-shot question → cypher + answer + latency
  GET  /catalog              — full data dictionary rendered as HTML
  GET  /catalog.json         — full data dictionary as JSON
  WS   /ws/qa                — streaming QA WebSocket

Changes vs previous version:
  - /ask: passes conversation_id and use_cache into stream_qa_response (was dropped)
  - /ask: captures "rewrite" event and returns rewritten_question in response
  - AskResponse: added rewritten_question field
  - /cache/clear: moved here from openai_compat (single place for cache admin)
"""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from graph.connection import Neo4jConnection
from qa.chain import stream_qa_response
from api.openai_compat import router as openai_router
from api.schema_routes import router as schema_router
from config import get_settings

logger = logging.getLogger(__name__)


# ── Request / response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    use_cache: bool = True          # set False to force a fresh run (eval, debug)


class AskResponse(BaseModel):
    question: str
    conversation_id: str | None = None
    cypher: str | None = None
    answer: str
    latency_s: float
    rewritten_question: str | None = None   # populated when rewriter fires


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting HealthGraph AI QA Server (RP)...")
    Neo4jConnection.get_driver()
    yield
    Neo4jConnection.close()
    logger.info("Server shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HealthGraph AI — RP Knowledge Graph QA",
    description="Streaming QA over the RP knowledge graph",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(openai_router)
app.include_router(schema_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        Neo4jConnection.run_query("RETURN 1 AS ok")
        return {"status": "ok", "neo4j": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.get("/stats")
async def graph_stats():
    """Node and relationship counts for the RP knowledge graph."""
    queries = {
        "patients":      "MATCH (n:Patient) RETURN count(n) AS count",
        "practices":     "MATCH (n:Practice) RETURN count(n) AS count",
        "locations":     "MATCH (n:Location) RETURN count(n) AS count",
        "visits":        "MATCH (n:Visit) RETURN count(n) AS count",
        "charges":       "MATCH (n:Charge) RETURN count(n) AS count",
        "transactions":  "MATCH (n:Transaction) RETURN count(n) AS count",
        "statements":    "MATCH (n:Statement) RETURN count(n) AS count",
        "insurance":     "MATCH (n:InsurancePlan) RETURN count(n) AS count",
        "rc_calls":      "MATCH (n:RCCall) RETURN count(n) AS count",
        "ivr_calls":     "MATCH (n:IVRInbound) RETURN count(n) AS count",
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
    }
    stats = {}
    for key, q in queries.items():
        result = Neo4jConnection.run_query(q)
        stats[key] = result[0]["count"] if result else 0
    return stats


@app.get("/sample-questions")
async def sample_questions():
    return {
        "questions": [
            "Which patients have the highest outstanding balance?",
            "What is the total bad debt by state?",
            "Show me catastrophe patients in Tennessee",
            "Which practices have the most self-pay patients?",
            "What are the most common procedures by modality?",
            "How much was collected through IVR pay-by-phone?",
            "Which insurance carriers cover the most visits?",
            "What is the average charge amount by procedure modality?",
            "Show me patients with bad debt over $5000",
            "Which locations have the lowest Birdeye ratings?",
            "How many multi-practice patients are there?",
            "What is the contractual adjustment total by practice?",
        ]
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    One-shot question → Cypher + answer + server latency.

    Passes conversation_id through to the chain so Phase B memory and
    the rewriter are active. Returns rewritten_question when the rewriter
    expanded an elliptical follow-up.
    """
    t0 = time.perf_counter()
    cypher: str | None = None
    answer = ""
    rewritten_question: str | None = None

    logger.info(
        f"/ask: question={req.question!r:.80} "
        f"conv_id={req.conversation_id!r} "
        f"use_cache={req.use_cache}"
    )

    async for chunk in stream_qa_response(
        req.question,
        conversation_id=req.conversation_id,   # ← was missing; now passed
        use_cache=req.use_cache,               # ← was missing; now passed
    ):
        t = chunk["type"]

        if t == "rewrite":
            rewritten_question = chunk.get("data")
            logger.info(f"/ask: rewriter fired → {rewritten_question!r:.120}")

        elif t == "cypher":
            cypher = chunk.get("data") or None

        elif t == "token":
            answer += chunk.get("data", "")

        elif t == "blocked":
            raise HTTPException(status_code=400, detail=chunk.get("data", "Blocked by guardrail"))

        elif t == "error":
            raise HTTPException(status_code=500, detail=chunk.get("data", "Pipeline error"))

        elif t == "end":
            break

    return AskResponse(
        question=req.question,
        conversation_id=req.conversation_id,
        cypher=cypher,
        answer=answer,
        latency_s=round(time.perf_counter() - t0, 3),
        rewritten_question=rewritten_question,
    )


# ── Cache admin ───────────────────────────────────────────────────────────────

@app.post("/cache/clear")
async def cache_clear():
    """Wipe the answer cache. Use after a schema or ingestion change."""
    from cache import get_answer_cache
    deleted = get_answer_cache().clear_all()
    logger.info(f"Cache cleared: {deleted} keys deleted")
    return {"cleared": deleted}


# ── Catalog routes ────────────────────────────────────────────────────────────

@app.get(
    "/catalog",
    response_class=HTMLResponse,
    summary="Data dictionary — HTML",
    description=(
        "Renders the full RP data dictionary as a browsable HTML page. "
        "Shows all node labels, properties, value maps, relationships, "
        "common Cypher paths, and the token-budget config used at runtime."
    ),
    tags=["Catalog"],
)
async def catalog_html():
    """Full data dictionary rendered as HTML — open in a browser."""
    from metadata.catalog import get_catalog
    html = get_catalog().to_html()
    return HTMLResponse(content=html)


@app.get(
    "/catalog.json",
    summary="Data dictionary — JSON",
    description="Returns the raw data_dictionary.yaml content as JSON.",
    tags=["Catalog"],
)
async def catalog_json():
    """Raw data dictionary as JSON — useful for tooling and debugging."""
    from metadata.catalog import get_catalog
    cat = get_catalog()
    return {
        "labels":         cat.labels,
        "relationships":  cat.relationships,
        "allowed_topics": cat.allowed_topics,
        "token_budget":   cat._doc.get("token_budget", {}),
        "label_keywords": cat._doc.get("label_keywords", {}),
        "path_keywords":  cat._doc.get("path_keywords", {}),
        "common_paths":   cat._doc.get("common_paths", {}),
    }


@app.get(
    "/catalog/schema",
    summary="Focused schema for a question",
    description=(
        "Returns the focused schema block that would be injected into the "
        "Cypher LLM prompt for a given question. "
        "Useful for debugging token budget and keyword matching."
    ),
    tags=["Catalog"],
)
async def catalog_schema(question: str):
    """
    Preview what schema block schema_for_question() produces for this question.

    Query param: ?question=show+me+patients+with+bad+debt
    """
    from metadata.catalog import get_catalog
    cat = get_catalog()
    schema = cat.schema_for_question(question)
    counts = cat.schema_token_count(question)
    return {
        "question":       question,
        "schema":         schema,
        "focused_tokens": counts["focused_tokens"],
        "full_tokens":    counts["full_tokens"],
        "budget_tokens":  cat._char_budget // 4,
        "within_budget":  counts["focused_tokens"] <= cat._char_budget // 4,
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/qa")
async def websocket_qa(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload         = json.loads(raw)
                question        = payload.get("question", "").strip()
                conversation_id = payload.get("conversation_id")
                use_cache       = payload.get("use_cache", False)
            except json.JSONDecodeError:
                question        = raw.strip()
                conversation_id = None
                use_cache       = False

            if not question:
                await websocket.send_json({"type": "error", "data": "Empty question"})
                continue

            logger.info(
                f"WebSocket question: {question!r:.80} "
                f"conv_id={conversation_id!r}"
            )
            await websocket.send_json(
                {"type": "thinking", "data": "Generating Cypher query..."}
            )

            try:
                async for chunk in stream_qa_response(
                    question,
                    conversation_id=conversation_id,
                    use_cache=use_cache,
                ):
                    await websocket.send_json(chunk)
            except Exception as e:
                logger.error(f"Chain error: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "data": str(e)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass