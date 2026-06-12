"""
FastAPI server for streaming QA over the RP Knowledge Graph.
Sutherland Global Services — HealthGraph AI for Radiology Partners.

WebSocket protocol (JSON messages):
  Client → Server: {"question": "Which patients have the highest balance?"}
  Server → Client: {"type": "cypher",  "data": "MATCH (p:Patient)...", "results": [...]}
  Server → Client: {"type": "token",   "data": "Based on"}
  Server → Client: {"type": "end",     "data": ""}
  Server → Client: {"type": "error",   "data": "error message"}

Also exposes OpenAI-compatible /v1/chat/completions for LibreChat.
"""
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from graph.connection import Neo4jConnection
from qa.chain import stream_qa_response
from api.openai_compat import router as openai_router
from config import get_settings

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting HealthGraph AI QA Server (Radiology Partners)...")
    Neo4jConnection.get_driver()   # warm up connection pool
    yield
    Neo4jConnection.close()
    logger.info("Server shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HealthGraph AI — RP Knowledge Graph QA",
    description="Streaming QA over the Radiology Partners knowledge graph",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(openai_router)

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
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.get("/stats")
async def graph_stats():
    """Return node and relationship counts for the RP knowledge graph."""
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


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/qa")
async def websocket_qa(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                question = payload.get("question", "").strip()
            except json.JSONDecodeError:
                question = raw.strip()

            if not question:
                await websocket.send_json({"type": "error", "data": "Empty question"})
                continue

            logger.info(f"Question: {question}")
            await websocket.send_json({"type": "thinking", "data": "Generating Cypher query..."})

            try:
                async for chunk in stream_qa_response(question):
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