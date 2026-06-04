"""
FastAPI WebSocket server for streaming QA.

WebSocket protocol (JSON messages):
  Client → Server: {"question": "Which patients have diabetes?"}
  Server → Client: {"type": "cypher",  "data": "MATCH (p:Patient)..."}
  Server → Client: {"type": "token",   "data": "Based on"}
  Server → Client: {"type": "token",   "data": " the results..."}
  Server → Client: {"type": "end",     "data": ""}
  Server → Client: {"type": "error",   "data": "error message"}
  Server → Client: {"type": "stats",   "data": {"nodes": 1000, ...}}
"""
import json
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from graph.connection import Neo4jConnection
from ingest.schema import create_schema
from qa.chain import stream_qa_response
from config import get_settings

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Synthea-Neo4j QA Server...")
    Neo4jConnection.get_driver()   # warm up connection pool
    yield
    Neo4jConnection.close()
    logger.info("Server shut down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Synthea Neo4j Knowledge Graph QA",
    description="WebSocket-based streaming QA over a healthcare knowledge graph",
    version="1.0.0",
    lifespan=lifespan,
)

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
    """Return node and relationship counts."""
    queries = {
        "patients":      "MATCH (n:Patient) RETURN count(n) AS count",
        "encounters":    "MATCH (n:Encounter) RETURN count(n) AS count",
        "conditions":    "MATCH (n:Condition) RETURN count(n) AS count",
        "medications":   "MATCH (n:Medication) RETURN count(n) AS count",
        "procedures":    "MATCH (n:Procedure) RETURN count(n) AS count",
        "providers":     "MATCH (n:Provider) RETURN count(n) AS count",
        "organizations": "MATCH (n:Organization) RETURN count(n) AS count",
        "relationships": "MATCH ()-[r]->() RETURN count(r) AS count",
    }
    stats = {}
    for key, q in queries.items():
        result = Neo4jConnection.run_query(q)
        stats[key] = result[0]["count"] if result else 0
    return stats


@app.post("/ingest")
async def trigger_ingestion(drop_first: bool = False):
    """Trigger data ingestion (runs in background)."""
    from ingest.ingestion import run_ingestion
    asyncio.create_task(
        asyncio.to_thread(run_ingestion, drop_first=drop_first)
    )
    return {"status": "ingestion started", "drop_first": drop_first}


@app.get("/sample-questions")
async def sample_questions():
    return {
        "questions": [
            "Which patients have diabetes?",
            "What are the most common conditions in the dataset?",
            "Which medications are most frequently prescribed?",
            "Show me patients who have both hypertension and diabetes",
            "What is the average cost of emergency encounters?",
            "Which providers have seen the most patients?",
            "What conditions are most common in female patients over 60?",
            "Show me the top 10 most expensive encounters",
            "What procedures are performed most often for diabetic patients?",
            "How many patients have been hospitalized (inpatient encounters)?",
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
