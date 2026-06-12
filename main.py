#!/usr/bin/env python3
"""
Entry point for the Synthea Neo4j Knowledge Graph QA system.

Usage:
  uv run main.py serve          # start WebSocket API server
  uv run main.py ingest         # ingest Synthea data into Neo4j Aura
  uv run main.py ingest --drop  # drop all data then re-ingest
  uv run main.py schema         # create schema only
  uv run main.py stats          # print graph statistics
  uv run main.py ask "question" # one-off question (no streaming)
"""
import sys
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def cmd_serve():
    import uvicorn
    from config import get_settings
    settings = get_settings()
    uvicorn.run(
        "api.websocket_server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


def cmd_ingest(drop: bool = False):
    from ingest.ingestion import run_ingestion
    run_ingestion(drop_first=drop)


def cmd_schema():
    from ingest.schema import create_schema
    create_schema()


def cmd_stats():
    from graph.connection import Neo4jConnection
    queries = {
        "Patients":       "MATCH (n:Patient) RETURN count(n) AS c",
        "Encounters":     "MATCH (n:Encounter) RETURN count(n) AS c",
        "Conditions":     "MATCH (n:Condition) RETURN count(n) AS c",
        "Medications":    "MATCH (n:Medication) RETURN count(n) AS c",
        "Procedures":     "MATCH (n:Procedure) RETURN count(n) AS c",
        "Providers":      "MATCH (n:Provider) RETURN count(n) AS c",
        "Organizations":  "MATCH (n:Organization) RETURN count(n) AS c",
        "Relationships":  "MATCH ()-[r]->() RETURN count(r) AS c",
    }
    print("\n📊 Graph Statistics")
    print("─" * 30)
    for label, q in queries.items():
        result = Neo4jConnection.run_query(q)
        count = result[0]["c"] if result else 0
        print(f"  {label:<20} {count:>8,}")
    print()


async def cmd_ask(question: str):
    from qa.chain import stream_qa_response
    print(f"\n❓ {question}\n")
    cypher_shown = False
    answer = ""
    async for chunk in stream_qa_response(question):
        if chunk["type"] == "cypher" and not cypher_shown:
            print(f"🔍 Cypher:\n{chunk['data']}\n\n💬 Answer:\n", end="", flush=True)
            cypher_shown = True
        elif chunk["type"] == "token":
            print(chunk["data"], end="", flush=True)
            answer += chunk["data"]
        elif chunk["type"] == "end":
            print("\n")
        elif chunk["type"] == "error":
            print(f"\n❌ Error: {chunk['data']}")


def main():
    args = sys.argv[1:]
    if not args or args[0] == "serve":
        cmd_serve()
    elif args[0] == "ingest":
        cmd_ingest(drop="--drop" in args)
    elif args[0] == "schema":
        cmd_schema()
    elif args[0] == "stats":
        cmd_stats()
    elif args[0] == "ask" and len(args) > 1:
        question = " ".join(args[1:])
        asyncio.run(cmd_ask(question))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
