#!/usr/bin/env python3
"""
Entry point for the SGS — RP
Knowledge Graph QA system (HealthGraph AI).

Usage:
  python main.py serve              # start API server (LibreChat-compatible)
  python main.py ingest             # ingest RP parquet data into Neo4j (Bolt)
  python main.py ingest --drop      # drop all data then re-ingest
  python main.py schema             # create constraints + indexes only
  python main.py stats              # print graph statistics
  python main.py vectorize          # embed Patients (Phase D hybrid retriever)
  python main.py vectorize --force  # re-embed every Patient (overwrite existing)
  python main.py vectorize --limit 5000   # embed only the first N
  python main.py ask "question"     # one-off question (no streaming)
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

    # ── RP Knowledge Graph node labels ────────────────────────────────
    node_queries = {
        "Patient":       "MATCH (n:Patient) RETURN count(n) AS c",
        "Practice":      "MATCH (n:Practice) RETURN count(n) AS c",
        "Location":      "MATCH (n:Location) RETURN count(n) AS c",
        "InsurancePlan": "MATCH (n:InsurancePlan) RETURN count(n) AS c",
        "Visit":         "MATCH (n:Visit) RETURN count(n) AS c",
        "Charge":        "MATCH (n:Charge) RETURN count(n) AS c",
        "Transaction":   "MATCH (n:Transaction) RETURN count(n) AS c",
        "Statement":     "MATCH (n:Statement) RETURN count(n) AS c",
        "DiagnosisCode": "MATCH (n:DiagnosisCode) RETURN count(n) AS c",
        "ProcedureCode": "MATCH (n:ProcedureCode) RETURN count(n) AS c",
        "RCCall":        "MATCH (n:RCCall) RETURN count(n) AS c",
        "IVRInbound":    "MATCH (n:IVRInbound) RETURN count(n) AS c",
        "DiallerCall":   "MATCH (n:DiallerCall) RETURN count(n) AS c",
        "PhoneBridge":   "MATCH (n:PhoneBridge) RETURN count(n) AS c",
        "Campaign":      "MATCH (n:Campaign) RETURN count(n) AS c",
        "BirdeyeReview": "MATCH (n:BirdeyeReview) RETURN count(n) AS c",
    }

    print("\n📊 RP Knowledge Graph — Node Counts")
    print("─" * 40)
    total_nodes = 0
    for label, q in node_queries.items():
        result = Neo4jConnection.run_query(q)
        count = result[0]["c"] if result else 0
        total_nodes += count
        print(f"  {label:<20} {count:>10,}")
    print("─" * 40)
    print(f"  {'TOTAL NODES':<20} {total_nodes:>10,}")

    # ── Relationship counts ───────────────────────────────────────────
    rel_result = Neo4jConnection.run_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS c ORDER BY c DESC"
    )
    print("\n📊 Relationship Counts")
    print("─" * 40)
    total_rels = 0
    for row in rel_result:
        total_rels += row["c"]
        print(f"  {row['rel']:<24} {row['c']:>10,}")
    print("─" * 40)
    print(f"  {'TOTAL RELATIONSHIPS':<24} {total_rels:>10,}")

    # ── Financial summary ─────────────────────────────────────────────
    fin = Neo4jConnection.run_query("""
        MATCH (p:Patient)
        RETURN sum(p.total_charged)        AS charged,
               sum(p.total_paid)           AS paid,
               sum(p.outstanding_balance)  AS outstanding,
               sum(p.adj_bad_debt)         AS bad_debt
    """)
    if fin and fin[0]["charged"]:
        f = fin[0]
        print("\n💰 Financial Summary")
        print("─" * 40)
        print(f"  {'Total Charged':<20} ${f['charged'] or 0:>14,.0f}")
        print(f"  {'Total Paid':<20} ${f['paid'] or 0:>14,.0f}")
        print(f"  {'Outstanding':<20} ${f['outstanding'] or 0:>14,.0f}")
        print(f"  {'Bad Debt':<20} ${f['bad_debt'] or 0:>14,.0f}")
    print()


def cmd_vectorize(force: bool = False, limit: int = 1_000_000):
    """Embed every Patient with sentence-transformers + write to Neo4j vector index."""
    from semantic.embeddings import vectorize_patients
    summary = vectorize_patients(force=force, limit=limit)
    print(f"\n✓ Vectorize complete: {summary}\n")


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
    elif args[0] == "vectorize":
        force = "--force" in args
        limit = 1_000_000
        for i, a in enumerate(args):
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
        cmd_vectorize(force=force, limit=limit)
    elif args[0] == "ask" and len(args) > 1:
        question = " ".join(args[1:])
        asyncio.run(cmd_ask(question))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()