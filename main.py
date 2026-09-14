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
  python main.py warm data/faqs/    # pre-populate answer cache from FAQ files
  python main.py warm data/faqs/patient.txt data/faqs/birdeye_review.txt
  python main.py ask "question"     # one-off question (no streaming)
  python main.py clearcache         # wipe the answer cache (all entries)
  python main.py clearcache "Annual payment trend"  # delete one specific entry
"""
import sys
import logging
import asyncio
from glob import glob
from pathlib import Path
from qa.chain import stream_qa_response
from dotenv import load_dotenv
load_dotenv()

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

    # ── Neo4j policy graph node labels ────────────────────────────────
    node_queries = {
        "Policy":             "MATCH (n:Policy) RETURN count(n) AS c",
        "Policyholder":       "MATCH (n:Policyholder) RETURN count(n) AS c",
        "Claim":              "MATCH (n:Claim) RETURN count(n) AS c",
        "HealthMember":       "MATCH (n:HealthMember) RETURN count(n) AS c",
        "Vehicle":            "MATCH (n:Vehicle) RETURN count(n) AS c",
        "Product":            "MATCH (n:Product) RETURN count(n) AS c",
        "Insurer":            "MATCH (n:Insurer) RETURN count(n) AS c",
        "DistributionChannel": "MATCH (n:DistributionChannel) RETURN count(n) AS c",
        "HealthClaimLine":    "MATCH (n:HealthClaimLine) RETURN count(n) AS c",
        "PreAuthorization":   "MATCH (n:PreAuthorization) RETURN count(n) AS c",
        "PolicyEvent":        "MATCH (n:PolicyEvent) RETURN count(n) AS c",
        "Accident":           "MATCH (n:Accident) RETURN count(n) AS c",
        "BenefitClass":       "MATCH (n:BenefitClass) RETURN count(n) AS c",
        "DiagnosisCode":      "MATCH (n:DiagnosisCode) RETURN count(n) AS c",
        "Reinsurer":         "MATCH (n:Reinsurer) RETURN count(n) AS c",
        "DamageAssessment":   "MATCH (n:DamageAssessment) RETURN count(n) AS c",
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
        MATCH (p:Policy)
        RETURN sum(coalesce(p.total_claimed, 0)) AS claimed,
               sum(coalesce(p.total_paid, 0)) AS paid,
               sum(coalesce(p.total_outstanding, 0)) AS outstanding,
               sum(coalesce(p.commission, 0)) AS commission
    """)
    if fin and fin[0]["claimed"]:
        f = fin[0]
        print("\n💰 Policy Financial Summary")
        print("─" * 40)
        print(f"  {'Total Claimed':<20} ${f['claimed'] or 0:>14,.0f}")
        print(f"  {'Total Paid':<20} ${f['paid'] or 0:>14,.0f}")
        print(f"  {'Outstanding':<20} ${f['outstanding'] or 0:>14,.0f}")
        print(f"  {'Commission':<20} ${f['commission'] or 0:>14,.0f}")
    print()


def cmd_vectorize(force: bool = False, limit: int = 1_000_000):
    """Policy-graph placeholder: the embedding pipeline still needs to be rebuilt for the new node model."""
    try:
        from semantic.embeddings import vectorize_patients
        summary = vectorize_patients(force=force, limit=limit)
        print(f"\n✓ Vectorize complete: {summary}\n")
    except Exception as exc:
        print(f"\n⚠ Vectorization is not yet mapped to the policy graph: {exc}\n")


def cmd_clearcache(question: str | None = None):
    """
    Wipe the answer cache.

    No argument  → deletes ALL cached entries.
    With question → deletes only that specific question's cache entry.

    Use after:
      - Rebuilding the ollama model (Modelfile changes)
      - Re-ingesting Neo4j data
      - Fixing a bad cached answer
    """
    from cache import get_answer_cache
    cache = get_answer_cache()

    if question:
        # ── Delete single entry ───────────────────────────────────────
        deleted = cache.delete(question)
        if deleted:
            print(f"\n✓ Cache entry deleted for: {question!r}\n")
        else:
            print(f"\n⚠ No cache entry found for: {question!r}\n")
    else:
        # ── Delete all entries ────────────────────────────────────────
        # Show count before wiping so the user knows what was cleared
        try:
            total = cache.size()          # implement size() if not present
        except AttributeError:
            total = "unknown number of"

        confirm = input(f"\n⚠ This will delete {total} cached answer(s). Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.\n")
            return

        deleted = cache.clear_all()
        print(f"\n✓ Cache cleared — {deleted} entry/entries removed.\n")


async def cmd_warm(paths: list[str]):
    """Pre-populate the answer cache from one or more FAQ files.

    Each file holds one question per line. Lines starting with '#' or blank
    are skipped. Globs (path/*.txt) and directories are expanded.

    Re-runnable: questions already in the cache report as 'cached' and don't
    re-hit the LLM. Errored questions don't block the rest of the run.
    """

    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.txt")))
        elif any(ch in p for ch in "*?["):
            files.extend(Path(f) for f in sorted(glob(p)))
        elif path.exists():
            files.append(path)
        else:
            print(f"⚠ skip (missing): {p}")

    if not files:
        print("No question files found.")
        return

    questions: list[tuple[str, str]] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                q = line.strip()
                if q and not q.startswith("#"):
                    questions.append((f.name, q))

    print(f"\n🔥 Warming {len(questions)} question(s) from {len(files)} file(s)\n")
    warmed = cached = blocked = err = 0

    for i, (src, q) in enumerate(questions, 1):
        print(f"[{i:>3}/{len(questions)}] {src}: {q[:90]}")
        outcome = {"hit": False, "blocked": None, "error": None}
        try:
            async for chunk in stream_qa_response(q, use_cache=True):
                t = chunk["type"]
                if t == "cache_hit":
                    outcome["hit"] = True
                elif t == "blocked":
                    outcome["blocked"] = chunk["data"]
                elif t == "error":
                    outcome["error"] = chunk["data"]
        except Exception as e:
            outcome["error"] = str(e)

        if outcome["error"]:
            err += 1
            print(f"      ✗ error: {outcome['error'][:140]}")
        elif outcome["blocked"]:
            blocked += 1
            print(f"      ⊘ blocked: {outcome['blocked'][:140]}")
        elif outcome["hit"]:
            cached += 1
            print("      ✓ already cached")
        else:
            warmed += 1
            print("      ✓ warmed")

    print()
    print(f"Summary: warmed={warmed}  already_cached={cached}  blocked={blocked}  errors={err}")


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
    elif args[0] == "clearcache":
        # Optional: pass a specific question to delete just that entry
        question = " ".join(args[1:]) if len(args) > 1 else None
        cmd_clearcache(question)
    elif args[0] == "warm" and len(args) > 1:
        asyncio.run(cmd_warm(args[1:]))
    elif args[0] == "ask" and len(args) > 1:
        question = " ".join(args[1:])
        asyncio.run(cmd_ask(question))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()