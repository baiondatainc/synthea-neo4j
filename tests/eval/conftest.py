"""Session fixtures + report writer for the Cypher-generation eval.

Pass criterion (asserted): the Cypher LLM produces non-empty Cypher.
Additional metrics recorded but not gating: guardrail pass, Neo4j EXPLAIN pass.
A JSON + Markdown report is written under tests/eval/reports/ on session end.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest

# Make the project root importable so we can reuse qa/graph/metadata code.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── CLI options ──────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--eval-limit",
        type=int,
        default=0,
        help="Run only the first N eval questions (0 = all). Useful for iteration.",
    )
    parser.addoption(
        "--eval-difficulty",
        type=str,
        default="",
        help="Comma-separated subset of difficulties to run (E,M,H).",
    )
    parser.addoption(
        "--eval-no-execute",
        action="store_true",
        default=False,
        help="Skip the Neo4j EXPLAIN check (still parses+generates Cypher).",
    )
    parser.addoption(
        "--eval-url",
        type=str,
        default="",
        help="Base URL of the running HealthGraph API (e.g. http://localhost:8001). "
             "When set, eval uses HTTP /ask instead of calling the LLM directly.",
    )


# ── Strip a fenced code block / 'Cypher:' label out of LLM output ────────────

_FENCE_RE = re.compile(r"```(?:cypher|cql)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_LABEL_RE = re.compile(r"^\s*(?:cypher|query)\s*:\s*", re.IGNORECASE)


def extract_cypher(text: str) -> str:
    if not text:
        return ""
    # Prefer the first fenced block, otherwise treat the whole response as Cypher.
    m = _FENCE_RE.search(text)
    body = m.group(1) if m else text
    body = _LABEL_RE.sub("", body, count=1)
    return body.strip()


# ── Cypher LLM + schema text (no Neo4j connection required) ──────────────────

@pytest.fixture(scope="session")
def schema_text() -> str:
    """The same schema string the production chain feeds to the Cypher LLM."""
    from graph.schema_text import GRAPH_SCHEMA
    from metadata.catalog import get_catalog
    return f"{GRAPH_SCHEMA}\n\n{get_catalog().schema_addendum()}"


@pytest.fixture(scope="session")
def cypher_llm():
    from qa.chain import get_cypher_llm
    return get_cypher_llm()


@pytest.fixture(scope="session")
def cypher_prompt():
    from qa.chain import CYPHER_GENERATION_PROMPT
    return CYPHER_GENERATION_PROMPT


@pytest.fixture(scope="session")
def generate_cypher(request, cypher_llm, cypher_prompt):
    base_url = request.config.getoption("--eval-url", default="").rstrip("/")

    if base_url:
        # ── HTTP path — hits the running API server ───────────────────────────
        import httpx
        client = httpx.Client(base_url=base_url, timeout=60.0)

        def _gen_http(question: str) -> str:
            try:
                resp = client.post("/ask", json={"question": question})
                resp.raise_for_status()
                data = resp.json()
                return data.get("cypher", "")
            except Exception as e:
                print(f"\n[eval] Error generating Cypher for question '{question}': {e}", file=sys.stderr)                
                return ""

        yield _gen_http
        client.close()

    else:
        # ── Direct LLM path — existing behaviour, no server needed ────────────
        from metadata.catalog import get_catalog
        from graph.schema_text import GRAPH_SCHEMA
        catalog = get_catalog()

        def _gen_direct(question: str) -> str:
            focused_schema = f"{GRAPH_SCHEMA}\n\n{catalog.schema_for_question(question)}"
            prompt = cypher_prompt.format(schema=focused_schema, question=question)
            resp = cypher_llm.invoke(prompt)
            content = getattr(resp, "content", None) or str(resp)
            return extract_cypher(content)

        yield _gen_direct


# ── Neo4j session for EXPLAIN-based executability check ──────────────────────

@pytest.fixture(scope="session")
def neo4j_driver(request):
    """Returns a neo4j.Driver, or None if --eval-no-execute or connection fails.

    We don't fail the session if Neo4j is unreachable — the eval still measures
    Cypher generation. Executable rate is reported as 'not tested' instead.
    """
    if request.config.getoption("--eval-no-execute"):
        return None
    try:
        from neo4j import GraphDatabase
        from config import get_settings
        s = get_settings()
        drv = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_username, s.neo4j_password))
        drv.verify_connectivity()
    except Exception as e:
        print(f"\n[eval] Neo4j unavailable — skipping EXPLAIN checks ({e})", file=sys.stderr)
        return None
    yield drv
    try:
        drv.close()
    except Exception:
        pass


# ── Metrics collector + report ───────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, rec: dict) -> None:
        self.records.append(rec)


@pytest.fixture(scope="session")
def metrics() -> Metrics:
    return Metrics()


def _summarize(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    def pct(count: int) -> float:
        return round(100.0 * count / n, 1)

    by_diff: dict[str, dict] = defaultdict(lambda: {"total": 0, "gen": 0, "guard": 0, "exec": 0})
    by_cat: dict[str, dict] = defaultdict(lambda: {"total": 0, "gen": 0, "guard": 0, "exec": 0})

    for r in records:
        for bucket in (by_diff[r["difficulty"]], by_cat[r["category"]]):
            bucket["total"] += 1
            bucket["gen"] += int(bool(r["cypher_generated"]))
            bucket["guard"] += int(bool(r["guardrail_ok"]))
            bucket["exec"] += int(bool(r.get("executable")))

    gen = sum(1 for r in records if r["cypher_generated"])
    guard = sum(1 for r in records if r["guardrail_ok"])
    execed = sum(1 for r in records if r.get("executable"))
    exec_tested = sum(1 for r in records if r.get("execution_tested"))

    return {
        "n": n,
        "cypher_generated_rate": pct(gen),
        "guardrail_pass_rate": pct(guard),
        "executable_rate": (round(100.0 * execed / exec_tested, 1) if exec_tested else None),
        "execute_tested": exec_tested,
        "by_difficulty": dict(by_diff),
        "by_category": dict(by_cat),
    }


def _write_markdown(path: Path, summary: dict, records: list[dict]) -> None:
    lines: list[str] = []
    lines.append(f"# Cypher Eval Report — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"**Questions:** {summary.get('n', 0)}")
    lines.append(f"- Cypher generated: **{summary.get('cypher_generated_rate', 0)}%**")
    lines.append(f"- Guardrail-passing: **{summary.get('guardrail_pass_rate', 0)}%**")
    exec_rate = summary.get("executable_rate")
    if exec_rate is None:
        lines.append("- Executable (EXPLAIN): _not tested — Neo4j unavailable or --eval-no-execute_")
    else:
        lines.append(
            f"- Executable (EXPLAIN): **{exec_rate}%** "
            f"({summary['execute_tested']} tested)"
        )
    lines.append("## Generated Cypher queries")
    lines.append("")
    lines.append("All questions with their generated Cypher, in order.")
    lines.append("")
    for r in records:
        if r.get("error"):
            lines.append(f"> ⚠️ Error: `{r['error']}`")
     
        if r.get("execution_error"):
            lines.append(f"> ⚠️ EXPLAIN error: `{r['execution_error']}`")
        lines.append("")
        
        status = "✅" if r["cypher_generated"] else "❌"
        exec_status = ""

        latency = r.get("latency_s")
        latency_str = f" · ⏱ {latency}s" if latency is not None else ""


        lines.append(f"### Q{r['number']} [{r['difficulty']}] {status}{exec_status}{latency_str} - **{r['question']}**")
        lines.append("")
        cypher = (r.get("cypher") or "").strip()
        if cypher:
            lines.append("```cypher")
            lines.append(cypher)
            lines.append("```")
        else:
            lines.append("_No Cypher generated._")


    def table(title: str, buckets: dict):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Bucket | N | Generated | Guardrail | Executable |")
        lines.append("|---|---:|---:|---:|---:|")
        for k, v in sorted(buckets.items()):
            total = v["total"] or 1
            lines.append(
                f"| {k} | {v['total']} | "
                f"{round(100 * v['gen'] / total, 1)}% | "
                f"{round(100 * v['guard'] / total, 1)}% | "
                f"{round(100 * v['exec'] / total, 1)}% |"
            )
        lines.append("")

    table("By difficulty", summary.get("by_difficulty", {}))
    table("By category", summary.get("by_category", {}))

    failures = [r for r in records if not r["cypher_generated"]]
    if failures:
        lines.append("## Generation failures")
        lines.append("")
        for r in failures:
            lines.append(f"- **Q{r['number']}** [{r['difficulty']}] {r['question']}")
            if r.get("error"):
                lines.append(f"  - error: `{r['error']}`")
        lines.append("")

    blocked = [r for r in records if r["cypher_generated"] and not r["guardrail_ok"]]
    if blocked:
        lines.append("## Guardrail blocks (Cypher generated but rejected)")
        lines.append("")
        for r in blocked:
            lines.append(f"- **Q{r['number']}** [{r['difficulty']}] {r['question']}")
            lines.append(f"  - reason: {r.get('guardrail_reason', '')}")
            lines.append(f"  - cypher: `{(r.get('cypher') or '').splitlines()[0][:140]}`")
        lines.append("")

    inexec = [
        r for r in records
        if r.get("execution_tested") and not r.get("executable")
    ]
    if inexec:
        lines.append("## Not executable on Neo4j (EXPLAIN failed)")
        lines.append("")
        for r in inexec:
            lines.append(f"- **Q{r['number']}** [{r['difficulty']}] {r['question']}")
            lines.append(f"  - error: `{(r.get('execution_error') or '').splitlines()[0][:200]}`")
        lines.append("")

    path.write_text("\n".join(lines))


def pytest_sessionfinish(session, exitstatus):
    metrics: Metrics | None = getattr(session, "_eval_metrics", None)
    if metrics is None or not metrics.records:
        return

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_{stamp}.json"
    md_path = out_dir / f"eval_{stamp}.md"

    summary = _summarize(metrics.records)
    json_path.write_text(json.dumps(
        {"summary": summary, "records": metrics.records}, indent=2, default=str
    ))
    _write_markdown(md_path, summary, metrics.records)

    # Update "latest" pointers for convenience.
    for name, src in (("latest.json", json_path), ("latest.md", md_path)):
        link = out_dir / name
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.name)
        except OSError:
            # Fallback for filesystems without symlinks.
            link.write_text(src.read_text())

    n = summary.get("n", 0)
    print(f"\n[eval] Wrote report: {md_path} ({n} questions)", file=sys.stderr)


# Stash the Metrics instance on the session so sessionfinish can find it.
@pytest.fixture(autouse=True)
def _stash_metrics(request, metrics):
    request.session._eval_metrics = metrics
    yield

# ─────────────────────────────────────────────────────────────────────────────
# ADD THESE TO YOUR EXISTING conftest.py
# ─────────────────────────────────────────────────────────────────────────────

# 1. New fixture: generate_cypher_with_conv
#    Drop this alongside the existing generate_cypher fixture.
#    It sends an extra conversation_id so the memory/rewriter layer fires.

@pytest.fixture(scope="session")
def generate_cypher_with_conv(request, cypher_llm, cypher_prompt):
    """
    Like generate_cypher but accepts a conversation_id so the full
    memory + rewriter pipeline runs. Returns a dict:
      {
        "cypher":             str,
        "rewritten_question": str | None,   # set if rewriter fired
        "answer":             str,
        "latency_s":          float,
      }

    HTTP mode (--eval-url): POST /ask with {"question": ..., "conversation_id": ...}
    Direct mode: calls stream_qa_response directly (requires running event loop).
    """
    base_url = request.config.getoption("--eval-url", default="").rstrip("/")

    if base_url:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=90.0)

        def _gen_http(question: str, conversation_id: str | None = None) -> dict:
            try:
                payload = {"question": question}
                if conversation_id:
                    payload["conversation_id"] = conversation_id
                resp = client.post("/ask", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "cypher":             data.get("cypher", ""),
                    "rewritten_question": data.get("rewritten_question"),
                    "answer":             data.get("answer", ""),
                    "latency_s":          data.get("latency_s"),
                }
            except Exception as exc:
                import sys
                print(f"\n[eval] HTTP error for {question!r}: {exc}", file=sys.stderr)
                return {"cypher": "", "rewritten_question": None, "answer": "", "latency_s": None}

        yield _gen_http
        client.close()

    else:
        # Direct path: run stream_qa_response in a fresh event loop per call.
        import asyncio
        from qa.chain import stream_qa_response

        def _gen_direct(question: str, conversation_id: str | None = None) -> dict:
            cypher = ""
            rewritten = None
            answer_parts: list[str] = []

            async def _run():
                nonlocal cypher, rewritten
                async for ev in stream_qa_response(
                    question,
                    conversation_id=conversation_id,
                    use_cache=False,   # always fresh during eval
                ):
                    if ev["type"] == "rewrite":
                        rewritten = ev.get("data")
                    elif ev["type"] == "token":
                        answer_parts.append(ev.get("data", ""))
                    elif ev["type"] == "cypher":
                        cypher = ev.get("data", "")

            asyncio.run(_run())
            return {
                "cypher":             cypher,
                "rewritten_question": rewritten,
                "answer":             "".join(answer_parts),
                "latency_s":          None,
            }

        yield _gen_direct


# ─────────────────────────────────────────────────────────────────────────────
# 2. Update _write_markdown to include a context chain section.
#    Find your existing _write_markdown function and add this block
#    AFTER the "Generated Cypher queries" section for structural questions.
# ─────────────────────────────────────────────────────────────────────────────

def _context_chain_section(records: list[dict]) -> list[str]:
    """
    Build the context chain section for the markdown report.
    Call this from _write_markdown after the structural Cypher section.
    """
    chain_records = [r for r in records if r.get("thread_id")]
    if not chain_records:
        return []

    lines: list[str] = []
    lines.append("---")
    lines.append("")
    lines.append("# Context chain eval")
    lines.append("")

    # Summary table
    threads_seen = {}
    for r in chain_records:
        tid = r["thread_id"]
        if tid not in threads_seen:
            threads_seen[tid] = {"label": r["thread_label"], "turns": [], "pass": True}
        threads_seen[tid]["turns"].append(r)
        if not r["cypher_generated"]:
            threads_seen[tid]["pass"] = False

    total_threads = len(threads_seen)
    passed_threads = sum(1 for v in threads_seen.values() if v["pass"])
    total_turns = len(chain_records)
    passed_turns = sum(1 for r in chain_records if r["cypher_generated"])
    rewrites = sum(1 for r in chain_records if r.get("rewritten"))
    rewrite_hint_fails = sum(
        1 for r in chain_records
        if r.get("rewritten") and not r.get("rewrite_hint_ok", True)
    )

    lines.append(f"**Threads:** {total_threads}  |  "
                 f"**Thread pass rate:** {passed_threads}/{total_threads}  |  "
                 f"**Turn pass rate:** {passed_turns}/{total_turns}  |  "
                 f"**Rewrites fired:** {rewrites}  |  "
                 f"**Rewrite hint mismatches:** {rewrite_hint_fails}")
    lines.append("")

    # Thread summary table
    lines.append("| Thread | Label | Turns | Pass | Rewrites |")
    lines.append("|---|---|---|---|---|")
    for tid, info in threads_seen.items():
        n = len(info["turns"])
        p = sum(1 for r in info["turns"] if r["cypher_generated"])
        rw = sum(1 for r in info["turns"] if r.get("rewritten"))
        status = "✅" if info["pass"] else "❌"
        lines.append(f"| {tid} | {info['label']} | {p}/{n} | {status} | {rw} |")
    lines.append("")

    # Per-thread detail
    lines.append("## Thread detail")
    lines.append("")

    for tid, info in threads_seen.items():
        lines.append(f"### {tid} — {info['label']}")
        lines.append("")
        for r in info["turns"]:
            turn = r["turn_index"]
            gen_ok = r["cypher_generated"]
            status = "✅" if gen_ok else "❌"
            latency = r.get("latency_s")
            latency_str = f" · ⏱ {latency}s" if latency is not None else ""
            exec_str = ""
            if r.get("execution_tested"):
                exec_str = " · EXPLAIN ✅" if r.get("executable") else " · EXPLAIN ❌"

            lines.append(f"#### Turn {turn} [{r['difficulty']}] {status}{exec_str}{latency_str}")
            lines.append(f"**Original:** {r['question']}")
            lines.append("")

            if r.get("rewritten"):
                hint_ok = r.get("rewrite_hint_ok", True)
                hint_flag = "" if hint_ok else " ⚠️ hint mismatch"
                lines.append(f"> **Rewritten:** {r['rewritten_text']}{hint_flag}")
                lines.append("")

            cypher = (r.get("cypher") or "").strip()
            if cypher:
                lines.append("```cypher")
                lines.append(cypher)
                lines.append("```")
            else:
                lines.append("_No Cypher generated._")

            if r.get("error"):
                lines.append(f"> ⚠️ Error: `{r['error']}`")
            if r.get("execution_error"):
                lines.append(f"> ⚠️ EXPLAIN error: `{r['execution_error']}`")
            if r.get("guardrail_reason"):
                lines.append(f"> ⚠️ Guardrail: `{r['guardrail_reason']}`")
            lines.append("")

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 3. Also update /ask endpoint in api/openai_compat.py (or websocket_server.py)
#    to return rewritten_question so the HTTP eval path captures it.
#    Add this field to the AskResponse and populate it from the "rewrite" event.
# ─────────────────────────────────────────────────────────────────────────────

ASK_ENDPOINT_PATCH = """
# In your /ask endpoint, add rewritten_question to the response:

class AskResponse(BaseModel):
    question: str
    cypher: str | None
    answer: str
    latency_s: float
    rewritten_question: str | None = None   # ← add this

@app.post("/ask")
async def ask(req: AskRequest):
    import time
    t0 = time.perf_counter()
    cypher = None
    answer = ""
    rewritten_question = None

    async for chunk in stream_qa_response(
        req.question,
        conversation_id=getattr(req, "conversation_id", None),
        use_cache=False,
    ):
        if chunk["type"] == "rewrite":
            rewritten_question = chunk["data"]   # ← capture rewrite event
        elif chunk["type"] == "cypher":
            cypher = chunk["data"]
        elif chunk["type"] == "token":
            answer += chunk["data"]
        elif chunk["type"] == "error":
            raise HTTPException(status_code=500, detail=chunk["data"])

    return AskResponse(
        question=req.question,
        cypher=cypher or "",
        answer=answer,
        latency_s=round(time.perf_counter() - t0, 3),
        rewritten_question=rewritten_question,   # ← include it
    )
"""