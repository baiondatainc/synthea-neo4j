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
