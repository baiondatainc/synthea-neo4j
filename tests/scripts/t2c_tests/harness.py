"""Shared helpers for the text2cypher regression suite.

Backends (T2C_MODE):
  ollama  - call the models directly: text2cypher model -> Cypher -> Neo4j -> narrator model -> answer
  api     - POST to your chat application; adapt `ask_api()` if its contract differs
"""
from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml
from neo4j import GraphDatabase

ROOT = Path(__file__).parent
QUESTIONS_FILE = ROOT / "questions.yaml"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# .env loading - suite folder first, then parents. Real env vars always win.
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    for folder in [ROOT, *ROOT.parents]:
        env_file = folder / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


_load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # backend: "ollama" = call the models directly, "api" = your chat app
    mode: str = os.getenv("T2C_MODE", "ollama")

    # ollama mode
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    t2c_model: str = os.getenv("T2C_MODEL", "text2cypher-ia:latest")
    narrator_model: str = os.getenv("NARRATOR_MODEL", "")          # e.g. llama3.1:8b; empty = skip narration

    # api mode
    api_url: str = os.getenv("T2C_API_URL", "http://localhost:8001/ask")
    api_key: str = os.getenv("T2C_API_KEY", "")
    field_cypher: str = os.getenv("T2C_FIELD_CYPHER", "cypher")
    field_rows: str = os.getenv("T2C_FIELD_ROWS", "rows")
    field_answer: str = os.getenv("T2C_FIELD_ANSWER", "answer")

    api_timeout: int = int(os.getenv("T2C_API_TIMEOUT", "180"))

    # neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")   # keep the real one in .env
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")


CFG = Config()


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------
@dataclass
class Question:
    id: str
    category: str
    question: str
    compare: str
    expected_cypher: str
    tolerance: float = 0.01
    round_to: int = 2
    topk: int = 10
    note: str = ""
    raw: dict = field(default_factory=dict)


_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def _fill(text: str, values: dict[str, Any]) -> str:
    def rep(m):
        k = m.group(1)
        if k not in values:
            raise KeyError(f"placeholder ${{{k}}} not defined in questions.yaml values")
        return str(values[k])
    return _PLACEHOLDER.sub(rep, text)


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = doc.get("values", {})
    defaults = doc.get("defaults", {})
    out = []
    for q in doc["questions"]:
        out.append(Question(
            id=q["id"],
            category=q.get("category", "misc"),
            question=_fill(q["question"], values),
            compare=q.get("compare", "rows"),
            expected_cypher=_fill(q["expected_cypher"], values).strip(),
            tolerance=float(q.get("tolerance", defaults.get("tolerance", 0.01))),
            round_to=int(q.get("round_to", defaults.get("round_to", 2))),
            topk=int(q.get("topk", 10)),
            note=q.get("note", ""),
            raw=q,
        ))
    return out


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, cfg: Config = CFG):
        self.driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        self.db = cfg.neo4j_database

    def verify(self) -> None:
        self.driver.verify_connectivity()
        self.run("RETURN 1 AS ok")

    def run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session(database=self.db) as s:
            return [r.data() for r in s.run(cypher, **params)]

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
@dataclass
class ChatResult:
    cypher: str | None
    rows: list[dict] | None
    answer: str | None
    raw: Any
    latency_s: float


_FENCE = re.compile(r"```(?:cypher)?\s*(.*?)```", re.S | re.I)


def _clean_cypher(text: str) -> str:
    """Strip markdown fences / trailing semicolon / chatter around the query."""
    m = _FENCE.search(text)
    c = m.group(1) if m else text
    c = c.strip()
    # if the model prefixed prose, keep from the first Cypher keyword
    kw = re.search(r"\b(MATCH|CALL|WITH|UNWIND|RETURN|OPTIONAL)\b", c, re.I)
    if kw and kw.start() > 0:
        c = c[kw.start():]
    return c.rstrip(";").strip()


def _ollama_generate(model: str, prompt: str, cfg: Config) -> dict:
    r = requests.post(
        f"{cfg.ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "keep_alive": -1},
        timeout=cfg.api_timeout,
    )
    r.raise_for_status()
    return r.json()


def warm_up(cfg: Config = CFG) -> None:
    """Load models + prefill system prompt once so Q01 latency is not the cold start."""
    if cfg.mode != "ollama":
        return
    _ollama_generate(cfg.t2c_model, "How many policies?", cfg)
    if cfg.narrator_model:
        _ollama_generate(cfg.narrator_model, "Say OK.", cfg)


def ask_ollama(question: str, graph: Graph, cfg: Config = CFG) -> ChatResult:
    """text2cypher model -> Cypher -> Neo4j -> (optional) narrator model -> answer."""
    t0 = time.perf_counter()
    gen = _ollama_generate(cfg.t2c_model, question, cfg)
    cypher = _clean_cypher(gen.get("response", ""))

    rows: list[dict] | None = None
    exec_error: str | None = None
    try:
        rows = graph.run(cypher)
    except Exception as e:  # noqa: BLE001
        exec_error = str(e)

    answer = None
    narr_stats: dict = {}
    if cfg.narrator_model and rows is not None:
        narr_prompt = (
            f"Question: {question}\n"
            f"Query results (JSON, first 20 rows): {json.dumps(rows[:20], default=str)}\n"
            f"Answer the question for a business user using only these results."
        )
        narr = _ollama_generate(cfg.narrator_model, narr_prompt, cfg)
        answer = narr.get("response")
        narr_stats = {k: narr.get(k) for k in ("prompt_eval_count", "eval_count", "eval_duration", "total_duration")}

    return ChatResult(
        cypher=cypher, rows=rows, answer=answer,
        raw={
            "t2c": {k: gen.get(k) for k in ("prompt_eval_count", "eval_count", "prompt_eval_duration",
                                             "eval_duration", "total_duration")},
            "narrator": narr_stats,
            "exec_error": exec_error,
        },
        latency_s=time.perf_counter() - t0,
    )


def ask_api(question: str, cfg: Config = CFG) -> ChatResult:
    """POST {question} to the chat backend. Expected JSON (any subset):
        {"cypher": "...", "rows": [{...}], "answer": "..."}
    If only `cypher` is returned the caller executes it against Neo4j.
    """
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    t0 = time.perf_counter()
    r = requests.post(cfg.api_url, json={"question": question}, headers=headers, timeout=cfg.api_timeout)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    cypher = body.get(cfg.field_cypher)
    return ChatResult(
        cypher=_clean_cypher(cypher) if cypher else None,
        rows=body.get(cfg.field_rows),
        answer=body.get(cfg.field_answer),
        raw={"api": body, "exec_error": None},
        latency_s=latency,
    )


def ask_chat(question: str, graph: Graph | None = None, cfg: Config = CFG) -> ChatResult:
    if cfg.mode == "ollama":
        if graph is None:
            raise ValueError("ollama mode needs a Graph to execute the generated Cypher")
        return ask_ollama(question, graph, cfg)
    return ask_api(question, cfg)


# ---------------------------------------------------------------------------
# Normalisation + comparison (column-name and column-order agnostic)
# ---------------------------------------------------------------------------
def _norm_value(v: Any, round_to: int) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, decimal.Decimal)):
        return round(float(v), round_to)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if hasattr(v, "iso_format"):            # neo4j temporal types
        return v.iso_format()
    if hasattr(v, "to_native"):
        return _norm_value(v.to_native(), round_to)
    if isinstance(v, dict):
        return json.dumps({k: _norm_value(x, round_to) for k, x in sorted(v.items())}, sort_keys=True, default=str)
    if isinstance(v, (list, tuple)):
        return json.dumps(sorted((_norm_value(x, round_to) for x in v), key=str), default=str)
    return str(v).strip().lower()


def _row_sig(row: dict, round_to: int) -> tuple:
    """Signature independent of column names and order."""
    return tuple(sorted((str(_norm_value(v, round_to)) for v in row.values()), key=str))


def _numeric_close(a: Any, b: Any, tol: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _first_number(rows: list[dict]) -> float | None:
    for row in rows[:1]:
        for v in row.values():
            if isinstance(v, (int, float, decimal.Decimal)) and not isinstance(v, bool):
                return float(v)
    return None


def compare(mode: str, got: list[dict], want: list[dict], q: Question) -> tuple[bool, str]:
    """text2cypher verdict. Return (passed, message)."""
    got = got or []
    want = want or []

    # guard: ground truth itself is broken (property missing -> all nulls)
    if want and all(v is None for r in want for v in r.values()):
        return (False, "GROUND TRUTH returned only nulls - property missing in graph? check schema drift")

    if mode == "nonempty":
        return (len(got) > 0, f"got {len(got)} rows")

    if mode == "count":
        g, w = _first_number(got), _first_number(want)
        if g is None:
            return (False, f"no numeric value in response; rows={got[:3]}")
        return (_numeric_close(g, w, q.tolerance), f"got {g} want {w} (tol {q.tolerance})")

    if mode == "value":
        if len(got) != 1 or len(want) != 1:
            return (False, f"expected exactly 1 row, got {len(got)} want {len(want)}")
        gv = sorted((_norm_value(v, q.round_to) for v in got[0].values() if v is not None), key=str)
        wv = sorted((_norm_value(v, q.round_to) for v in want[0].values() if v is not None), key=str)
        if len(gv) != len(wv):
            return (False, f"value count differs: got {gv} want {wv}")
        for a, b in zip(gv, wv):
            if a == b or _numeric_close(a, b, q.tolerance):
                continue
            return (False, f"mismatch: got {gv} want {wv}")
        return (True, "ok")

    if mode == "topk":
        k = q.topk
        want_keys = [_norm_value(next(iter(r.values())), q.round_to) for r in want[:k]]
        if not got:
            return (False, "empty response")
        for c in got[0].keys():
            got_keys = [_norm_value(r.get(c), q.round_to) for r in got[:k]]
            if got_keys == want_keys:
                return (True, f"ordered top-{k} match on column '{c}'")
        return (False, f"no column reproduces top-{k} keys {want_keys[:5]}...; got first row {got[0]}")

    if mode == "rows":
        # Column-agnostic AND tolerant of EXTRA columns in the response:
        # a response row matches an expected row when every expected value is
        # found (once) among the response row's values, within numeric tolerance.
        if len(got) != len(want):
            return (False, f"row count differs: got {len(got)} want {len(want)}")
        def vals(r): return [_norm_value(v, q.round_to) for v in r.values()]
        def row_match(g, w):
            pool = list(g)
            for wv in w:
                hit = next((i for i, gv in enumerate(pool) if gv == wv or _numeric_close(gv, wv, q.tolerance)), None)
                if hit is None:
                    return False
                pool.pop(hit)
            return True
        pool = [vals(r) for r in got]
        unmatched = []
        for w in (vals(r) for r in want):
            hit = next((i for i, g in enumerate(pool) if row_match(g, w)), None)
            if hit is None:
                unmatched.append(w)
            else:
                pool.pop(hit)
        if not unmatched:
            return (True, f"{len(want)} rows match")
        return (False, f"{len(unmatched)} expected rows not found, e.g. {unmatched[:2]}; unmatched response rows e.g. {pool[:2]}")

    return (False, f"unknown compare mode {mode}")


# ---------------------------------------------------------------------------
# Narration faithfulness (verdict on the description model)
# ---------------------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")


def check_narration(answer: str | None, rows: list[dict], q: Question) -> tuple[bool, str]:
    """Is the narrator's answer faithful to the rows the Cypher returned?"""
    if not answer:
        return (False, "empty answer")
    if not rows:
        return (True, "no rows to narrate")
    # 1. every number quoted in the answer must exist in the result set
    row_nums: set[float] = set()
    for r in rows:
        for v in r.values():
            if isinstance(v, (int, float, decimal.Decimal)) and not isinstance(v, bool):
                f = float(v)
                row_nums.update({round(f, 0), round(f, 1), round(f, 2)})
    quoted = [float(n.replace(",", "")) for n in _NUM.findall(answer)]
    suspicious = [n for n in quoted if n > 20 and not any(abs(n - r) <= max(q.tolerance, 0.5) for r in row_nums)]
    if suspicious:
        return (False, f"numbers in answer not in results: {suspicious[:5]}")

    # 2. ranked questions: top entities named, in order
    if q.compare == "topk":
        first_col = next(iter(rows[0].keys()))
        names = [str(r[first_col]) for r in rows[:min(3, q.topk)] if r.get(first_col) is not None]
        pos = [answer.lower().find(n.lower()) for n in names]
        if any(p < 0 for p in pos):
            return (False, f"top entities missing from answer: {[n for n, p in zip(names, pos) if p < 0]}")
        if pos != sorted(pos):
            return (False, f"top entities mentioned out of order: {names}")

    # 3. single-number questions must state the number
    if q.compare == "count":
        n = _first_number(rows)
        if n is not None and not any(abs(n - x) <= q.tolerance for x in quoted):
            return (False, f"count {n:g} not stated in answer")
    

    return (True, "faithful")


# ---------------------------------------------------------------------------
# Results sink
# ---------------------------------------------------------------------------
class Results:
    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.jsonl = RESULTS_DIR / f"run-{stamp}.jsonl"
        self.summary = RESULTS_DIR / f"run-{stamp}.md"
        self.records: list[dict] = []

    def add(self, rec: dict):
        self.records.append(rec)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def write_summary(self):
        if not self.records:
            return
        total = len(self.records)
        passed = sum(1 for r in self.records if r["passed"])
        narrated = [r for r in self.records if r.get("narration_passed") is not None]
        narr_ok = sum(1 for r in narrated if r["narration_passed"])

        by_cat: dict[str, list[int]] = {}
        for r in self.records:
            by_cat.setdefault(r["category"], [0, 0, 0, 0])
            by_cat[r["category"]][1] += 1
            by_cat[r["category"]][0] += int(bool(r["passed"]))
            if r.get("narration_passed") is not None:
                by_cat[r["category"]][3] += 1
                by_cat[r["category"]][2] += int(bool(r["narration_passed"]))

        lat = [r.get("latency_s", 0) for r in self.records if r.get("latency_s")]
        lines = [
            f"# text2cypher run - {CFG.mode} / {CFG.t2c_model if CFG.mode == 'ollama' else CFG.api_url}",
            "",
            f"- **Cypher correct:** {passed}/{total} ({100 * passed / total:.0f}%)",
        ]
        if narrated:
            lines.append(f"- **Narration faithful:** {narr_ok}/{len(narrated)} ({100 * narr_ok / len(narrated):.0f}%) "
                         f"({CFG.narrator_model or 'app'})")
        if lat:
            lines.append(f"- **Latency:** avg {sum(lat) / len(lat):.1f}s, max {max(lat):.1f}s")
        lines += ["", "| category | cypher | narration |", "|---|---|---|"]
        lines += [f"| {c} | {p}/{t} | {np}/{nt} |" if nt else f"| {c} | {p}/{t} | - |"
                  for c, (p, t, np, nt) in sorted(by_cat.items())]
        lines += ["", "| id | cypher | narration | s | question | detail |", "|---|---|---|---|---|---|"]
        for r in self.records:
            n = r.get("narration_passed")
            nflag = "-" if n is None else ("PASS" if n else "FAIL")
            detail = str(r["detail"]).replace("|", "/")[:140]
            if n is False:
                detail += f" // narration: {str(r.get('narration_detail', '')).replace('|', '/')[:80]}"
            lines.append(f"| {r['id']} | {'PASS' if r['passed'] else 'FAIL'} | {nflag} | "
                         f"{r.get('latency_s', 0):.1f} | {r['question']} | {detail} |")
        self.summary.write_text("\n".join(lines), encoding="utf-8")