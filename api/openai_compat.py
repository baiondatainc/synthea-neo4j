"""
OpenAI-compatible /v1/chat/completions endpoint for LibreChat integration.
Returns streamed answers + Recharts artifact for chart-worthy results.

Fixes applied:
  1. detect_chart()  — two-pass key detection, never confuses label vs numeric
  2. build_artifact() — jsx initialised before if/elif, all three chart types restored
  3. build_artifact_old() — removed (dead code)
  4. generate_stream() — wrapped build_artifact in try/except so crashes don't kill the stream
"""
import json
import time
import uuid
import logging
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from qa.chain import stream_qa_response
from config import get_settings
from guardrails import redact_text

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Chart detection ───────────────────────────────────────────────────────────
def detect_chart(question: str, results: list) -> dict | None:
    """
    Inspects query results and decides whether a chart makes sense.
    Returns a chart spec dict or None.

    Label key  — the x-axis: string preferred, integer accepted (year, month)
    Numeric key — the y-axis: int/float, never the label key

    Handles:
      - String labels:  {"year_month": "2023-01", "review_count": 12}
      - Integer labels: {"year": 2023, "review_count": 12}
      - Float-cast:     {"year_month": "2023-01", "total": "1234.5"}
    """
    if not results or len(results) < 2:
        return None

    first = results[0]
    label_key = None
    numeric_key = None

    # ── Pass 1: prefer a STRING key as label ─────────────────────────────
    for k, v in first.items():
        if isinstance(v, str) and label_key is None:
            label_key = k
        if isinstance(v, (int, float)) and numeric_key is None:
            numeric_key = k

    # ── Pass 1b: no string label found — accept first integer as label ───
    # Handles: {"year": 2023, "count": 450} or {"month": 6, "total": 1200}
    if label_key is None:
        for k, v in first.items():
            if isinstance(v, (int, float)):
                label_key = k
                break
        # Reset numeric so we re-scan excluding the label key
        numeric_key = None

    # ── Pass 2: find numeric key (must differ from label_key) ────────────
    if label_key is not None and numeric_key is None:
        for k, v in first.items():
            if k == label_key:
                continue
            if isinstance(v, (int, float)):
                numeric_key = k
                break

    # ── Pass 3: float-cast fallback (handles stringified numbers) ────────
    if numeric_key is None:
        for k, v in first.items():
            if k == label_key:
                continue
            try:
                float(v)
                numeric_key = k
                break
            except (TypeError, ValueError):
                pass

    if not label_key or not numeric_key:
        logger.warning(
            f"detect_chart: could not identify label+numeric keys. "
            f"Keys={list(first.keys())} Values={list(first.values())}"
        )
        return None

    # ── Chart type from question keywords ─────────────────────────────────
    q = question.lower()
    if any(w in q for w in [
        "trend", "over time", "monthly", "yearly",
        "by year", "by month", "per year", "each year",
        "per month", "each month", "timeline", "over the years",
    ]):
        chart_type = "line"
    elif any(w in q for w in [
        "distribution", "breakdown", "proportion", "share",
        "gender", "race", "ethnicity", "pie", "split",
    ]):
        chart_type = "pie"
    else:
        chart_type = "bar"

    # ── Build data rows ───────────────────────────────────────────────────
    data = []
    for r in results[:20]:
        raw_label = r.get(label_key, "")
        label = str(raw_label)[:35]
        try:
            value = float(r.get(numeric_key, 0))
        except (TypeError, ValueError):
            value = 0.0
        data.append({label_key: label, numeric_key: round(value, 2)})

    # ── Guard: need at least 2 distinct labels ────────────────────────────
    unique_labels = len(set(row[label_key] for row in data))
    if unique_labels < 2:
        logger.info(
            f"detect_chart: only {unique_labels} unique label(s) — skipping chart. "
            f"Property '{label_key}' may lack diversity in Neo4j."
        )
        return None

    # ── Guard: reject birth-year ranges on line charts ────────────────────
    # "2023-01" style strings are valid trends — only check pure year integers
    if chart_type == "line":
        try:
            pure_years = []
            for row in data:
                val = row[label_key]
                if isinstance(val, str):
                    # Accept 4-digit year strings, skip "YYYY-MM" patterns
                    if len(val) == 4 and val.isdigit():
                        pure_years.append(int(val))
                    # "YYYY-MM" or "YYYY-MM-DD" — valid trend label, skip check
                elif isinstance(val, (int, float)):
                    pure_years.append(int(val))

            if pure_years and max(pure_years) < 1990:
                logger.warning(
                    f"detect_chart: year range {min(pure_years)}-{max(pure_years)} "
                    f"looks like birth years, not encounter/event dates — skipping chart. "
                    f"Verify the query returns service/event dates, not patient DOB."
                )
                return None
        except (ValueError, TypeError):
            pass

    logger.info(
        f"detect_chart: type={chart_type} label_key={label_key!r} "
        f"numeric_key={numeric_key!r} rows={len(data)} unique_labels={unique_labels}"
    )

    return {
        "type": chart_type,
        "title": question[:70] + ("..." if len(question) > 70 else ""),
        "data": data,
        "x_key": label_key,
        "y_key": numeric_key,
    }


# ── Artifact builder ──────────────────────────────────────────────────────────

def build_artifact(chart: dict) -> str:
    """
    Builds a LibreChat artifact fence containing a Recharts JSX component.

    All three chart types (bar, pie, line) are handled.
    jsx is always initialised before the if/elif block — no UnboundLocalError.
    """
    data_json = json.dumps(chart["data"], ensure_ascii=False)
    x = chart["x_key"]
    y = chart["y_key"]
    title = chart["title"].replace('"', '\\"')
    ctype = chart["type"]
    cid = uuid.uuid4().hex[:8]

        # ── Always initialise jsx — prevents UnboundLocalError ────────────────
    jsx = ""

    if ctype == "bar":
        jsx = f"""import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell }} from 'recharts';
        const data = {data_json};
        const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7'];

        export default function Chart() {{
        return (
            <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
            <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
            <ResponsiveContainer width="100%" height={{380}}>
                <BarChart data={{data}} margin={{{{top:5, right:20, left:10, bottom:100}}}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="{x}" tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-40}} textAnchor="end" interval={{0}} />
                <YAxis tick={{{{fill:'#a0aec0', fontSize:12}}}} />
                <Tooltip contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
                <Bar dataKey="{y}" radius={{[4,4,0,0]}}>
                    {{data.map((_, i) => <Cell key={{i}} fill={{COLORS[i % COLORS.length]}} />)}}
                </Bar>
                </BarChart>
            </ResponsiveContainer>
            </div>
    );
    }}"""
    elif ctype == "pie":
        jsx = f"""import {{ PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer }} from 'recharts';

        const data = {data_json};
        const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7','#fbb6ce','#90cdf4'];

        export default function Chart() {{
        return (
            <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
            <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
            <ResponsiveContainer width="100%" height={{380}}>
                <PieChart>
                <Pie data={{data}} dataKey="{y}" nameKey="{x}" cx="50%" cy="50%" outerRadius={{130}}
                    label={{({{name, percent}}) => name + ' ' + (percent*100).toFixed(0) + '%'}}
                    labelLine={{true}}>
                    {{data.map((_, i) => <Cell key={{i}} fill={{COLORS[i % COLORS.length]}} />)}}
                </Pie>
                <Tooltip contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
                <Legend wrapperStyle={{{{color:'#a0aec0'}}}} />
                </PieChart>
            </ResponsiveContainer>
            </div>
        );
        }}"""

    elif ctype == "line":
            jsx = f"""import {{ AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer }} from 'recharts';

        const data = {data_json};

        export default function Chart() {{
        return (
            <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
            <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
            <ResponsiveContainer width="100%" height={{380}}>
                <AreaChart data={{data}} margin={{{{top:10, right:20, left:10, bottom:60}}}}>
                <defs>
                    <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#63b3ed" stopOpacity={{0.3}} />
                    <stop offset="95%" stopColor="#63b3ed" stopOpacity={{0}} />
                    </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis dataKey="{x}" tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-35}} textAnchor="end" interval={{0}} />
                <YAxis tick={{{{fill:'#a0aec0', fontSize:12}}}} />
                <Tooltip contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
                <Area type="monotone" dataKey="{y}" stroke="#63b3ed" strokeWidth={{2}}
                    fill="url(#colorVal)" dot={{{{r:4, fill:'#63b3ed', stroke:'#1a1d2e', strokeWidth:2}}}} />
                </AreaChart>
            </ResponsiveContainer>
            </div>
        );
        }}"""

    else:
            # Unknown chart type — fall back to bar rather than crash
            logger.warning(f"build_artifact: unknown chart type '{ctype}' — falling back to bar")
            jsx = f"""import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell }} from 'recharts';

            const data = {data_json};
            const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7'];

            export default function Chart() {{
            return (
                <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
                <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
                <ResponsiveContainer width="100%" height={{380}}>
                    <BarChart data={{data}} margin={{{{top:5, right:20, left:10, bottom:100}}}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                    <XAxis dataKey="{x}" tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-40}} textAnchor="end" interval={{0}} />
                    <YAxis tick={{{{fill:'#a0aec0', fontSize:12}}}} />
                    <Tooltip contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
                    <Bar dataKey="{y}" radius={{[4,4,0,0]}}>
                        {{data.map((_, i) => <Cell key={{i}} fill={{COLORS[i % COLORS.length]}} />)}}
                    </Bar>
                    </BarChart>
                </ResponsiveContainer>
                </div>
            );
            }}"""

    # Safety guard — should never be empty after the blocks above
    if not jsx.strip():
        logger.error("build_artifact: jsx is empty after all branches — skipping chart")
        return ""

    return (
        f'\n\n'
        f':::artifact{{identifier="chart-{cid}" type="application/vnd.ant.react" title="{title}"}}\n'
        f'{jsx.strip()}\n'
        f':::\n\n'
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse_chunk(content: str, model: str) -> str:
    return (
        f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
    )


def sse_done() -> str:
    return "data: [DONE]\n\n"


# ── Streaming generator ───────────────────────────────────────────────────────

async def generate_stream(
    question: str,
    model: str,
    conversation_id: str | None = None,
    use_cache: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Buffers all LLM tokens, then emits:
      1. (optional) Follow-up rewrite notice
      2. (optional) Cache-hit marker
      3. Full answer text
      4. Cypher code block
      5. Chart artifact (if data is chartable)
      6. [DONE]

    Wrapped in try/except so any crash yields a clean error message
    instead of a hard disconnect ("terminated").
    """
    buffered_tokens = []
    neo4j_results = []
    cypher_query = ""
    rewritten_question = None
    cache_hit = False

    try:
        async for chunk in stream_qa_response(
            question,
            conversation_id=conversation_id,
            use_cache=use_cache,
        ):
            t = chunk["type"]

            if t == "token":
                buffered_tokens.append(chunk["data"])

            elif t == "rewrite":
                rewritten_question = chunk.get("data", "")

            elif t == "cache_hit":
                cache_hit = True

            elif t == "cypher":
                cypher_query = chunk.get("data", "")
                neo4j_results = chunk.get("results", [])

            elif t == "end":
                break

            elif t == "blocked":
                # Guardrail rejected the input or generated Cypher.
                # Stream the reason as the answer text and stop cleanly.
                yield sse_chunk(f"⚠️ {chunk['data']}", model)
                yield sse_done()
                return

            elif t == "error":
                error_msg = chunk["data"]
                # Surface a readable message — not a raw stack trace
                if "SyntaxError" in error_msg or "GqlError" in error_msg:
                    friendly = (
                        "The query generator produced invalid Cypher. "
                        "Try rephrasing your question.\n\n"
                        f"> **Details:** `{error_msg[:300]}`"
                    )
                else:
                    friendly = f"An error occurred while processing the request: {error_msg}"
                yield sse_chunk(friendly, model)
                yield sse_done()
                return

    except Exception as e:
        logger.error(f"generate_stream error: {e}", exc_info=True)
        yield sse_chunk(f"An unexpected error occurred: {e}", model)
        yield sse_done()
        return

    # ── 0a. Follow-up rewrite notice (only if memory rewrote the question) ─
    if rewritten_question:
        yield sse_chunk(
            f"> _Interpreted as: {rewritten_question}_\n\n",
            model,
        )

    # ── 0b. Cache marker — subtle, just so users know responses can repeat ─
    if cache_hit:
        yield sse_chunk("> _(cached)_\n\n", model)

    # ── 1. Answer text (belt-and-suspenders redaction; rows already redacted) ─
    full_answer = "".join(buffered_tokens)
    if full_answer:
        if get_settings().guardrails_redact_output:
            full_answer = redact_text(full_answer)
        yield sse_chunk(full_answer, model)

    # ── 2. Cypher block ───────────────────────────────────────────────────
    if cypher_query:
        yield sse_chunk(f"\n\n```cypher\n{cypher_query}\n```\n\n", model)

    # ── 3. Chart artifact ─────────────────────────────────────────────────
    if neo4j_results:
        logger.info(
            f"Chart check: {len(neo4j_results)} rows, "
            f"keys: {list(neo4j_results[0].keys())}"
        )
    else:
        logger.info("Chart check: no results")

    chart = detect_chart(question, neo4j_results)

    if chart:
        logger.info(f"Generating {chart['type']} chart with {len(chart['data'])} points")
        try:
            artifact = build_artifact(chart)
            if artifact:
                yield sse_chunk(artifact, model)
        except Exception as e:
            # Chart crash must NOT kill the whole stream — answer already sent
            logger.error(f"build_artifact failed: {e}", exc_info=True)
            yield sse_chunk(
                f"\n\n> ⚠️ Chart rendering failed: {e}\n",
                model,
            )
    else:
        logger.info(f"No chart — sample: {neo4j_results[:1]}")

    # ── 4. Done ───────────────────────────────────────────────────────────
    yield sse_done()


# ── /v1/chat/completions ──────────────────────────────────────────────────────

def _extract_conversation_id(body: dict, request: Request) -> str | None:
    """Find a stable per-conversation identifier from the OpenAI-style body.

    LibreChat doesn't reliably send a conversation_id in the standard OpenAI
    request shape, so we check a few common locations and fall back to a hash
    of the first user message — stable across turns of the same conversation
    because LibreChat re-sends the full transcript on every turn.
    """
    meta = body.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("conversation_id", "conversationId", "chat_id", "chatId"):
            if meta.get(k):
                return str(meta[k])

    for k in ("conversation_id", "conversationId", "chat_id", "chatId"):
        if body.get(k):
            return str(body[k])

    if body.get("user"):
        return f"user-{body['user']}"

    header_id = request.headers.get("x-conversation-id")
    if header_id:
        return header_id

    messages = body.get("messages") or []
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user:
        import hashlib
        content = first_user.get("content", "")
        seed = content if isinstance(content, str) else str(content)
        return "anon-" + hashlib.sha256(seed.encode()).hexdigest()[:16]
    return None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    messages = body.get("messages", [])
    model = body.get("model", "neo4j-kg")

    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            question = content if isinstance(content, str) else str(content)
            break

    if not question:
        return JSONResponse(status_code=400, content={"error": "No user message"})

    conversation_id = _extract_conversation_id(body, request)
    nocache = request.query_params.get("nocache", "").lower() in ("1", "true", "yes")
    use_cache = not nocache
    logger.info(f"Question: {question[:80]}  conv_id={conversation_id}  cache={use_cache}")

    return StreamingResponse(
        generate_stream(question, model, conversation_id=conversation_id, use_cache=use_cache),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /v1/models ────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "neo4j-kg",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "synthea-neo4j",
            }
        ],
    }


# ── Cache admin (Phase C) ─────────────────────────────────────────────────────

@router.post("/cache/clear")
async def cache_clear():
    """Wipe the answer cache. Use after a schema/ingestion change."""
    from cache import get_answer_cache
    deleted = get_answer_cache().clear_all()
    return {"cleared": deleted}