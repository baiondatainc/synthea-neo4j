"""
OpenAI-compatible /v1/chat/completions endpoint for LibreChat integration.
Returns streamed answers + Recharts artifact for chart-worthy results.

Charts now work for BOTH:
  - Cypher path: results come via {"type": "cypher", "results": [...]}
  - Summary path: results come via {"type": "summary_data", "data": {...}, "label": "Financial"}

Summary chart types per label:
  Financial   → payments by year (line) + outstanding by state (bar)
  Patient     → patients by state (bar) + by cohort (bar)
  Location    → visits by location (bar) + reviews by location (bar)
  Campaign    → calls by campaign (bar)
  Charge      → charges by year (line) + by modality (bar)
  Transaction → payments by year (line)
  BirdeyeReview → rating distribution (bar)
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
    if not results or len(results) < 2:
        return None

    first = results[0]
    label_key = None
    numeric_key = None

    for k, v in first.items():
        if isinstance(v, str) and label_key is None:
            label_key = k
        if isinstance(v, (int, float)) and numeric_key is None:
            numeric_key = k

    if label_key is None:
        for k, v in first.items():
            if isinstance(v, (int, float)):
                label_key = k
                break
        numeric_key = None

    if label_key is not None and numeric_key is None:
        for k, v in first.items():
            if k == label_key:
                continue
            if isinstance(v, (int, float)):
                numeric_key = k
                break

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
        return None

    q = question.lower()
    if any(w in q for w in ["trend", "over time", "monthly", "yearly", "by year",
                              "by month", "per year", "each year", "per month",
                              "each month", "timeline", "over the years"]):
        chart_type = "line"
    elif any(w in q for w in ["distribution", "breakdown", "proportion", "share",
                                "gender", "race", "ethnicity", "pie", "split"]):
        chart_type = "pie"
    else:
        chart_type = "bar"

    data = []
    for r in results[:20]:
        raw_label = r.get(label_key, "")
        label = str(raw_label)[:35]
        try:
            value = float(r.get(numeric_key, 0))
        except (TypeError, ValueError):
            value = 0.0
        data.append({label_key: label, numeric_key: round(value, 2)})

    unique_labels = len(set(row[label_key] for row in data))
    if unique_labels < 2:
        return None

    if chart_type == "line":
        try:
            pure_years = []
            for row in data:
                val = row[label_key]
                if isinstance(val, str):
                    if len(val) == 4 and val.isdigit():
                        pure_years.append(int(val))
                elif isinstance(val, (int, float)):
                    pure_years.append(int(val))
            if pure_years and max(pure_years) < 1990:
                return None
        except (ValueError, TypeError):
            pass

    logger.info(f"detect_chart: type={chart_type} label={label_key!r} numeric={numeric_key!r} rows={len(data)}")
    return {
        "type":  chart_type,
        "title": question[:70] + ("..." if len(question) > 70 else ""),
        "data":  data,
        "x_key": label_key,
        "y_key": numeric_key,
    }


# ── Summary chart builder ─────────────────────────────────────────────────────

def _summary_charts(label: str, data: dict) -> list[dict]:
    """
    Extract chart-ready row lists from aggregate summary data.
    Returns a list of chart spec dicts (one per chart to render).
    Each spec: {"title": str, "rows": list[dict], "chart_type": "bar"|"line"|"pie"}
    """
    charts = []

    if label == "Financial":
        py = data.get("payments_by_year", [])
        if len(py) >= 2:
            charts.append({
                "title":      "Annual Payment Trend",
                "rows":       [{"year": str(r["year"]), "total_payments": r["total_payments"]} for r in py],
                "chart_type": "line",
            })
        bs = data.get("by_state", [])
        if len(bs) >= 2:
            charts.append({
                "title":      "Outstanding Balance by State",
                "rows":       [{"state": r["state"], "outstanding": r["outstanding"]} for r in bs],
                "chart_type": "bar",
            })
        cy = data.get("charges_by_year", [])
        if len(cy) >= 2:
            charts.append({
                "title":      "Annual Charge Trend",
                "rows":       [{"year": str(r["year"]), "total_amount": r["total_amount"]} for r in cy],
                "chart_type": "line",
            })

    elif label == "Patient":
        bs = data.get("by_state", [])
        if len(bs) >= 2:
            charts.append({
                "title":      "Patients by State",
                "rows":       [{"state": r["state"], "patient_count": r["patient_count"]} for r in bs],
                "chart_type": "bar",
            })
        bc = data.get("by_cohort", [])
        if len(bc) >= 2:
            charts.append({
                "title":      "Outstanding Balance by Payor Cohort",
                "rows":       [{"cohort": r["cohort"], "outstanding": r["outstanding"]} for r in bc],
                "chart_type": "bar",
            })
        bg = data.get("by_gender", [])
        if len(bg) >= 2:
            charts.append({
                "title":      "Patients by Gender",
                "rows":       [{"gender": r["gender"], "count": r["count"]} for r in bg],
                "chart_type": "pie",
            })

    elif label == "Location":
        vs = data.get("visit_stats", [])
        if len(vs) >= 2:
            charts.append({
                "title":      "Top Locations by Visit Count",
                "rows":       [{"location": r["location"], "visit_count": r["visit_count"]} for r in vs],
                "chart_type": "bar",
            })
        rs = data.get("review_stats", [])
        if len(rs) >= 2:
            charts.append({
                "title":      "Birdeye Avg Rating by Location",
                "rows":       [{"location": r["location"], "avg_rating": r["avg_rating"]} for r in rs],
                "chart_type": "bar",
            })

    elif label == "Campaign":
        cs = data.get("call_stats", [])
        if len(cs) >= 2:
            charts.append({
                "title":      "Calls by Campaign",
                "rows":       [{"campaign": r["campaign"], "total_calls": r["total_calls"]} for r in cs],
                "chart_type": "bar",
            })

    elif label == "Charge":
        cy = data.get("by_year", [])
        if len(cy) >= 2:
            charts.append({
                "title":      "Annual Charge Trend",
                "rows":       [{"year": str(r["year"]), "total_amount": r["total_amount"]} for r in cy],
                "chart_type": "line",
            })
        bm = data.get("by_modality", [])
        if len(bm) >= 2:
            charts.append({
                "title":      "Total Charges by Modality",
                "rows":       [{"modality": r["modality"], "total_amount": r["total_amount"]} for r in bm],
                "chart_type": "bar",
            })

    elif label == "Transaction":
        py = data.get("by_year", [])
        if len(py) >= 2:
            charts.append({
                "title":      "Annual Payment Trend",
                "rows":       [{"year": str(r["year"]), "total_payments": r["total_payments"]} for r in py],
                "chart_type": "line",
            })

    elif label == "BirdeyeReview":
        o = data.get("overall", {})
        if o:
            rating_rows = [
                {"rating": "1-Star", "count": o.get("one_star", 0)},
                {"rating": "2-Star", "count": o.get("two_star", 0)},
                {"rating": "3-Star", "count": o.get("three_star", 0)},
                {"rating": "4-Star", "count": o.get("four_star", 0)},
                {"rating": "5-Star", "count": o.get("five_star", 0)},
            ]
            charts.append({
                "title":      "Review Rating Distribution",
                "rows":       rating_rows,
                "chart_type": "bar",
            })
        bl = data.get("by_location", [])
        if len(bl) >= 2:
            charts.append({
                "title":      "Avg Rating by Location",
                "rows":       [{"location": r["location"], "avg_rating": r["avg_rating"]} for r in bl],
                "chart_type": "bar",
            })

    elif label == "Practice":
        bp = data.get("by_practice", [])
        if len(bp) >= 2:
            charts.append({
                "title":      "Patients by Practice",
                "rows":       [{"practice": r["practice"], "patient_count": r["patient_count"]} for r in bp],
                "chart_type": "bar",
            })

    return charts


def _build_summary_chart(spec: dict) -> dict | None:
    """Convert a summary chart spec into a detect_chart-compatible dict."""
    rows = spec.get("rows", [])
    if len(rows) < 2:
        return None
    first = rows[0]
    keys = list(first.keys())
    if len(keys) < 2:
        return None
    x_key = keys[0]
    y_key = keys[1]
    return {
        "type":  spec["chart_type"],
        "title": spec["title"],
        "data":  rows,
        "x_key": x_key,
        "y_key": y_key,
    }


# ── Artifact builder ──────────────────────────────────────────────────────────

def build_artifact(chart: dict) -> str:
    data_json = json.dumps(chart["data"], ensure_ascii=False)
    x     = chart["x_key"]
    y     = chart["y_key"]
    title = chart["title"].replace('"', '\\"')
    ctype = chart["type"]
    cid   = uuid.uuid4().hex[:8]

    jsx = ""

    # ── Value formatter (shared across chart types) ───────────────────────
    fmt_fn = """
const formatValue = (v) => {
  if (v === null || v === undefined) return '';
  const n = parseFloat(v);
  if (isNaN(n)) return v;
  if (n >= 1_000_000) return '$' + (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000)     return '$' + (n / 1_000).toFixed(1) + 'K';
  return n.toLocaleString();
};
const formatMonth = (val) => {
  if (!val || !String(val).includes('-')) return val;
  const [y, m] = String(val).split('-');
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return (months[parseInt(m, 10) - 1] || m) + ' ' + y;
};
const isMonthLabel = (val) => val && String(val).match(/^\\d{{4}}-\\d{{2}}$/);
"""

    if ctype == "bar":
        jsx = f"""import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell }} from 'recharts';
const data = {data_json};
const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7','#fbb6ce','#90cdf4'];
{fmt_fn}
export default function Chart() {{
  return (
    <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
      <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
      <ResponsiveContainer width="100%" height={{380}}>
        <BarChart data={{data}} margin={{{{top:5, right:30, left:20, bottom:100}}}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
          <XAxis dataKey="{x}" tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-40}} textAnchor="end" interval={{0}} />
          <YAxis tickFormatter={{formatValue}} tick={{{{fill:'#a0aec0', fontSize:12}}}} width={{70}} />
          <Tooltip formatter={{(v) => [formatValue(v), '{y}']}}
            contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
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
{fmt_fn}
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
          <Tooltip formatter={{(v) => [formatValue(v), '{y}']}}
            contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
          <Legend wrapperStyle={{{{color:'#a0aec0'}}}} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}}"""

    elif ctype == "line":
        jsx = f"""import {{ AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer }} from 'recharts';
const data = {data_json};
{fmt_fn}
export default function Chart() {{
  const useMonthFmt = data.length > 0 && isMonthLabel(data[0]['{x}']);
  return (
    <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
      <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
      <ResponsiveContainer width="100%" height={{380}}>
        <AreaChart data={{data}} margin={{{{top:10, right:30, left:20, bottom:60}}}}>
          <defs>
            <linearGradient id="colorVal{cid}" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#63b3ed" stopOpacity={{0.3}} />
              <stop offset="95%" stopColor="#63b3ed" stopOpacity={{0}} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
          <XAxis dataKey="{x}"
            tickFormatter={{useMonthFmt ? formatMonth : (v => v)}}
            tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-35}} textAnchor="end" interval={{0}} />
          <YAxis tickFormatter={{formatValue}} tick={{{{fill:'#a0aec0', fontSize:12}}}} width={{70}} />
          <Tooltip
            labelFormatter={{useMonthFmt ? formatMonth : (v => v)}}
            formatter={{(v) => [formatValue(v), '{y}']}}
            contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
          <Area type="monotone" dataKey="{y}" stroke="#63b3ed" strokeWidth={{2}}
            fill="url(#colorVal{cid})"
            dot={{{{r:4, fill:'#63b3ed', stroke:'#1a1d2e', strokeWidth:2}}}} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}}"""

    else:
        logger.warning(f"build_artifact: unknown chart type '{ctype}' — falling back to bar")
        return build_artifact({**chart, "type": "bar"})

    if not jsx.strip():
        logger.error("build_artifact: jsx is empty — skipping chart")
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
    buffered_tokens   = []
    neo4j_results     = []
    cypher_query      = ""
    rewritten_question = None
    cache_hit         = False
    summary_data      = None   # {"label": str, "data": dict}

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
                cypher_query  = chunk.get("data", "")
                neo4j_results = chunk.get("results", [])

            elif t == "summary_data":
                # Emitted by summarize_all_nodes / summarize_node
                summary_data = {
                    "label": chunk.get("label", ""),
                    "data":  chunk.get("data", {}),
                }

            elif t == "end":
                break

            elif t == "blocked":
                yield sse_chunk(f"⚠️ {chunk['data']}", model)
                yield sse_done()
                return

            elif t == "error":
                error_msg = chunk["data"]
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

    # ── 0a. Rewrite notice ────────────────────────────────────────────────
    if rewritten_question:
        yield sse_chunk(f"> _Interpreted as: {rewritten_question}_\n\n", model)

    # ── 0b. Cache marker ──────────────────────────────────────────────────
    if cache_hit:
        yield sse_chunk("> _(cached)_\n\n", model)

    # ── 1. Answer text ────────────────────────────────────────────────────
    full_answer = "".join(buffered_tokens)
    if full_answer:
        if get_settings().guardrails_redact_output:
            full_answer = redact_text(full_answer)
        yield sse_chunk(full_answer, model)

    # ── 2. Cypher block (Cypher path only) ────────────────────────────────
    if cypher_query:
        yield sse_chunk(f"\n\n```cypher\n{cypher_query}\n```\n\n", model)

    # ── 3. Charts ─────────────────────────────────────────────────────────
    # 3a. Cypher path — standard single chart from Neo4j result rows
    if neo4j_results:
        logger.info(f"Chart check (cypher): {len(neo4j_results)} rows, keys: {list(neo4j_results[0].keys())}")
        chart = detect_chart(question, neo4j_results)
        if chart:
            logger.info(f"Generating {chart['type']} chart — {len(chart['data'])} points")
            try:
                artifact = build_artifact(chart)
                if artifact:
                    yield sse_chunk(artifact, model)
            except Exception as e:
                logger.error(f"build_artifact failed: {e}", exc_info=True)
                yield sse_chunk(f"\n\n> ⚠️ Chart rendering failed: {e}\n", model)
        else:
            logger.info(f"No chart — sample: {neo4j_results[:1]}")

    # 3b. Summary path — multiple charts from structured summary data
    elif summary_data:
        label = summary_data["label"]
        data  = summary_data["data"]
        logger.info(f"Chart check (summary): label={label}")
        specs = _summary_charts(label, data)
        logger.info(f"Summary charts: {len(specs)} chart(s) for label={label}")
        for spec in specs:
            chart = _build_summary_chart(spec)
            if chart:
                try:
                    artifact = build_artifact(chart)
                    if artifact:
                        yield sse_chunk(artifact, model)
                except Exception as e:
                    logger.error(f"build_artifact (summary) failed: {e}", exc_info=True)

    # ── 4. Done ───────────────────────────────────────────────────────────
    yield sse_done()


# ── /v1/chat/completions ──────────────────────────────────────────────────────

def _extract_conversation_id(body: dict, request: Request) -> str | None:
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
    messages   = body.get("messages") or []
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user:
        import hashlib
        content = first_user.get("content", "")
        seed    = content if isinstance(content, str) else str(content)
        return "anon-" + hashlib.sha256(seed.encode()).hexdigest()[:16]
    return None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    messages = body.get("messages", [])
    model    = body.get("model", "neo4j-kg")
    question = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content  = msg.get("content", "")
            question = content if isinstance(content, str) else str(content)
            break

    if not question:
        return JSONResponse(status_code=400, content={"error": "No user message"})

    conversation_id = _extract_conversation_id(body, request)
    nocache  = request.query_params.get("nocache", "").lower() in ("1", "true", "yes")
    use_cache = not nocache
    logger.info(f"Question: {question[:80]}  conv_id={conversation_id}  cache={use_cache}")

    return StreamingResponse(
        generate_stream(question, model, conversation_id=conversation_id, use_cache=use_cache),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       "neo4j-kg",
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "synthea-neo4j",
        }],
    }


@router.post("/cache/clear")
async def cache_clear():
    from cache import get_answer_cache
    deleted = get_answer_cache().clear_all()
    return {"cleared": deleted}