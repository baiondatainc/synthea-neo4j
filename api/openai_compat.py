"""
OpenAI-compatible /v1/chat/completions endpoint for LibreChat integration.
Returns streamed answers + Recharts artifact for chart-worthy results.
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
from graph.connection import Neo4jConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Chart detection ───────────────────────────────────────────────────────────

def detect_chart(question: str, results: list) -> dict | None:
    if not results or len(results) < 2:
        return None

    first = results[0]
    numeric_key = None
    label_key = None

    for k, v in first.items():
        if isinstance(v, (int, float)) and v >= 0:
            numeric_key = k
        elif isinstance(v, str):
            label_key = k

    if not numeric_key:
        for k, v in first.items():
            try:
                float(v)
                numeric_key = k
                break
            except (TypeError, ValueError):
                pass

    if not numeric_key or not label_key:
        return None

    q = question.lower()
    if any(w in q for w in ["trend", "over time", "monthly", "yearly", "by year", "by month"]):
        chart_type = "line"
    elif any(w in q for w in ["distribution", "breakdown", "proportion", "share", "gender", "race"]):
        chart_type = "pie"
    else:
        chart_type = "bar"

    data = []
    for r in results[:20]:
        label = str(r.get(label_key, ""))[:35]
        try:
            value = float(r.get(numeric_key, 0))
        except (TypeError, ValueError):
            value = 0
        data.append({label_key: label, numeric_key: round(value, 2)})

    return {
        "type": chart_type,
        "title": question[:70] + ("..." if len(question) > 70 else ""),
        "data": data,
        "x_key": label_key,
        "y_key": numeric_key,
    }


# ── Artifact builder ──────────────────────────────────────────────────────────

def build_artifact(chart: dict) -> str:
    data_json = json.dumps(chart["data"], ensure_ascii=False)
    x = chart["x_key"]
    y = chart["y_key"]
    title = chart["title"].replace('"', '\\"')
    ctype = chart["type"]
    cid = uuid.uuid4().hex[:8]

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
            label={{({{name, percent}}) => `${{name}} ${{(percent*100).toFixed(0)}}%`}}
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

    else:
        jsx = f"""import {{ LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer }} from 'recharts';

const data = {data_json};

export default function Chart() {{
  return (
    <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
      <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
      <ResponsiveContainer width="100%" height={{380}}>
        <LineChart data={{data}} margin={{{{top:5, right:20, left:10, bottom:80}}}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
          <XAxis dataKey="{x}" tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-35}} textAnchor="end" interval={{0}} />
          <YAxis tick={{{{fill:'#a0aec0', fontSize:12}}}} />
          <Tooltip contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
          <Line type="monotone" dataKey="{y}" stroke="#63b3ed" strokeWidth={{2}} dot={{{{r:4, fill:'#63b3ed'}}}} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}}"""

    return (
        f'\n\n:::artifact{{identifier=\"chart-{cid}\" type=\"application/vnd.ant.react\" title=\"{title}\"}}\n'
        f'{jsx.strip()}\n'
        f':::'
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse_chunk(content: str, model: str) -> str:
    return f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

def sse_done() -> str:
    return "data: [DONE]\n\n"


# ── Streaming generator ───────────────────────────────────────────────────────

async def generate_stream(question: str, model: str) -> AsyncGenerator[str, None]:
    # Buffer all tokens — send as one chunk so artifact tags are never split
    buffered_tokens = []
    neo4j_results = []
    cypher_query = ""

    async for chunk in stream_qa_response(question):
        t = chunk["type"]

        if t == "token":
            buffered_tokens.append(chunk["data"])

        elif t == "cypher":
            cypher_query = chunk.get("data", "")
            neo4j_results = chunk.get("results", [])

        elif t == "end":
            break

        elif t == "error":
            yield sse_chunk(f"❌ {chunk['data']}", model)
            yield sse_done()
            return

    # Send buffered answer as single chunk
    full_answer = "".join(buffered_tokens)
    if full_answer:
        yield sse_chunk(full_answer, model)

    # Send cypher block as single chunk
    if cypher_query:
        yield sse_chunk(f"\n\n```cypher\n{cypher_query}\n```\n\n", model)

    # Send chart artifact as single chunk
    logger.info(f"Chart check: {len(neo4j_results)} rows, keys: {list(neo4j_results[0].keys()) if neo4j_results else []}")
    chart = detect_chart(question, neo4j_results)
    if chart:
        logger.info(f"Generating {chart['type']} chart with {len(chart['data'])} points")
        artifact = build_artifact(chart)
        yield sse_chunk(artifact, model)
    else:
        logger.info(f"No chart — rows: {len(neo4j_results)}, sample: {neo4j_results[:1]}")

    yield sse_done()


# ── /v1/chat/completions ──────────────────────────────────────────────────────

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

    logger.info(f"Question: {question[:80]}")

    return StreamingResponse(
        generate_stream(question, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /v1/models ────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "neo4j-kg", "object": "model", "created": int(time.time()), "owned_by": "synthea-neo4j"}]
    }