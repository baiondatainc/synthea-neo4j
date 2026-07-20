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

IMPORTANT (memory-persistence fix):
  stream_qa_response() persists the conversation transcript in its Phase F block,
  which runs AFTER it yields the "end" event. If this consumer `break`s on "end",
  the async generator is suspended at that yield and never resumes — Phase F never
  runs, and the transcript stays empty (breaking follow-up rewriting on the next
  turn). We therefore `continue` on "end" and let the generator finish naturally.
  Everything we render below (answer, cypher, charts) was already captured from
  events that arrive before "end", so draining costs nothing.

CHART-DETECTION REWRITE (this version):
  The old detect_chart picked label_key = first string column and
  numeric_key = FIRST numeric column. Two failure modes:
    1. `year, month, visit_count` (no strings) → charted month vs YEAR and
       ignored the measure entirely.
    2. `state, year, month, visit_count` → x = state, y = YEAR — the flat
       "$2.0K" line was literally the year 2025/1000 with dollar formatting.
  Fixes in this version:
    - Time columns (year/month/quarter/...) are recognised and merged into a
      single sortable "period" x-axis label ("2025-11", "2025-Q3", "2025").
    - The y-axis is the MEASURE column (count/total/amount/... hints, else the
      last numeric non-time column) — never a time column.
    - A leftover category column alongside time columns (e.g. state) becomes a
      multi-series line chart (top 8 categories by total, one line each).
    - "$" formatting only applies when the y column name looks monetary
      (amount/paid/balance/...). Counts render as plain numbers.
    - isMonthLabel regex fixed — it previously reached the browser as
      /^\\d{{4}}-\\d{{2}}$/ (invalid) because fmt_fn is a plain string whose
      doubled braces were never consumed by an f-string.
    - Line charts keep up to 60 points (multi-year monthly trends); bar/pie
      keep the original 20.
"""
import json
import time
import uuid
import logging
import asyncio
from collections import OrderedDict
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from qa.chain import stream_qa_response
from config import get_settings
from guardrails import redact_text

logger = logging.getLogger(__name__)
router = APIRouter()

# chat title
# ── Title-generation (bypasses KG pipeline, no charts) ────────────────────────

def _is_title_request(body: dict, question: str) -> bool:
    q = question.lower()
    markers = (
        "title for the conversation",
        "concise, 5-word-or-less title",
        "using title case",
        "concise title",
        "detected language",
    )
    has_marker = any(m in q for m in markers)
    tiny = body.get("max_tokens", 9999) <= 12
    return has_marker or (tiny and len(q) <= 40 and "?" not in q)


async def _generate_title(question: str, model: str) -> str:
    """Short title via Ollama (llama3.2), echo fallback. No KG, no charts."""
    import httpx
    payload = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 16,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "http://localhost:11434/v1/chat/completions", json=payload
            )
            data = r.json()
            title = data["choices"][0]["message"]["content"].strip(' "\'.:\n')
    except Exception as e:
        logger.error(f"Title proxy failed: {e}")
        title = question.strip().split("\n")[-1][:60].strip(' "\'.:') or "New Conversation"
    return title or "New Conversation"


# ── Chart detection ───────────────────────────────────────────────────────────

_TIME_KEYS = {"year", "month", "quarter", "year_month", "yearmonth", "period",
              "date", "week", "day"}

_MONEY_HINTS = ("amount", "charged", "charge", "paid", "payment", "payments",
                "balance", "outstanding", "debt", "revenue", "cost", "adjust",
                "allowed", "deductible", "copay", "co_pay", "coinsurance")

_MEASURE_HINTS = _MONEY_HINTS + ("count", "total", "sum", "avg", "average",
                                 "num_", "patients", "visits", "charges",
                                 "calls", "reviews", "transactions", "rating")

_LINE_POINT_CAP = 60   # multi-year monthly trends need > 20 points
_BAR_POINT_CAP = 20
_MAX_SERIES = 8        # cap category breakdown lines


def _is_money_key(key: str) -> bool:
    k = (key or "").lower()
    return any(h in k for h in _MONEY_HINTS)


def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _time_label(row: dict, time_keys: list) -> str:
    """Build a sortable x-axis label from whatever time columns are present."""
    lk = {k.lower(): k for k in time_keys}
    if "year" in lk and "month" in lk:
        y = row.get(lk["year"])
        m = row.get(lk["month"])
        try:
            return f"{int(y)}-{int(m):02d}"
        except (TypeError, ValueError):
            return f"{y}-{m}"
    if "year" in lk and "quarter" in lk:
        return f"{row.get(lk['year'])}-Q{row.get(lk['quarter'])}"
    # single time column (year, year_month, date, ...)
    return str(row.get(time_keys[0], ""))


def _pivot_series(results: list, time_keys: list, series_key: str,
                  y_key: str) -> tuple:
    """Pivot long rows (period, category, value) into wide multi-series rows.

    Returns (data, series_names):
      data = [{"period": "2025-11", "TX": 12, "AZ": 9}, ...]
    Categories beyond the top-_MAX_SERIES by total are dropped.
    """
    totals: dict = {}
    for r in results:
        s = str(r.get(series_key, ""))[:25]
        try:
            totals[s] = totals.get(s, 0.0) + float(r.get(y_key) or 0)
        except (TypeError, ValueError):
            continue
    top = [s for s, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:_MAX_SERIES]]
    top_set = set(top)

    rows: "OrderedDict[str, dict]" = OrderedDict()
    for r in results:
        s = str(r.get(series_key, ""))[:25]
        if s not in top_set:
            continue
        x = _time_label(r, time_keys)
        row = rows.setdefault(x, {"period": x})
        try:
            row[s] = round(float(r.get(y_key) or 0), 2)
        except (TypeError, ValueError):
            row[s] = 0.0

    data = list(rows.values())[:_LINE_POINT_CAP]
    return data, top


def detect_chart(question: str, results: list) -> dict | None:
    if not results or len(results) < 2:
        return None

    first = results[0]
    keys = list(first.keys())

    time_keys = [k for k in keys if k.lower() in _TIME_KEYS]
    numeric_keys = [k for k in keys if _is_numeric(first.get(k))]
    string_keys = [k for k in keys if isinstance(first.get(k), str)]

    # ── y-axis: the measure column, never a time column ──────────────────
    measure_candidates = [k for k in numeric_keys if k.lower() not in _TIME_KEYS]
    y_key = None
    for k in measure_candidates:
        if any(h in k.lower() for h in _MEASURE_HINTS):
            y_key = k
            break
    if y_key is None and measure_candidates:
        y_key = measure_candidates[-1]   # last column is usually the aggregate
    if y_key is None:
        return None                       # nothing to plot

    is_money = _is_money_key(y_key)

    # ── chart type ────────────────────────────────────────────────────────
    q = question.lower()
    trend_words = ("trend", "over time", "monthly", "yearly", "by year",
                   "by month", "per year", "each year", "per month",
                   "each month", "timeline", "over the years", "growth")
    dist_words = ("distribution", "breakdown", "proportion", "share",
                  "gender", "race", "ethnicity", "pie", "split")

    if time_keys or any(w in q for w in trend_words):
        chart_type = "line"
    elif any(w in q for w in dist_words):
        chart_type = "pie"
    else:
        chart_type = "bar"

    title = question[:70] + ("..." if len(question) > 70 else "")

    # ── Case 1: time series ───────────────────────────────────────────────
    if time_keys and chart_type == "line":
        # a leftover string column (e.g. state) = category → multi-series
        series_key = string_keys[0] if string_keys else None

        if series_key:
            data, series = _pivot_series(results, time_keys, series_key, y_key)
            if len(data) < 2 or not series:
                return None
            logger.info(
                f"detect_chart: multi-series line — x=period, y={y_key!r}, "
                f"series={series_key!r} ({len(series)}), points={len(data)}"
            )
            return {
                "type": "line", "title": title, "data": data,
                "x_key": "period", "y_key": y_key,
                "series": series, "is_money": is_money,
            }

        # single series over time
        data = []
        for r in results[:_LINE_POINT_CAP]:
            try:
                val = round(float(r.get(y_key) or 0), 2)
            except (TypeError, ValueError):
                val = 0.0
            data.append({"period": _time_label(r, time_keys), y_key: val})
        if len({row["period"] for row in data}) < 2:
            return None
        logger.info(f"detect_chart: line — x=period, y={y_key!r}, points={len(data)}")
        return {
            "type": "line", "title": title, "data": data,
            "x_key": "period", "y_key": y_key, "is_money": is_money,
        }

    # ── Case 2: categorical (bar / pie) ───────────────────────────────────
    label_key = string_keys[0] if string_keys else None
    if label_key is None:
        # numeric label (e.g. plain year column with no trend wording)
        label_candidates = [k for k in numeric_keys if k != y_key]
        label_key = label_candidates[0] if label_candidates else None
    if label_key is None:
        return None

    data = []
    for r in results[:_BAR_POINT_CAP]:
        label = str(r.get(label_key, ""))[:35]
        try:
            val = round(float(r.get(y_key) or 0), 2)
        except (TypeError, ValueError):
            val = 0.0
        data.append({label_key: label, y_key: val})

    if len({row[label_key] for row in data}) < 2:
        return None

    logger.info(
        f"detect_chart: {chart_type} — label={label_key!r}, y={y_key!r}, rows={len(data)}"
    )
    return {
        "type": chart_type, "title": title, "data": data,
        "x_key": label_key, "y_key": y_key, "is_money": is_money,
    }


# ── Summary chart builder ─────────────────────────────────────────────────────

def _summary_charts(label: str, data: dict) -> list:
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
        # is_money intentionally omitted — build_artifact infers it from y_key.
    }


# ── Artifact builder ──────────────────────────────────────────────────────────

def build_artifact(chart: dict) -> str:
    data_json = json.dumps(chart["data"], ensure_ascii=False)
    x     = chart["x_key"]
    y     = chart["y_key"]
    title = chart["title"].replace('"', '\\"')
    ctype = chart["type"]
    series = chart.get("series") or []
    is_money = chart.get("is_money", _is_money_key(y))
    cid   = uuid.uuid4().hex[:8]

    jsx = ""

    # ── Value formatter (shared across chart types) ───────────────────────
    # Plain string (NOT an f-string) — single braces reach the browser intact.
    # IS_MONEY is substituted below; "$" only shows for monetary y columns.
    fmt_fn = """
const IS_MONEY = __IS_MONEY__;
const formatValue = (v) => {
  if (v === null || v === undefined) return '';
  const n = parseFloat(v);
  if (isNaN(n)) return v;
  let out;
  if (Math.abs(n) >= 1_000_000) out = (n / 1_000_000).toFixed(1) + 'M';
  else if (Math.abs(n) >= 1_000) out = (n / 1_000).toFixed(1) + 'K';
  else out = n.toLocaleString();
  return IS_MONEY ? '$' + out : out;
};
const formatMonth = (val) => {
  if (!val || !String(val).includes('-')) return val;
  const [y, m] = String(val).split('-');
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return (months[parseInt(m, 10) - 1] || m) + ' ' + y;
};
const isMonthLabel = (val) => val && String(val).match(/^\\d{4}-\\d{2}$/);
""".replace("__IS_MONEY__", "true" if is_money else "false")

    COLORS_JS = "const COLORS = ['#63b3ed','#68d391','#f6ad55','#fc8181','#b794f4','#76e4f7','#fbb6ce','#90cdf4'];"

    if ctype == "bar":
        jsx = f"""import {{ BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell }} from 'recharts';
const data = {data_json};
{COLORS_JS}
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
{COLORS_JS}
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

    elif ctype == "line" and series:
        # Multi-series trend (e.g. visit trends by state) — one line per category.
        series_json = json.dumps(series, ensure_ascii=False)
        jsx = f"""import {{ LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} from 'recharts';
const data = {data_json};
const SERIES = {series_json};
{COLORS_JS}
{fmt_fn}
export default function Chart() {{
  const useMonthFmt = data.length > 0 && isMonthLabel(data[0]['{x}']);
  return (
    <div style={{{{padding:'20px', background:'#1a1d2e', borderRadius:'12px', color:'#e2e8f0'}}}}>
      <h3 style={{{{marginBottom:'16px', fontSize:'15px', color:'#e2e8f0'}}}}>{title}</h3>
      <ResponsiveContainer width="100%" height={{420}}>
        <LineChart data={{data}} margin={{{{top:10, right:30, left:20, bottom:60}}}}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
          <XAxis dataKey="{x}"
            tickFormatter={{useMonthFmt ? formatMonth : (v => v)}}
            tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-35}} textAnchor="end" />
          <YAxis tickFormatter={{formatValue}} tick={{{{fill:'#a0aec0', fontSize:12}}}} width={{70}} />
          <Tooltip
            labelFormatter={{useMonthFmt ? formatMonth : (v => v)}}
            formatter={{(v, name) => [formatValue(v), name]}}
            contentStyle={{{{background:'#2d3748', border:'none', color:'#e2e8f0', borderRadius:'8px'}}}} />
          <Legend wrapperStyle={{{{color:'#a0aec0'}}}} />
          {{SERIES.map((s, i) => (
            <Line key={{s}} type="monotone" dataKey={{s}} stroke={{COLORS[i % COLORS.length]}}
              strokeWidth={{2}} dot={{{{r:3}}}} connectNulls />
          ))}}
        </LineChart>
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
            tick={{{{fill:'#a0aec0', fontSize:11}}}} angle={{-35}} textAnchor="end" />
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
    saw_end           = False

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
                # FIX: do NOT break here. stream_qa_response runs its Phase F
                # memory-persist AFTER yielding "end"; breaking would suspend the
                # generator at this yield and skip persistence, leaving the
                # transcript empty and breaking follow-up rewriting next turn.
                # Mark that we've seen the terminal event and keep draining so
                # the generator finishes (it yields nothing further and returns).
                saw_end = True
                continue

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
    if not saw_end:
        logger.warning("generate_stream: stream ended without an explicit 'end' event")
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

    if _is_title_request(body, question):
        logger.info("Title request detected — bypassing KG pipeline")
        title = await _generate_title(question, model)
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": title},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

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
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "radiologyPartner-kg", "object": "model", "created": now, "owned_by": "rp"},
            {"id": "neo4j-kg",            "object": "model", "created": now, "owned_by": "synthea-neo4j"},
        ],
    }


@router.post("/cache/clear")
async def cache_clear():
    from cache import get_answer_cache
    deleted = get_answer_cache().clear_all()
    return {"cleared": deleted}