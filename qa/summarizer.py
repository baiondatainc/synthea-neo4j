"""
qa/summarizer.py
────────────────
Summarize any graph node (Patient, Location, Campaign, etc.)
by fetching its properties + 1-hop neighbors and passing to the QA LLM.

Usage in chain.py / router:
    from qa.summarizer import summarize_node, detect_summary_request

    intent = detect_summary_request(question)
    if intent:
        async for chunk in summarize_node(intent["label"], intent["id"]):
            yield chunk

Supported trigger phrases (examples):
    "summarize patient 12345"
    "give me a summary of location SMED Radiology Location 1"
    "patient profile for P-98765"
    "tell me about campaign ABC"
    "overview of location Houston Radiology"
"""

import re
import logging
from typing import AsyncGenerator, Any

from graph.connection import Neo4jConnection
from qa.llm import get_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

# Maps trigger keywords → node label
_LABEL_KEYWORDS = {
    "patient":   "Patient",
    "location":  "Location",
    "campaign":  "Campaign",
    "practice":  "Practice",
    "insurance": "InsurancePlan",
    "plan":      "InsurancePlan",
    "review":    "BirdeyeReview",
    "charge":    "Charge",
}

# Trigger verbs/phrases that indicate a summary request
_TRIGGER_RE = re.compile(
    r'\b(summarize|summary|profile|overview|tell me about|describe|detail[s]? (of|for|about))\b',
    re.IGNORECASE,
)


def detect_summary_request(question: str) -> dict | None:
    """
    Returns {"label": "Patient", "id": "12345", "id_field": "patientId"}
    or None if this isn't a summary request.

    Handles:
      "summarize patient 12345"
      "patient profile for P-98765"
      "overview of location SMED Radiology Location 1"
      "tell me about campaign ABC Campaign"
    """
    if not _TRIGGER_RE.search(question):
        return None

    q_lower = question.lower()

    for keyword, label in _LABEL_KEYWORDS.items():
        if keyword in q_lower:
            # Extract the ID/name — everything after the keyword
            pattern = re.compile(
                rf'\b{keyword}\b\s+(?:for\s+|of\s+|about\s+|profile\s+for\s+)?([^\s,\.]+(?:\s+[^\s,\.]+)*)',
                re.IGNORECASE,
            )
            m = pattern.search(question)
            entity_id = m.group(1).strip() if m else None

            # Strip filler words from extracted ID
            if entity_id:
                for filler in ["profile", "summary", "overview", "details", "for", "of", "about"]:
                    entity_id = re.sub(rf'^\s*{filler}\s+', '', entity_id, flags=re.IGNORECASE)
                entity_id = entity_id.strip()

            id_field = _get_id_field(label)
            logger.info(f"detect_summary_request: label={label} id={entity_id!r} id_field={id_field}")
            return {"label": label, "id": entity_id, "id_field": id_field}

    return None


def _get_id_field(label: str) -> str:
    return {
        "Patient":      "patientId",
        "Location":     "name",
        "Campaign":     "name",
        "Practice":     "code",
        "InsurancePlan": "plan_name",
        "BirdeyeReview": "birdeyeId",
        "Charge":       "chargeId",
    }.get(label, "name")


# ─────────────────────────────────────────────────────────────────────────────
# Node fetchers — one per label
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_patient(patient_id: str) -> dict:
    """Fetch Patient node + key relationships."""

    # ── Core patient properties ───────────────────────────────────────────
    rows = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)
        WHERE p.patientId = $id
           OR toLower(p.patientId) = toLower($id)
        RETURN p
        LIMIT 1
        """,
        {"id": patient_id},
    )
    if not rows:
        return {}

    patient = dict(rows[0]["p"])

    # ── Recent visits ─────────────────────────────────────────────────────
    visits = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)-[:PERFORMED_AT]->(l:Location)
        WHERE p.patientId = $id
        RETURN v.visitId AS visit_id, v.admit_date AS admit_date,
               v.discharge_date AS discharge_date, l.name AS location
        ORDER BY v.admit_date DESC LIMIT 5
        """,
        {"id": patient_id},
    )

    # ── Financial summary ─────────────────────────────────────────────────
    financials = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)
        WHERE p.patientId = $id
        RETURN p.total_charged AS total_charged,
               p.total_paid    AS total_paid,
               p.outstanding_balance AS outstanding_balance,
               p.adj_bad_debt  AS bad_debt,
               p.carrier_name  AS carrier,
               p.plan_name     AS plan,
               p.payor_cohort  AS cohort
        """,
        {"id": patient_id},
    )

    # ── Call activity ─────────────────────────────────────────────────────
    calls = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)
        WHERE p.patientId = $id
        RETURN p.total_calls_window AS total_calls,
               p.has_any_calls      AS has_calls,
               p.call_tier          AS call_tier
        """,
        {"id": patient_id},
    )

    # ── Statements ────────────────────────────────────────────────────────
    statements = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:RECEIVED_STATEMENT]->(s:Statement)
        WHERE p.patientId = $id
        RETURN count(s) AS statement_count,
               sum(s.patient_balance) AS total_patient_balance
        """,
        {"id": patient_id},
    )

    # ── Top diagnoses ─────────────────────────────────────────────────────
    diagnoses = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:HAS_CHARGE]->(c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)
        WHERE p.patientId = $id
        RETURN d.code AS code, count(c) AS count
        ORDER BY count DESC LIMIT 5
        """,
        {"id": patient_id},
    )

    return {
        "node": patient,
        "visits": visits,
        "financials": financials[0] if financials else {},
        "calls": calls[0] if calls else {},
        "statements": statements[0] if statements else {},
        "diagnoses": diagnoses,
    }


def _fetch_location(location_name: str) -> dict:
    """Fetch Location node + review stats + visit counts."""

    rows = Neo4jConnection.run_query(
        """
        MATCH (l:Location)
        WHERE toLower(l.name) CONTAINS toLower($name)
           OR l.locationId = $name
        RETURN l LIMIT 1
        """,
        {"name": location_name},
    )
    if not rows:
        return {}

    location = dict(rows[0]["l"])

    reviews = Neo4jConnection.run_query(
        """
        MATCH (b:BirdeyeReview)-[:REVIEWS]->(l:Location)
        WHERE toLower(l.name) CONTAINS toLower($name)
        RETURN count(b) AS review_count,
               round(avg(b.rating), 2) AS avg_rating,
               sum(CASE WHEN b.rating = 1 THEN 1 ELSE 0 END) AS one_star_count
        """,
        {"name": location_name},
    )

    visits = Neo4jConnection.run_query(
        """
        MATCH (v:Visit)-[:PERFORMED_AT]->(l:Location)
        WHERE toLower(l.name) CONTAINS toLower($name)
        RETURN count(v) AS visit_count,
               min(v.admit_date) AS first_visit,
               max(v.admit_date) AS last_visit
        """,
        {"name": location_name},
    )

    top_patients = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)-[:PERFORMED_AT]->(l:Location)
        WHERE toLower(l.name) CONTAINS toLower($name)
        RETURN count(DISTINCT p) AS patient_count
        """,
        {"name": location_name},
    )

    return {
        "node": location,
        "reviews": reviews[0] if reviews else {},
        "visits": visits[0] if visits else {},
        "patient_count": top_patients[0] if top_patients else {},
    }


def _fetch_campaign(campaign_name: str) -> dict:
    """Fetch Campaign node + call stats + attributed patients."""

    rows = Neo4jConnection.run_query(
        """
        MATCH (c:Campaign)
        WHERE toLower(c.name) CONTAINS toLower($name)
           OR c.campaignId = $name
        RETURN c LIMIT 1
        """,
        {"name": campaign_name},
    )
    if not rows:
        return {}

    campaign = dict(rows[0]["c"])

    call_stats = Neo4jConnection.run_query(
        """
        MATCH (rc:RCCall)-[:PART_OF_CAMPAIGN]->(c:Campaign)
        WHERE toLower(c.name) CONTAINS toLower($name)
        RETURN count(rc) AS total_calls,
               round(avg(rc.total_time), 2) AS avg_call_time,
               sum(CASE WHEN rc.abandon = true THEN 1 ELSE 0 END) AS abandoned_calls
        """,
        {"name": campaign_name},
    )

    patient_stats = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:IDENTIFIED_BY_PHONE]->(pb:PhoneBridge)
              <-[:ATTRIBUTED_TO_PHONE]-(rc:RCCall)-[:PART_OF_CAMPAIGN]->(c:Campaign)
        WHERE toLower(c.name) CONTAINS toLower($name)
        RETURN count(DISTINCT p) AS patient_count,
               round(avg(p.outstanding_balance), 2) AS avg_outstanding_balance,
               round(sum(p.outstanding_balance), 2) AS total_outstanding_balance
        """,
        {"name": campaign_name},
    )

    return {
        "node": campaign,
        "call_stats": call_stats[0] if call_stats else {},
        "patient_stats": patient_stats[0] if patient_stats else {},
    }


def _fetch_practice(practice_code: str) -> dict:
    rows = Neo4jConnection.run_query(
        """
        MATCH (pr:Practice)
        WHERE pr.code = $code OR pr.practiceId = $code
        RETURN pr LIMIT 1
        """,
        {"code": practice_code},
    )
    if not rows:
        return {}

    practice = dict(rows[0]["pr"])

    stats = Neo4jConnection.run_query(
        """
        MATCH (p:Patient)-[:REGISTERED_AT]->(pr:Practice)
        WHERE pr.code = $code
        RETURN count(p) AS patient_count,
               round(sum(p.outstanding_balance), 2) AS total_outstanding,
               round(sum(p.adj_bad_debt), 2) AS total_bad_debt
        """,
        {"code": practice_code},
    )

    locations = Neo4jConnection.run_query(
        """
        MATCH (l:Location)-[:BELONGS_TO_PRACTICE]->(pr:Practice)
        WHERE pr.code = $code
        RETURN l.name AS location, l.city AS city, l.state AS state
        """,
        {"code": practice_code},
    )

    return {
        "node": practice,
        "stats": stats[0] if stats else {},
        "locations": locations,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_FETCHERS = {
    "Patient":      _fetch_patient,
    "Location":     _fetch_location,
    "Campaign":     _fetch_campaign,
    "Practice":     _fetch_practice,
}


def fetch_node_context(label: str, entity_id: str) -> dict:
    """Fetch all relevant data for a node. Returns empty dict if not found."""
    fetcher = _FETCHERS.get(label)
    if not fetcher:
        logger.warning(f"No fetcher registered for label={label}")
        return {}
    try:
        return fetcher(entity_id)
    except Exception as e:
        logger.error(f"fetch_node_context failed: label={label} id={entity_id} error={e}", exc_info=True)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates per label
# ─────────────────────────────────────────────────────────────────────────────

_PATIENT_PROMPT = """You are a healthcare data analyst. Summarize this patient's profile clearly and concisely.
Be factual. Do not invent information not in the data.

Patient Data:
{context}

Structure your response with these sections:

## Patient Profile
Markdown table with columns: Field | Value
Rows: Patient ID, Date of Birth, Gender, Address, Bad Address Flag.

## Financial Summary
Markdown table with columns: Metric | Amount
Rows: Total Charged, Total Paid, Outstanding Balance, Bad Debt, Carrier, Plan, Payor Cohort.

## Patient Flags
Bullet points (only show flags that are True): Self-Pay, Has Insurance, BAI, Catastrophe, Friction, Clean, Is SAPA, Is Tennessee, Is Atlanta 404, Multi-Practice.

## Call Activity
Bullet points: Call Tier, Has Any Calls, Total Calls in Window.

## Statement History
Bullet points: Statement Count, Total Patient Balance on Statements.

## Recent Visits
Markdown table with columns: Location | Admit Date | Discharge Date
Show up to 5 most recent visits.

## Top Diagnosis Codes
Markdown table with columns: Diagnosis Code | Charge Count

Summary:"""


_LOCATION_PROMPT = """You are a healthcare data analyst. Summarize this location's profile.
Be factual. Do not invent information not in the data.

Location Data:
{context}

Structure your response with these sections:

## Location Profile
Markdown table with columns: Field | Value
Rows: Name, Type, Address, City, State, ZIP, NPI, Phone.

## Visit & Patient Stats
Markdown table with columns: Metric | Value
Rows: Total Visits, Total Patients, First Visit, Last Visit.

## Birdeye Review Stats
Markdown table with columns: Metric | Value
Rows: Total Reviews, Avg Rating, 1-Star Reviews.

## Notable Insights
2-3 bullet points.

Summary:"""


_CAMPAIGN_PROMPT = """You are a healthcare data analyst. Summarize this campaign's performance.
Be factual. Do not invent information not in the data.

Campaign Data:
{context}

Structure your response with these sections:

## Campaign Profile
Bullet points: Campaign Name, Campaign ID, Notes.

## Call Statistics
Markdown table with columns: Metric | Value
Rows: Total Calls, Avg Call Time (s), Abandoned Calls.

## Patient Reach
Markdown table with columns: Metric | Value
Rows: Patients Reached, Avg Outstanding Balance, Total Outstanding Balance.

## Notable Insights
2-3 bullet points.

Summary:"""


_PRACTICE_PROMPT = """You are a healthcare data analyst. Summarize this practice.
Be factual. Do not invent information not in the data.

Practice Data:
{context}

Structure your response with these sections:

## Practice Profile
Bullet points: Practice Code, Practice ID.

## Financial Performance
Markdown table with columns: Metric | Value
Rows: Patient Count, Total Outstanding Balance, Total Bad Debt.

## Locations
Markdown table with columns: Location | City | State

## Notable Insights
2-3 bullet points.

Summary:"""


_PROMPTS = {
    "Patient":   _PATIENT_PROMPT,
    "Location":  _LOCATION_PROMPT,
    "Campaign":  _CAMPAIGN_PROMPT,
    "Practice":  _PRACTICE_PROMPT,
}

_DEFAULT_PROMPT = """You are a healthcare data analyst. Summarize this node's data.

Data:
{context}

Provide a clear, factual bullet-point summary.

Summary:"""


# ─────────────────────────────────────────────────────────────────────────────
# Main summarizer
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_node(
    label: str,
    entity_id: str,
) -> AsyncGenerator[dict, None]:
    """
    Stream a summary of any graph node.

    Yields the same chunk format as stream_qa_response:
      {"type": "token", "data": "..."}
      {"type": "end",   "data": ""}
      {"type": "error", "data": "..."}
    """
    if not entity_id:
        yield {"type": "error", "data": f"Could not extract a {label} ID from your question. Please include the ID or name."}
        return

    logger.info(f"summarize_node: label={label} id={entity_id!r}")

    # ── 1. Fetch context from Neo4j ───────────────────────────────────────
    context = fetch_node_context(label, entity_id)
    if not context:
        yield {
            "type": "token",
            "data": f"No {label} found matching **{entity_id}**. Please check the ID or name and try again.",
        }
        yield {"type": "end", "data": ""}
        return

    # ── 2. Format context as readable text ────────────────────────────────
    context_text = _format_context(label, context)
    logger.debug(f"summarize_node context ({len(context_text)} chars):\n{context_text[:500]}")

    # ── 3. Build prompt ───────────────────────────────────────────────────
    prompt_template = _PROMPTS.get(label, _DEFAULT_PROMPT)
    prompt = prompt_template.format(context=context_text)

    # ── 4. Stream through QA LLM ──────────────────────────────────────────
    try:
        import asyncio
        llm = get_llm(streaming=True)
        queue: asyncio.Queue = asyncio.Queue()

        from langchain_core.callbacks import AsyncCallbackHandler
        from langchain_core.outputs import LLMResult

        class _CB(AsyncCallbackHandler):
            async def on_llm_new_token(self, token: str, **kw: Any) -> None:
                await queue.put({"type": "token", "data": token})
            async def on_llm_end(self, response: LLMResult, **kw: Any) -> None:
                await queue.put({"type": "end", "data": ""})
            async def on_llm_error(self, error: Exception, **kw: Any) -> None:
                await queue.put({"type": "error", "data": str(error)})

        llm.callbacks = [_CB()]

        async def _run():
            try:
                await llm.ainvoke(prompt)
            except Exception as e:
                await queue.put({"type": "error", "data": str(e)})

        task = asyncio.create_task(_run())

        while True:
            item = await queue.get()
            yield item
            if item["type"] in ("end", "error"):
                break

        await task

    except Exception as e:
        logger.error(f"summarize_node LLM error: {e}", exc_info=True)
        yield {"type": "error", "data": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Context formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_context(label: str, context: dict) -> str:
    """Convert fetched context dict to readable text for the LLM prompt."""
    lines = []

    if label == "Patient":
        node = context.get("node", {})
        fin  = context.get("financials", {})
        calls = context.get("calls", {})
        stmts = context.get("statements", {})
        visits = context.get("visits", [])
        diagnoses = context.get("diagnoses", [])

        lines += [
            "=== PATIENT PROFILE ===",
            f"Patient ID     : {node.get('patientId', 'N/A')}",
            f"Name           : {node.get('first_name', '')} {node.get('last_name', '')}".strip(),
            f"DOB            : {node.get('dob', 'N/A')}",
            f"Gender         : {node.get('gender', 'N/A')}",
            f"Race/Ethnicity : {node.get('race', 'N/A')} / {node.get('ethnicity', 'N/A')}",
            f"Address        : {node.get('city', '')}, {node.get('state', '')} {node.get('zip', '')}",
            f"Bad Address    : {node.get('bad_address_indicator', False)}",
            "",
            "=== FINANCIAL ===",
            f"Total Charged  : ${_fmt(fin.get('total_charged'))}",
            f"Total Paid     : ${_fmt(fin.get('total_paid'))}",
            f"Outstanding    : ${_fmt(fin.get('outstanding_balance'))}",
            f"Bad Debt       : ${_fmt(fin.get('bad_debt'))}",
            f"Carrier        : {fin.get('carrier', 'N/A')}",
            f"Plan           : {fin.get('plan', 'N/A')}",
            f"Payor Cohort   : {fin.get('cohort', 'N/A')}",
            "",
            "=== FLAGS ===",
            f"Self-Pay       : {node.get('is_self_pay', False)}",
            f"Has Insurance  : {node.get('has_insurance', False)}",
            f"Is BAI         : {node.get('is_bai', False)}",
            f"Catastrophe    : {node.get('is_catastrophe', False)}",
            f"Friction       : {node.get('is_friction', False)}",
            f"Is Clean       : {node.get('is_clean', False)}",
            f"Is SAPA        : {node.get('is_sapa', False)}",
            f"Is Tennessee   : {node.get('is_tennessee', False)}",
            "",
            "=== CALLS ===",
            f"Call Tier      : {calls.get('call_tier', 'N/A')}",
            f"Has Any Calls  : {calls.get('has_calls', False)}",
            f"Total Calls    : {calls.get('total_calls', 0)}",
            "",
            "=== STATEMENTS ===",
            f"Statement Count        : {stmts.get('statement_count', 0)}",
            f"Total Patient Balance  : ${_fmt(stmts.get('total_patient_balance'))}",
            "",
            "=== RECENT VISITS ===",
        ]
        for v in visits:
            lines.append(
                f"  - {v.get('location', 'N/A')} | Admit: {v.get('admit_date', 'N/A')} | Discharge: {v.get('discharge_date', 'N/A')}"
            )

        lines += ["", "=== TOP DIAGNOSES ==="]
        for d in diagnoses:
            lines.append(f"  - {d.get('code', 'N/A')} ({d.get('count', 0)} charges)")

    elif label == "Location":
        node = context.get("node", {})
        rev  = context.get("reviews", {})
        vis  = context.get("visits", {})
        pat  = context.get("patient_count", {})

        lines += [
            "=== LOCATION PROFILE ===",
            f"Name           : {node.get('name', 'N/A')}",
            f"Type           : {node.get('location_type', 'N/A')}",
            f"Address        : {node.get('address', '')}, {node.get('city', '')}, {node.get('state', '')} {node.get('zip', '')}",
            f"NPI            : {node.get('npi', 'N/A')}",
            f"Phone          : {node.get('phone_norm', 'N/A')}",
            "",
            "=== VISIT STATS ===",
            f"Total Visits   : {vis.get('visit_count', 0):,}",
            f"Total Patients : {pat.get('patient_count', 0):,}",
            f"First Visit    : {vis.get('first_visit', 'N/A')}",
            f"Last Visit     : {vis.get('last_visit', 'N/A')}",
            "",
            "=== BIRDEYE REVIEWS ===",
            f"Total Reviews  : {rev.get('review_count', 0)}",
            f"Avg Rating     : {rev.get('avg_rating', 'N/A')}",
            f"1-Star Count   : {rev.get('one_star_count', 0)}",
        ]

    elif label == "Campaign":
        node = context.get("node", {})
        cs   = context.get("call_stats", {})
        ps   = context.get("patient_stats", {})

        lines += [
            "=== CAMPAIGN PROFILE ===",
            f"Name           : {node.get('name', 'N/A')}",
            f"Campaign ID    : {node.get('campaignId', 'N/A')}",
            f"Notes          : {node.get('notes', 'N/A')}",
            "",
            "=== CALL STATS ===",
            f"Total Calls    : {cs.get('total_calls', 0):,}",
            f"Avg Call Time  : {cs.get('avg_call_time', 0)}s",
            f"Abandoned      : {cs.get('abandoned_calls', 0):,}",
            "",
            "=== PATIENT REACH ===",
            f"Patients Reached       : {ps.get('patient_count', 0):,}",
            f"Avg Outstanding Balance: ${_fmt(ps.get('avg_outstanding_balance'))}",
            f"Total Outstanding      : ${_fmt(ps.get('total_outstanding_balance'))}",
        ]

    elif label == "Practice":
        node = context.get("node", {})
        st   = context.get("stats", {})
        locs = context.get("locations", [])

        lines += [
            "=== PRACTICE PROFILE ===",
            f"Code           : {node.get('code', 'N/A')}",
            f"Practice ID    : {node.get('practiceId', 'N/A')}",
            "",
            "=== STATS ===",
            f"Patient Count  : {st.get('patient_count', 0):,}",
            f"Total Outstanding : ${_fmt(st.get('total_outstanding'))}",
            f"Total Bad Debt    : ${_fmt(st.get('total_bad_debt'))}",
            "",
            "=== LOCATIONS ===",
        ]
        for loc in locs:
            lines.append(f"  - {loc.get('location', 'N/A')} | {loc.get('city', '')}, {loc.get('state', '')}")

    else:
        # Generic fallback
        lines.append(str(context))

    return "\n".join(lines)


def _fmt(value) -> str:
    """Format a numeric value with commas and 2dp, or N/A."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)