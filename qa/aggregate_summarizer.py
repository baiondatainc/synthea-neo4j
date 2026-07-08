"""
qa/aggregate_summarizer.py
──────────────────────────
Summarize ALL nodes of a given label — portfolio-level insights.

Trigger phrases:
    "summarize all patients"
    "give me a patient portfolio summary"
    "overview of all locations"
    "practice summary"
    "campaign performance summary"
    "summarize all birdeye reviews"
    "financial overview"
    "complete financial summary"
    "how are we doing overall"

Wire into chain.py BEFORE the single-node summarizer check:

    from qa.aggregate_summarizer import detect_aggregate_summary, summarize_all_nodes

    agg_intent = detect_aggregate_summary(question)
    if agg_intent:
        async for chunk in summarize_all_nodes(agg_intent["label"]):
            yield chunk
        return
"""

import re
import logging
import asyncio
from typing import AsyncGenerator, Any

from graph.connection import Neo4jConnection
from qa.llm import get_llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: financial phrase check — highest priority, checked first
_FINANCIAL_PHRASES = re.compile(
    r'\b(financial summary|financial overview|complete financial|'
    r'revenue summary|collections overview|overall financial|'
    r'how are we doing financially|finance summary|financial report)\b',
    re.IGNORECASE,
)

# Phase 2: aggregate trigger words
_AGG_TRIGGERS = re.compile(
    r'\b(all|overall|portfolio|entire|whole|every|across all|aggregate|'
    r'how are we doing|full summary|complete summary|complete|high.?level|'
    r'performance summary|executive summary|give me a|show me a|provide a)\b',
    re.IGNORECASE,
)

# Phase 3: summary intent words
_SUMMARY_INTENT = re.compile(
    r'\b(summarize|summary|overview|profile|how are we|performance|report|breakdown)\b',
    re.IGNORECASE,
)

# Single-word label map only — multi-word phrases handled separately above
_LABEL_MAP = {
    "patient":       "Patient",
    "patients":      "Patient",
    "location":      "Location",
    "locations":     "Location",
    "campaign":      "Campaign",
    "campaigns":     "Campaign",
    "practice":      "Practice",
    "practices":     "Practice",
    "review":        "BirdeyeReview",
    "reviews":       "BirdeyeReview",
    "birdeye":       "BirdeyeReview",
    "charge":        "Charge",
    "charges":       "Charge",
    "transaction":   "Transaction",
    "transactions":  "Transaction",
    "financial":     "Financial",
    "finance":       "Financial",
    "finances":      "Financial",
    "revenue":       "Financial",
    "collections":   "Financial",
    "outstanding":   "Financial",
}


def detect_aggregate_summary(question: str) -> dict | None:
    """
    Returns {"label": "Patient"} or None.

    Detection order (first match wins):
      1. Financial phrase match  → Financial  (highest priority)
      2. Label keyword match     → specific label
      3. Fallback trigger match  → Financial  (lowest priority)

    Guard: if a specific ID follows the label keyword, returns None
           so single-node summarizer handles it instead.
    """
    q_lower = question.lower()

    # ── 1. Financial phrase — check before anything else ─────────────────
    # Catches: "complete financial summary", "financial overview", etc.
    if _FINANCIAL_PHRASES.search(question):
        logger.info("detect_aggregate_summary: financial phrase match → Financial")
        return {"label": "Financial"}

    has_agg_trigger    = bool(_AGG_TRIGGERS.search(question))
    has_summary_intent = bool(_SUMMARY_INTENT.search(question))

    # Need at least one trigger or summary intent to proceed
    if not (has_agg_trigger or has_summary_intent):
        return None

    # ── Intent words that look like IDs but are NOT ──────────────────────
    _INTENT_WORDS = {
        "overview", "summary", "profile", "report", "breakdown", "details",
        "information", "analysis", "data", "insight", "performance",
        "statistics", "stats", "metrics", "all", "complete", "full",
    }

    # ── 2. Label keyword match ────────────────────────────────────────────
    for keyword, label in _LABEL_MAP.items():
        if re.search(rf'\b{re.escape(keyword)}\b', q_lower):
            # Guard: only defer to single-node summarizer if a REAL ID follows
            # Real IDs contain digits (12345, P-98765) OR are all-uppercase (RADM)
            # Plain English words like "overview", "summary" must NOT trigger this guard
            id_after = re.compile(
                rf'\b{re.escape(keyword)}\b\s+(?:for\s+|of\s+)?([A-Za-z0-9][A-Za-z0-9\-:_]{{3,}})',
                re.IGNORECASE,
            )
            m = id_after.search(question)
            is_real_id = False
            if m:
                candidate = m.group(1)
                if candidate.lower() not in _INTENT_WORDS:
                    # Contains digit → real ID (e.g. 12345, P-98765, RADM:1000000)
                    # Contains colon or dash with digits → composite ID (e.g. RADM:1000000)
                    # Pure uppercase short codes (≤6 chars) like RADM, SMED, TRI are
                    # ambiguous — require digits to confirm they are IDs not location names
                    has_digit = bool(re.search(r'\d', candidate))
                    has_colon = ':'  in candidate
                    if has_digit or has_colon:
                        is_real_id = True
            if is_real_id:
                logger.info(
                    f"detect_aggregate_summary: label={label} but specific ID "
                    f"'{m.group(1)}' found → deferring to single-node summarizer"
                )
                return None
            logger.info(f"detect_aggregate_summary: keyword={keyword!r} → label={label}")
            return {"label": label}

    # ── 3. Fallback: generic trigger + summary intent → Financial ─────────
    # Catches: "how are we doing overall", "give me an overall summary"
    if has_agg_trigger and has_summary_intent:
        logger.info("detect_aggregate_summary: generic fallback → Financial")
        return {"label": "Financial"}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate fetchers — one per label
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_all_patients() -> dict:
    """Portfolio-level patient stats."""

    totals = Neo4jConnection.run_query("""
        MATCH (p:Patient)
        RETURN count(p)                              AS total_patients,
               round(sum(p.total_charged), 2)        AS total_charged,
               round(sum(p.total_paid), 2)            AS total_paid,
               round(sum(p.outstanding_balance), 2)  AS total_outstanding,
               round(sum(p.adj_bad_debt), 2)          AS total_bad_debt,
               round(avg(p.outstanding_balance), 2)  AS avg_outstanding
    """)

    flags = Neo4jConnection.run_query("""
        MATCH (p:Patient)
        RETURN
          sum(CASE WHEN p.is_self_pay = true THEN 1 ELSE 0 END)            AS self_pay_count,
          sum(CASE WHEN p.has_insurance = true THEN 1 ELSE 0 END)           AS insured_count,
          sum(CASE WHEN p.is_bai = true THEN 1 ELSE 0 END)                  AS bai_count,
          sum(CASE WHEN p.is_catastrophe = true THEN 1 ELSE 0 END)          AS catastrophe_count,
          sum(CASE WHEN p.is_friction = true THEN 1 ELSE 0 END)             AS friction_count,
          sum(CASE WHEN p.is_clean = true THEN 1 ELSE 0 END)                AS clean_count,
          sum(CASE WHEN p.has_any_calls = true THEN 1 ELSE 0 END)           AS called_count,
          sum(CASE WHEN p.bad_address_indicator = true THEN 1 ELSE 0 END)   AS bad_address_count,
          sum(CASE WHEN p.multi_practice_flag = true THEN 1 ELSE 0 END)     AS multi_practice_count,
          count(p)                                                           AS total_for_pct
    """)

    by_state = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.state IS NOT NULL
        RETURN p.state AS state, count(p) AS patient_count,
               round(sum(p.outstanding_balance), 2) AS outstanding
        ORDER BY patient_count DESC LIMIT 10
    """)

    by_cohort = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.payor_cohort IS NOT NULL
        RETURN p.payor_cohort AS cohort, count(p) AS count,
               round(sum(p.outstanding_balance), 2) AS outstanding
        ORDER BY count DESC LIMIT 10
    """)

    by_call_tier = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.call_tier IS NOT NULL
        RETURN p.call_tier AS tier, count(p) AS count
        ORDER BY count DESC
    """)

    by_gender = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.gender IS NOT NULL
        RETURN p.gender AS gender, count(p) AS count
        ORDER BY count DESC
    """)

    return {
        "totals":       totals[0]    if totals    else {},
        "flags":        flags[0]     if flags     else {},
        "by_state":     by_state,
        "by_cohort":    by_cohort,
        "by_call_tier": by_call_tier,
        "by_gender":    by_gender,
    }


def _fetch_all_locations() -> dict:
    totals = Neo4jConnection.run_query("""
        MATCH (l:Location)
        RETURN count(l) AS total_locations
    """)

    visit_stats = Neo4jConnection.run_query("""
        MATCH (v:Visit)-[:PERFORMED_AT]->(l:Location)
        RETURN l.name AS location, count(v) AS visit_count
        ORDER BY visit_count DESC LIMIT 10
    """)

    review_stats = Neo4jConnection.run_query("""
        MATCH (b:BirdeyeReview)-[:REVIEWS]->(l:Location)
        RETURN l.name AS location,
               count(b) AS review_count,
               round(avg(b.rating), 2) AS avg_rating
        ORDER BY review_count DESC LIMIT 10
    """)

    overall_reviews = Neo4jConnection.run_query("""
        MATCH (b:BirdeyeReview)-[:REVIEWS]->(l:Location)
        RETURN count(b) AS total_reviews,
               round(avg(b.rating), 2) AS overall_avg_rating,
               sum(CASE WHEN b.rating = 1 THEN 1 ELSE 0 END) AS one_star_count,
               sum(CASE WHEN b.rating = 5 THEN 1 ELSE 0 END) AS five_star_count
    """)

    patient_by_location = Neo4jConnection.run_query("""
        MATCH (p:Patient)-[:HAD_VISIT]->(v:Visit)-[:PERFORMED_AT]->(l:Location)
        RETURN l.name AS location, count(DISTINCT p) AS patient_count
        ORDER BY patient_count DESC LIMIT 10
    """)

    return {
        "totals":              totals[0]         if totals         else {},
        "visit_stats":         visit_stats,
        "review_stats":        review_stats,
        "overall_reviews":     overall_reviews[0] if overall_reviews else {},
        "patient_by_location": patient_by_location,
    }


def _fetch_all_campaigns() -> dict:
    totals = Neo4jConnection.run_query("""
        MATCH (c:Campaign)
        RETURN count(c) AS total_campaigns
    """)

    call_stats = Neo4jConnection.run_query("""
        MATCH (rc:RCCall)-[:PART_OF_CAMPAIGN]->(c:Campaign)
        RETURN c.name AS campaign,
               count(rc) AS total_calls,
               round(avg(rc.total_time), 2) AS avg_call_time,
               sum(CASE WHEN rc.abandon = true THEN 1 ELSE 0 END) AS abandoned
        ORDER BY total_calls DESC LIMIT 10
    """)

    overall_calls = Neo4jConnection.run_query("""
        MATCH (rc:RCCall)-[:PART_OF_CAMPAIGN]->(c:Campaign)
        RETURN count(rc) AS total_calls,
               round(avg(rc.total_time), 2) AS avg_call_time,
               sum(CASE WHEN rc.abandon = true THEN 1 ELSE 0 END) AS total_abandoned,
               count(DISTINCT c) AS campaigns_with_calls
    """)

    return {
        "totals":        totals[0]        if totals        else {},
        "overall_calls": overall_calls[0] if overall_calls else {},
        "call_stats":    call_stats,
    }


def _fetch_all_practices() -> dict:
    totals = Neo4jConnection.run_query("""
        MATCH (pr:Practice)
        RETURN count(pr) AS total_practices
    """)

    by_practice = Neo4jConnection.run_query("""
        MATCH (p:Patient)-[:REGISTERED_AT]->(pr:Practice)
        RETURN pr.code AS practice,
               count(p) AS patient_count,
               round(sum(p.outstanding_balance), 2) AS total_outstanding,
               round(sum(p.adj_bad_debt), 2) AS total_bad_debt
        ORDER BY patient_count DESC
    """)

    return {
        "totals":      totals[0] if totals else {},
        "by_practice": by_practice,
    }


def _fetch_all_reviews() -> dict:
    overall = Neo4jConnection.run_query("""
        MATCH (b:BirdeyeReview)
        RETURN count(b) AS total_reviews,
               round(avg(b.rating), 2) AS avg_rating,
               sum(CASE WHEN b.rating = 1 THEN 1 ELSE 0 END) AS one_star,
               sum(CASE WHEN b.rating = 2 THEN 1 ELSE 0 END) AS two_star,
               sum(CASE WHEN b.rating = 3 THEN 1 ELSE 0 END) AS three_star,
               sum(CASE WHEN b.rating = 4 THEN 1 ELSE 0 END) AS four_star,
               sum(CASE WHEN b.rating = 5 THEN 1 ELSE 0 END) AS five_star,
               sum(CASE WHEN b.phi_flagged = true THEN 1 ELSE 0 END) AS phi_flagged_count
    """)

    by_source = Neo4jConnection.run_query("""
        MATCH (b:BirdeyeReview) WHERE b.source IS NOT NULL
        RETURN b.source AS source, count(b) AS count,
               round(avg(b.rating), 2) AS avg_rating
        ORDER BY count DESC
    """)

    by_location = Neo4jConnection.run_query("""
        MATCH (b:BirdeyeReview)-[:REVIEWS]->(l:Location)
        RETURN l.name AS location,
               count(b) AS review_count,
               round(avg(b.rating), 2) AS avg_rating
        ORDER BY avg_rating ASC LIMIT 10
    """)

    return {
        "overall":     overall[0] if overall else {},
        "by_source":   by_source,
        "by_location": by_location,
    }


def _fetch_financial_overview() -> dict:
    """Cross-node financial summary."""

    patient_fin = Neo4jConnection.run_query("""
        MATCH (p:Patient)
        RETURN round(sum(p.total_charged), 2)         AS total_charged,
               round(sum(p.total_paid), 2)             AS total_paid,
               round(sum(p.outstanding_balance), 2)   AS total_outstanding,
               round(sum(p.adj_bad_debt), 2)           AS total_bad_debt,
               round(sum(p.adj_contractual), 2)        AS total_contractual,
               round(sum(p.adj_collection_agency), 2) AS total_collection_agency
    """)

    by_state = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.state IS NOT NULL
        RETURN p.state AS state,
               round(sum(p.outstanding_balance), 2) AS outstanding,
               round(sum(p.adj_bad_debt), 2) AS bad_debt
        ORDER BY outstanding DESC LIMIT 10
    """)

    by_cohort = Neo4jConnection.run_query("""
        MATCH (p:Patient) WHERE p.payor_cohort IS NOT NULL
        RETURN p.payor_cohort AS cohort,
               count(p) AS patient_count,
               round(sum(p.outstanding_balance), 2) AS outstanding,
               round(sum(p.adj_bad_debt), 2) AS bad_debt
        ORDER BY outstanding DESC LIMIT 10
    """)

    charge_summary = Neo4jConnection.run_query("""
        MATCH (c:Charge)
        RETURN count(c) AS total_charges,
               round(sum(c.charge_amount), 2) AS total_charge_amount,
               round(avg(c.charge_amount), 2) AS avg_charge_amount
    """)

    # ── FIX: separate query for charges by year ───────────────────────────
    charges_by_year = Neo4jConnection.run_query("""
        MATCH (c:Charge) WHERE c.service_date IS NOT NULL
        RETURN c.service_date.year AS year,
               count(c) AS charge_count,
               round(sum(c.charge_amount), 2) AS total_amount
        ORDER BY year
    """)

    transaction_summary = Neo4jConnection.run_query("""
        MATCH (t:Transaction)
        RETURN count(t) AS total_transactions,
               round(sum(t.payment_amount), 2) AS total_payments,
               round(sum(t.bad_debt_adjustments), 2) AS total_bad_debt_adj
    """)

    # ── FIX: separate query for payments by year ──────────────────────────
    payments_by_year = Neo4jConnection.run_query("""
        MATCH (t:Transaction) WHERE t.post_date IS NOT NULL
        RETURN t.post_date.year AS year,
               round(sum(t.payment_amount), 2) AS total_payments,
               count(t) AS transaction_count
        ORDER BY year
    """)

    return {
        "patient_fin":        patient_fin[0]        if patient_fin        else {},
        "by_state":           by_state,
        "by_cohort":          by_cohort,
        "charge_summary":     charge_summary[0]     if charge_summary     else {},
        "charges_by_year":    charges_by_year,
        "transaction_summary": transaction_summary[0] if transaction_summary else {},
        "payments_by_year":   payments_by_year,
    }


def _fetch_all_charges() -> dict:
    overall = Neo4jConnection.run_query("""
        MATCH (c:Charge)
        RETURN count(c) AS total_charges,
               round(sum(c.charge_amount), 2) AS total_amount,
               round(avg(c.charge_amount), 2) AS avg_amount
    """)

    by_modality = Neo4jConnection.run_query("""
        MATCH (c:Charge) WHERE c.procedure_modality IS NOT NULL
        RETURN c.procedure_modality AS modality,
               count(c) AS charge_count,
               round(sum(c.charge_amount), 2) AS total_amount
        ORDER BY total_amount DESC
    """)

    by_year = Neo4jConnection.run_query("""
        MATCH (c:Charge) WHERE c.service_date IS NOT NULL
        RETURN c.service_date.year AS year,
               count(c) AS charge_count,
               round(sum(c.charge_amount), 2) AS total_amount
        ORDER BY year
    """)

    top_diagnoses = Neo4jConnection.run_query("""
        MATCH (c:Charge)-[:DIAGNOSED_WITH]->(d:DiagnosisCode)
        RETURN d.code AS code, count(c) AS count
        ORDER BY count DESC LIMIT 10
    """)

    return {
        "overall":       overall[0] if overall else {},
        "by_modality":   by_modality,
        "by_year":       by_year,
        "top_diagnoses": top_diagnoses,
    }


def _fetch_all_transactions() -> dict:
    overall = Neo4jConnection.run_query("""
        MATCH (t:Transaction)
        RETURN count(t) AS total_transactions,
               round(sum(t.payment_amount), 2) AS total_payments,
               round(sum(t.adjustment_amount), 2) AS total_adjustments,
               round(sum(t.bad_debt_adjustments), 2) AS total_bad_debt,
               round(avg(t.payment_amount), 2) AS avg_payment
    """)

    by_year = Neo4jConnection.run_query("""
        MATCH (t:Transaction) WHERE t.post_date IS NOT NULL
        RETURN t.post_date.year AS year,
               round(sum(t.payment_amount), 2) AS total_payments,
               count(t) AS transaction_count
        ORDER BY year
    """)

    by_type = Neo4jConnection.run_query("""
        MATCH (t:Transaction) WHERE t.transaction_type IS NOT NULL
        RETURN t.transaction_type AS type,
               count(t) AS count,
               round(sum(t.payment_amount), 2) AS total_payments
        ORDER BY count DESC LIMIT 10
    """)

    return {
        "overall": overall[0] if overall else {},
        "by_year": by_year,
        "by_type": by_type,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_AGG_FETCHERS = {
    "Patient":       _fetch_all_patients,
    "Location":      _fetch_all_locations,
    "Campaign":      _fetch_all_campaigns,
    "Practice":      _fetch_all_practices,
    "BirdeyeReview": _fetch_all_reviews,
    "Financial":     _fetch_financial_overview,
    "Charge":        _fetch_all_charges,
    "Transaction":   _fetch_all_transactions,
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_PROMPTS = {
    "Patient": """You are a healthcare data analyst. Provide an executive summary of the entire patient portfolio.

IMPORTANT: All flag values below are COUNTS of patients (integers), NOT boolean flags.
For example "Self-Pay patients: 27,176" means 27,176 patients are self-pay.
Report each count as a number. Never say "no instances" unless the count is 0.

Data:
{context}

Structure your response with these sections:

## Portfolio Overview
Bullet points: total patients, total charged, total paid, collection rate, outstanding balance, bad debt, avg outstanding per patient.

## Patient Flags
Bullet points with counts and percentages: self-pay, insured, BAI, catastrophe, friction, clean, has calls, bad address, multi-practice.

## Patients by State
Markdown table with columns: State | Patients | Outstanding Balance
Show top 10 states.

## Patients by Payor Cohort
Markdown table with columns: Cohort | Patients | Outstanding Balance
Show all cohorts.

## Call Tier Breakdown
Markdown table with columns: Call Tier | Patient Count

## Gender Split
Markdown table with columns: Gender | Count

## Notable Insights
3-5 bullet points on key risks, concentration areas, and recommendations.

Summary:""",

    "Location": """You are a healthcare data analyst. Summarize all locations in the network.

Data:
{context}

Structure your response with these sections:

## Location Overview
Bullet points: total location count, overall Birdeye avg rating, total reviews.

## Top Locations by Visit Volume
Markdown table with columns: Location | Visit Count

## Top Locations by Patient Count
Markdown table with columns: Location | Patient Count

## Birdeye Review Performance
Markdown table with columns: Location | Reviews | Avg Rating
Show top 10 locations by review count.

## Notable Insights
3-5 bullet points on best/worst performing locations.

Summary:""",

    "Campaign": """You are a healthcare data analyst. Summarize all campaign performance.

Data:
{context}

Structure your response with these sections:

## Campaign Overview
Bullet points: total campaigns, total calls, avg call time, total abandoned, abandonment rate %.

## Top Campaigns by Call Volume
Markdown table with columns: Campaign | Total Calls | Avg Call Time (s) | Abandoned

## Notable Insights
3-5 bullet points on top performers, underperformers, and recommendations.

Summary:""",

    "Practice": """You are a healthcare data analyst. Summarize all practices.

Data:
{context}

Structure your response with these sections:

## Practice Overview
Bullet point: total practice count.

## Practice Performance
Markdown table with columns: Practice | Patients | Outstanding Balance | Bad Debt
Sort by patient count descending. Include ALL practices from the data.

## Notable Insights
5 bullet points: highest outstanding, highest bad debt, best/worst performers, recommendations.

Summary:""",

    "BirdeyeReview": """You are a healthcare data analyst. Summarize all Birdeye reviews.

Data:
{context}

Structure your response with these sections:

## Review Overview
Bullet points: total reviews, overall avg rating, PHI flagged count, 1-star count, 5-star count.

## Rating Distribution
Markdown table with columns: Rating | Count | Percentage
Show 1-star through 5-star.

## Reviews by Source Platform
Markdown table with columns: Source | Reviews | Avg Rating

## Lowest Rated Locations (Risk Areas)
Markdown table with columns: Location | Reviews | Avg Rating
Show locations with lowest ratings.

## Notable Insights
3-5 bullet points on sentiment trends and risk areas.

Summary:""",

    "Financial": """You are a healthcare data analyst. Provide a complete financial overview of the organization.

Data:
{context}

Structure your response with these sections:

## Financial Overview
Markdown table with columns: Metric | Amount
Rows: Total Charged, Total Paid, Collection Rate, Total Outstanding, Total Bad Debt, Contractual Adjustments, Collection Agency Adjustments.

## Charge Summary
Bullet points: total charges, total charge amount, avg charge amount.

## Transaction Summary
Bullet points: total transactions, total payments, total bad debt adjustments.

## Outstanding Balance by State
Markdown table with columns: State | Outstanding Balance | Bad Debt
Top 10 states.

## Outstanding Balance by Payor Cohort
Markdown table with columns: Cohort | Patients | Outstanding Balance | Bad Debt

## Annual Charge Trend
Markdown table with columns: Year | Charges | Total Amount

## Annual Payment Trend
Markdown table with columns: Year | Transactions | Total Payments

## Key Financial Risks & Insights
5-7 bullet points on risks, concentration areas, trends, and recommendations.

Format all dollar amounts with $ and commas. Format percentages to 1 decimal place.

Summary:""",

    "Charge": """You are a healthcare data analyst. Summarize all charge data.

Data:
{context}

Structure your response with these sections:

## Charge Overview
Bullet points: total charges, total amount, avg charge amount.

## Charges by Procedure Modality
Markdown table with columns: Modality | Charge Count | Total Amount

## Annual Charge Trend
Markdown table with columns: Year | Charge Count | Total Amount

## Top Diagnosis Codes
Markdown table with columns: Diagnosis Code | Charge Count

## Notable Insights
3-5 bullet points on trends and patterns.

Summary:""",

    "Transaction": """You are a healthcare data analyst. Summarize all transaction data.

Data:
{context}

Structure your response with these sections:

## Transaction Overview
Markdown table with columns: Metric | Value
Rows: Total Transactions, Total Payments, Total Adjustments, Total Bad Debt, Avg Payment.

## Annual Payment Trend
Markdown table with columns: Year | Transactions | Total Payments

## Transaction Type Breakdown
Markdown table with columns: Type | Count | Total Payments

## Notable Insights
3-5 bullet points on payment trends, anomalies, and recommendations.

Summary:""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Context formatter
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    """Format numeric value with commas and 2dp, or N/A."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(count, total) -> str:
    """Format count as percentage of total."""
    try:
        return f"{float(count) / float(total) * 100:.1f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"


def _rows_to_text(rows: list[dict], indent: str = "  ") -> str:
    if not rows:
        return f"{indent}(no data)"
    lines = []
    for row in rows:
        parts = []
        for k, v in row.items():
            if isinstance(v, float):
                parts.append(f"{k}: {_fmt(v)}")
            elif isinstance(v, int):
                parts.append(f"{k}: {v:,}")
            else:
                parts.append(f"{k}: {v}")
        lines.append(f"{indent}- " + " | ".join(parts))
    return "\n".join(lines)


def _format_context(label: str, data: dict) -> str:
    lines = []

    if label == "Patient":
        t     = data.get("totals", {})
        f     = data.get("flags", {})
        total = t.get("total_patients", 1)

        lines += [
            "=== PORTFOLIO TOTALS ===",
            f"Total Patients            : {t.get('total_patients', 0):,}",
            f"Total Charged             : ${_fmt(t.get('total_charged'))}",
            f"Total Paid                : ${_fmt(t.get('total_paid'))}",
            f"Total Outstanding Balance : ${_fmt(t.get('total_outstanding'))}",
            f"Total Bad Debt            : ${_fmt(t.get('total_bad_debt'))}",
            f"Avg Outstanding / Patient : ${_fmt(t.get('avg_outstanding'))}",
            "",
            "=== PATIENT FLAGS (these are counts of patients, not booleans) ===",
            f"Self-Pay patients         : {f.get('self_pay_count', 0):,} ({_pct(f.get('self_pay_count',0), total)} of total)",
            f"Insured patients          : {f.get('insured_count', 0):,} ({_pct(f.get('insured_count',0), total)} of total)",
            f"BAI patients              : {f.get('bai_count', 0):,} ({_pct(f.get('bai_count',0), total)} of total)",
            f"Catastrophe patients      : {f.get('catastrophe_count', 0):,} ({_pct(f.get('catastrophe_count',0), total)} of total)",
            f"Friction patients         : {f.get('friction_count', 0):,} ({_pct(f.get('friction_count',0), total)} of total)",
            f"Clean patients            : {f.get('clean_count', 0):,} ({_pct(f.get('clean_count',0), total)} of total)",
            f"Patients with any calls   : {f.get('called_count', 0):,} ({_pct(f.get('called_count',0), total)} of total)",
            f"Bad address patients      : {f.get('bad_address_count', 0):,} ({_pct(f.get('bad_address_count',0), total)} of total)",
            f"Multi-practice patients   : {f.get('multi_practice_count', 0):,} ({_pct(f.get('multi_practice_count',0), total)} of total)",
            "",
            "=== BY STATE (TOP 10) ===",
            _rows_to_text(data.get("by_state", [])),
            "",
            "=== BY PAYOR COHORT ===",
            _rows_to_text(data.get("by_cohort", [])),
            "",
            "=== BY CALL TIER ===",
            _rows_to_text(data.get("by_call_tier", [])),
            "",
            "=== BY GENDER ===",
            _rows_to_text(data.get("by_gender", [])),
        ]

    elif label == "Location":
        t = data.get("totals", {})
        r = data.get("overall_reviews", {})
        lines += [
            f"Total Locations           : {t.get('total_locations', 0):,}",
            "",
            "=== OVERALL BIRDEYE REVIEWS ===",
            f"Total Reviews             : {r.get('total_reviews', 0):,}",
            f"Overall Avg Rating        : {r.get('overall_avg_rating', 'N/A')} / 5.0",
            f"1-Star Reviews            : {r.get('one_star_count', 0):,}",
            f"5-Star Reviews            : {r.get('five_star_count', 0):,}",
            "",
            "=== TOP LOCATIONS BY VISITS ===",
            _rows_to_text(data.get("visit_stats", [])),
            "",
            "=== TOP LOCATIONS BY PATIENTS ===",
            _rows_to_text(data.get("patient_by_location", [])),
            "",
            "=== REVIEW STATS BY LOCATION ===",
            _rows_to_text(data.get("review_stats", [])),
        ]

    elif label == "Campaign":
        t = data.get("totals", {})
        o = data.get("overall_calls", {})
        lines += [
            f"Total Campaigns           : {t.get('total_campaigns', 0):,}",
            "",
            "=== OVERALL CALL STATS ===",
            f"Total Calls               : {o.get('total_calls', 0):,}",
            f"Avg Call Time             : {o.get('avg_call_time', 'N/A')}s",
            f"Total Abandoned           : {o.get('total_abandoned', 0):,}",
            f"Campaigns with Calls      : {o.get('campaigns_with_calls', 0):,}",
            "",
            "=== TOP CAMPAIGNS BY CALLS ===",
            _rows_to_text(data.get("call_stats", [])),
        ]

    elif label == "Practice":
        t = data.get("totals", {})
        lines += [
            f"Total Practices           : {t.get('total_practices', 0):,}",
            "",
            "=== BY PRACTICE ===",
            _rows_to_text(data.get("by_practice", [])),
        ]

    elif label == "BirdeyeReview":
        o = data.get("overall", {})
        lines += [
            "=== OVERALL ===",
            f"Total Reviews             : {o.get('total_reviews', 0):,}",
            f"Avg Rating                : {o.get('avg_rating', 'N/A')} / 5.0",
            f"1-Star                    : {o.get('one_star', 0):,}",
            f"2-Star                    : {o.get('two_star', 0):,}",
            f"3-Star                    : {o.get('three_star', 0):,}",
            f"4-Star                    : {o.get('four_star', 0):,}",
            f"5-Star                    : {o.get('five_star', 0):,}",
            f"PHI Flagged               : {o.get('phi_flagged_count', 0):,}",
            "",
            "=== BY SOURCE PLATFORM ===",
            _rows_to_text(data.get("by_source", [])),
            "",
            "=== LOWEST RATED LOCATIONS ===",
            _rows_to_text(data.get("by_location", [])),
        ]

    elif label == "Financial":
        f  = data.get("patient_fin", {})
        c  = data.get("charge_summary", {})
        tx = data.get("transaction_summary", {})

        # Compute collection rate
        charged = f.get("total_charged") or 0
        paid    = f.get("total_paid") or 0
        coll_rate = _pct(paid, charged) if charged else "N/A"

        lines += [
            "=== PATIENT FINANCIALS ===",
            f"Total Charged             : ${_fmt(f.get('total_charged'))}",
            f"Total Paid                : ${_fmt(f.get('total_paid'))}",
            f"Collection Rate           : {coll_rate}",
            f"Total Outstanding Balance : ${_fmt(f.get('total_outstanding'))}",
            f"Total Bad Debt            : ${_fmt(f.get('total_bad_debt'))}",
            f"Contractual Adjustments   : ${_fmt(f.get('total_contractual'))}",
            f"Collection Agency Adj     : ${_fmt(f.get('total_collection_agency'))}",
            "",
            "=== CHARGE SUMMARY ===",
            f"Total Charges             : {c.get('total_charges', 0):,}",
            f"Total Charge Amount       : ${_fmt(c.get('total_charge_amount'))}",
            f"Avg Charge Amount         : ${_fmt(c.get('avg_charge_amount'))}",
            "",
            "=== TRANSACTION SUMMARY ===",
            f"Total Transactions        : {tx.get('total_transactions', 0):,}",
            f"Total Payments            : ${_fmt(tx.get('total_payments'))}",
            f"Total Bad Debt Adj        : ${_fmt(tx.get('total_bad_debt_adj'))}",
            "",
            "=== OUTSTANDING BALANCE BY STATE (TOP 10) ===",
            _rows_to_text(data.get("by_state", [])),
            "",
            "=== OUTSTANDING BALANCE BY PAYOR COHORT ===",
            _rows_to_text(data.get("by_cohort", [])),
            "",
            "=== CHARGES BY YEAR ===",
            # FIX: use dedicated charges_by_year key, not nested dict
            _rows_to_text(data.get("charges_by_year", [])),
            "",
            "=== PAYMENTS BY YEAR ===",
            # FIX: use dedicated payments_by_year key, not nested dict
            _rows_to_text(data.get("payments_by_year", [])),
        ]

    elif label == "Charge":
        o = data.get("overall", {})
        lines += [
            "=== OVERALL ===",
            f"Total Charges             : {o.get('total_charges', 0):,}",
            f"Total Amount              : ${_fmt(o.get('total_amount'))}",
            f"Avg Amount                : ${_fmt(o.get('avg_amount'))}",
            "",
            "=== BY MODALITY ===",
            _rows_to_text(data.get("by_modality", [])),
            "",
            "=== BY YEAR ===",
            _rows_to_text(data.get("by_year", [])),
            "",
            "=== TOP DIAGNOSES ===",
            _rows_to_text(data.get("top_diagnoses", [])),
        ]

    elif label == "Transaction":
        o = data.get("overall", {})
        lines += [
            "=== OVERALL ===",
            f"Total Transactions        : {o.get('total_transactions', 0):,}",
            f"Total Payments            : ${_fmt(o.get('total_payments'))}",
            f"Total Adjustments         : ${_fmt(o.get('total_adjustments'))}",
            f"Total Bad Debt            : ${_fmt(o.get('total_bad_debt'))}",
            f"Avg Payment               : ${_fmt(o.get('avg_payment'))}",
            "",
            "=== BY YEAR ===",
            _rows_to_text(data.get("by_year", [])),
            "",
            "=== BY TYPE ===",
            _rows_to_text(data.get("by_type", [])),
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main aggregate summarizer
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_all_nodes(label: str) -> AsyncGenerator[dict, None]:
    """
    Stream an aggregate summary of all nodes of a given label.
    Yields same chunk format as stream_qa_response.
    """
    logger.info(f"summarize_all_nodes: label={label}")

    fetcher = _AGG_FETCHERS.get(label)
    if not fetcher:
        yield {"type": "error", "data": f"No aggregate summarizer for label: {label}"}
        return

    # ── 1. Fetch all stats from Neo4j ─────────────────────────────────────
    try:
        data = fetcher()
    except Exception as e:
        logger.error(f"summarize_all_nodes fetch failed: {e}", exc_info=True)
        yield {"type": "error", "data": f"Failed to fetch {label} data: {e}"}
        return

    # ── 2. Emit structured data for chart rendering in openai_compat.py ──
    # This chunk is consumed by generate_stream() to build Recharts artifacts.
    # It must be yielded BEFORE tokens so the SSE consumer can buffer it.
    yield {"type": "summary_data", "label": label, "data": data}

    # ── 3. Format context as readable text ────────────────────────────────
    context_text = _format_context(label, data)
    logger.debug(f"aggregate context ({len(context_text)} chars)")

    # ── 4. Build prompt ───────────────────────────────────────────────────
    prompt_template = _PROMPTS.get(label, "Summarize this data:\n{context}\n\nSummary:")
    prompt = prompt_template.format(context=context_text)

    # ── 5. Stream through QA LLM ──────────────────────────────────────────
    try:
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
        logger.error(f"summarize_all_nodes LLM error: {e}", exc_info=True)
        yield {"type": "error", "data": str(e)}