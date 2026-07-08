"""
cypher_autofix.py
─────────────────
Auto-corrects common Cypher mistakes produced by the text2cypher model
BEFORE the guardrail sees the query.

Apply in qa/chain.py immediately after the model generates Cypher:

    from cypher_autofix import autofix_cypher
    cypher = autofix_cypher(cypher, logger)

Fixes applied (in order):
  1. SQL date functions → dot notation:  year(x) → x.year
  2. substring() on DateTime → dot notation: substring(t.post_date, 0, 4) → t.post_date.year
  3. GROUP BY removed (Cypher groups implicitly)
  4. Semicolons removed (multi-statement guard)
  5. greater_than(N) → > N  (hallucinated function)
  6. date_trunc() → explicit year+quarter expression
  7. ORDER BY on un-aggregated variable after WITH aggregation
  8. Hallucinated source_db filter values like 'your_source_db'
  9. IS NOT NULL inside node property map → moved to WHERE clause
     {state: IS NOT NULL} → WHERE n.state IS NOT NULL
 10. Unaliased dotted property in WITH → add AS alias
     WITH l.state, avg(...) → WITH l.state AS state, avg(...)
 11. Comparison operator inside node property map → moved to WHERE clause
     {outstanding_balance > 1000} → WHERE p.outstanding_balance > 1000
 12. Patient-[:REGISTERED_AT]->Location wrong hop → rewritten to correct path
     (p:Patient)-[:REGISTERED_AT]->(l:Location)
     → (p:Patient)-[:HAD_VISIT]->(v:Visit)-[:PERFORMED_AT]->(l:Location)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Native DateTime properties — dot notation is correct for these ────────────
# substring() and datetime() wrapping are WRONG on these fields
_NATIVE_DATETIME_PROPS = [
    "post_date", "service_date", "admit_date", "discharge_date",
    "dob", "call_datetime", "created_date", "released_date",
]

# ── STRING date properties — substring() is correct, dot notation is WRONG ───
_STRING_DATE_PROPS = [
    "date_posted", "start_date",
]


def _log(original: str, fixed: str, rule: str) -> None:
    if original != fixed:
        logger.warning(
            f"autofix [{rule}]:\n  BEFORE: {original.strip()}\n  AFTER:  {fixed.strip()}"
        )


# ── Fix 1: SQL date functions → dot notation ──────────────────────────────────
# year(t.post_date) → t.post_date.year
# month(c.service_date) → c.service_date.month
_SQL_DATE_FN_RE = re.compile(
    r'\b(year|month|day)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
    re.IGNORECASE,
)

def _fix_sql_date_functions(cypher: str) -> str:
    def replacer(m):
        fn   = m.group(1).lower()
        prop = m.group(2).strip()
        return f"{prop}.{fn}"
    return _SQL_DATE_FN_RE.sub(replacer, cypher)


# ── Fix 2: quarter(x) → explicit computation ──────────────────────────────────
# quarter(c.service_date) → (c.service_date.month - 1) / 3 + 1
_QUARTER_FN_RE = re.compile(
    r'\bquarter\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\)',
    re.IGNORECASE,
)

def _fix_quarter_function(cypher: str) -> str:
    def replacer(m):
        prop = m.group(1).strip()
        return f"({prop}.month - 1) / 3 + 1"
    return _QUARTER_FN_RE.sub(replacer, cypher)


# ── Fix 3: date_trunc() → year + quarter ──────────────────────────────────────
# date_trunc('quarter', c.service_date) → c.service_date.year, (c.service_date.month-1)/3+1
_DATE_TRUNC_RE = re.compile(
    r"date_trunc\s*\(\s*'([^']+)'\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\s*\)",
    re.IGNORECASE,
)

def _fix_date_trunc(cypher: str) -> str:
    def replacer(m):
        granularity = m.group(1).lower()
        prop = m.group(2).strip()
        if granularity == "quarter":
            return f"{prop}.year, ({prop}.month - 1) / 3 + 1"
        elif granularity == "month":
            return f"{prop}.year, {prop}.month"
        elif granularity == "year":
            return f"{prop}.year"
        else:
            return f"{prop}.year"
    return _DATE_TRUNC_RE.sub(replacer, cypher)


# ── Fix 4: substring() on native DateTime → dot notation ──────────────────────
# substring(t.post_date, 0, 4) → t.post_date.year
# substring(t.post_date, 0, 7) → toString(t.post_date.year) + '-' + ...
#   (simplified: just use .year and .month — chart handles separate columns)
_SUBSTRING_DATETIME_RE = re.compile(
    r'substring\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\.('
    + '|'.join(_NATIVE_DATETIME_PROPS)
    + r')\s*,\s*0\s*,\s*(\d+)\s*\)',
    re.IGNORECASE,
)

def _fix_substring_on_datetime(cypher: str) -> str:
    def replacer(m):
        node = m.group(1)
        prop = m.group(2)
        length = int(m.group(3))
        full = f"{node}.{prop}"
        if length <= 4:
            return f"{full}.year"
        elif length <= 7:
            # Return year-month as separate columns isn't easy inline;
            # best approximation: toString() so Neo4j formats it
            return f"toString({full}.year) + '-' + right('0' + toString({full}.month), 2)"
        else:
            return f"{full}.year"
    return _SUBSTRING_DATETIME_RE.sub(replacer, cypher)


# ── Fix 5: GROUP BY → remove (Cypher groups implicitly) ──────────────────────
_GROUP_BY_RE = re.compile(r'\bGROUP\s+BY\b[^\n]*', re.IGNORECASE)

def _fix_group_by(cypher: str) -> str:
    return _GROUP_BY_RE.sub('', cypher)


# ── Fix 6: Trailing semicolons ─────────────────────────────────────────────────
def _fix_semicolons(cypher: str) -> str:
    return cypher.rstrip().rstrip(';').rstrip()


# ── Fix 7: greater_than(N) → > N ─────────────────────────────────────────────
_GREATER_THAN_RE = re.compile(r':\s*greater_than\s*\(\s*(\d+(?:\.\d+)?)\s*\)', re.IGNORECASE)

def _fix_greater_than(cypher: str) -> str:
    # {outstanding_balance: greater_than(10000)} → WHERE clause can't be auto-fixed inline
    # but we can at least neutralise the crash by converting to a comment note
    # Real fix: model should use WHERE p.outstanding_balance > N
    # We do a best-effort replacement in simple cases
    return _GREATER_THAN_RE.sub(r' > \1', cypher)


# ── Fix 8: Hallucinated source_db values ──────────────────────────────────────
# {source_db: 'your_source_db'} or WHERE x.source_db = 'your_source_db'
_FAKE_SOURCE_DB_RE = re.compile(
    r"\{[^}]*source_db\s*:\s*'your_source_db'[^}]*\}|"
    r"AND\s+\w+\.source_db\s*=\s*'your_source_db'|"
    r"WHERE\s+\w+\.source_db\s*=\s*'your_source_db'\s*AND\s*|"
    r"WHERE\s+\w+\.source_db\s*=\s*'your_source_db'",
    re.IGNORECASE,
)

def _fix_fake_source_db(cypher: str) -> str:
    # Remove the node property filter entirely
    cypher = re.sub(
        r'\s*\{\s*source_db\s*:\s*\'your_source_db\'\s*\}',
        '',
        cypher,
        flags=re.IGNORECASE,
    )
    # Remove WHERE / AND clause variants
    cypher = re.sub(
        r'\s+AND\s+\w+\.source_db\s*=\s*\'your_source_db\'',
        '',
        cypher,
        flags=re.IGNORECASE,
    )
    cypher = re.sub(
        r'WHERE\s+\w+\.source_db\s*=\s*\'your_source_db\'\s+AND\s+',
        'WHERE ',
        cypher,
        flags=re.IGNORECASE,
    )
    cypher = re.sub(
        r'WHERE\s+\w+\.source_db\s*=\s*\'your_source_db\'',
        '',
        cypher,
        flags=re.IGNORECASE,
    )
    return cypher


# ── Fix 9b (was Fix 7): ORDER BY on pre-aggregation variable after WITH ──────
_WITH_ORDERBY_LOST_VAR_RE = re.compile(
    r'(WITH\s+.+?)\s+ORDER\s+BY\s+count\((\w+)\)',
    re.IGNORECASE | re.DOTALL,
)

def _fix_orderby_lost_var(cypher: str) -> str:
    """
    Detects ORDER BY count(var) where var was consumed by a prior WITH aggregation.
    Injects count(var) AS _cnt into the WITH clause and rewrites ORDER BY.
    WITH cam.name AS campaign_name, avg(...) AS avg_bal
    ORDER BY count(rc) DESC
    → WITH cam.name AS campaign_name, avg(...) AS avg_bal, count(rc) AS _rc_count
    ORDER BY _rc_count DESC
    """
    m = _WITH_ORDERBY_LOST_VAR_RE.search(cypher)
    if not m:
        return cypher
    with_clause = m.group(1)
    var = m.group(2)
    fixed_with = with_clause.rstrip(', ') + f', count({var}) AS _{var}_count'
    cypher = cypher.replace(m.group(0), fixed_with + f'\nORDER BY _{var}_count')
    return cypher


# ── Fix 9: IS NOT NULL inside node property map ──────────────────────────────
# MATCH (l:Location {state: IS NOT NULL})
# → MATCH (l:Location) WHERE l.state IS NOT NULL
_INLINE_IS_NOT_NULL_NODE_RE = re.compile(
    r'\((\w+):(\w+)\s*\{([^}]+)\}\)',
    re.IGNORECASE,
)

def _fix_inline_is_not_null(cypher: str) -> str:
    """
    Moves IS NOT NULL / IS NULL checks from inline node property maps to WHERE clause.

    MATCH (l:Location {state: IS NOT NULL})
    → MATCH (l:Location) WHERE l.state IS NOT NULL

    MATCH (p:Patient {state: IS NOT NULL, gender: IS NOT NULL})
    → MATCH (p:Patient) WHERE p.state IS NOT NULL AND p.gender IS NOT NULL

    Leaves valid inline filters like {is_self_pay: true} untouched.
    """
    null_checks_collected = []

    def replacer(m):
        var, label, props_str = m.group(1), m.group(2), m.group(3)
        if not re.search(r'IS\s+(NOT\s+)?NULL', props_str, re.IGNORECASE):
            return m.group(0)  # no IS NULL/IS NOT NULL — leave unchanged
        null_checks, remaining = [], []
        for part in re.split(r',\s*', props_str):
            part = part.strip()
            if not part:
                continue
            if re.match(r'(\w+)\s*:\s*IS\s+NOT\s+NULL', part, re.IGNORECASE):
                key = re.match(r'(\w+)', part).group(1)
                null_checks.append(f"{var}.{key} IS NOT NULL")
            elif re.match(r'(\w+)\s*:\s*IS\s+NULL', part, re.IGNORECASE):
                key = re.match(r'(\w+)', part).group(1)
                null_checks.append(f"{var}.{key} IS NULL")
            else:
                remaining.append(part)
        null_checks_collected.extend(null_checks)
        if remaining:
            return f"({var}:{label} {{{', '.join(remaining)}}})"
        return f"({var}:{label})"

    fixed = _INLINE_IS_NOT_NULL_NODE_RE.sub(replacer, cypher)

    if not null_checks_collected:
        return cypher

    all_conds = " AND ".join(null_checks_collected)
    if re.search(r'\bWHERE\b', fixed, re.IGNORECASE):
        fixed = re.sub(r'\bWHERE\b\s*', f'WHERE {all_conds} AND ', fixed, count=1, flags=re.IGNORECASE)
    else:
        fixed = re.sub(r'(MATCH\s[^\n]+)', r'\1\nWHERE ' + all_conds, fixed, count=1, flags=re.IGNORECASE)
    return fixed


# ── Fix 10: unaliased dotted property in WITH ─────────────────────────────────
# WITH l.state, avg(b.rating) AS avg_rating
# → WITH l.state AS state, avg(b.rating) AS avg_rating
_WITH_BODY_RE = re.compile(
    r'\bWITH\b(.+?)(?=\bRETURN\b|\bWHERE\b|\bORDER\b|\bMATCH\b|$)',
    re.IGNORECASE | re.DOTALL,
)

def _fix_unaliased_with(cypher: str) -> str:
    """
    Finds bare dotted properties in WITH clauses that lack an AS alias and adds one.
    Only fixes simple node.property tokens — leaves function calls untouched.

    WITH l.state, avg(b.rating) AS avg_rating
    → WITH l.state AS state, avg(b.rating) AS avg_rating
    """
    def fix_body(m):
        body  = m.group(1)
        items = re.split(r',(?![^(]*\))', body)
        fixed = []
        for item in items:
            s = item.strip()
            if re.match(r'^\w+\.\w+$', s) and ' AS ' not in item.upper():
                prop_name = s.split('.')[1]
                fixed.append(f" {s} AS {prop_name}")
            else:
                fixed.append(item)
        return "WITH" + ",".join(fixed)

    return _WITH_BODY_RE.sub(fix_body, cypher)


# ── Fix 11: comparison operator inside node property map ─────────────────────
# MATCH (p:Patient {outstanding_balance > 1000})
# → MATCH (p:Patient) WHERE p.outstanding_balance > 1000
#
# Also fixes: {outstanding_balance > 1000, is_self_pay: true}
# → (p:Patient {is_self_pay: true}) WHERE p.outstanding_balance > 1000
_INLINE_COMPARISON_RE = re.compile(
    r'\((\w+):(\w+)\s*\{([^}]+)\}\)',
    re.IGNORECASE,
)
_COMPARISON_PROP_RE = re.compile(
    r'(\w+)\s*(>=|<=|!=|<>|>|<)\s*([^\s,}]+)',
    re.IGNORECASE,
)

def _fix_inline_comparison(cypher: str) -> str:
    """
    Moves comparison expressions from inline node maps to WHERE clause.

    MATCH (p:Patient {outstanding_balance > 1000})
    → MATCH (p:Patient) WHERE p.outstanding_balance > 1000

    MATCH (p:Patient {outstanding_balance > 1000, is_self_pay: true})
    → MATCH (p:Patient {is_self_pay: true}) WHERE p.outstanding_balance > 1000

    Leaves valid key:value filters like {is_self_pay: true} untouched.
    """
    comparisons_collected = []

    def replacer(m):
        var, label, props_str = m.group(1), m.group(2), m.group(3)
        if not _COMPARISON_PROP_RE.search(props_str):
            return m.group(0)  # no comparison operators — leave unchanged

        remaining = []
        for part in re.split(r',\s*', props_str):
            part = part.strip()
            if not part:
                continue
            cm = _COMPARISON_PROP_RE.match(part)
            if cm:
                key, op, val = cm.group(1), cm.group(2), cm.group(3)
                comparisons_collected.append(f"{var}.{key} {op} {val}")
            else:
                remaining.append(part)

        if remaining:
            return f"({var}:{label} {{{', '.join(remaining)}}})"
        return f"({var}:{label})"

    fixed = _INLINE_COMPARISON_RE.sub(replacer, cypher)

    if not comparisons_collected:
        return cypher

    all_conds = " AND ".join(comparisons_collected)
    if re.search(r'\bWHERE\b', fixed, re.IGNORECASE):
        fixed = re.sub(
            r'\bWHERE\b\s*',
            f'WHERE {all_conds} AND ',
            fixed, count=1, flags=re.IGNORECASE,
        )
    else:
        fixed = re.sub(
            r'(MATCH\s[^\n]+)',
            r'\1\nWHERE ' + all_conds,
            fixed, count=1, flags=re.IGNORECASE,
        )
    return fixed


# ── Fix 12: Patient-[:REGISTERED_AT]->Location wrong hop ─────────────────────
# Patient registers at Practice, NEVER at Location.
# Correct path: Patient-[:HAD_VISIT]->Visit-[:PERFORMED_AT]->Location
_WRONG_REGISTERED_AT_RE = re.compile(
    r'\((\w+):Patient\)-\[:REGISTERED_AT\]->\((\w+):Location\)',
    re.IGNORECASE,
)

def _fix_wrong_registered_at(cypher: str) -> str:
    """
    (p:Patient)-[:REGISTERED_AT]->(l:Location) is always wrong.
    Rewrites to correct 2-hop path through Visit.
    """
    def replacer(m):
        p = m.group(1)
        l = m.group(2)
        return (
            f"({p}:Patient)-[:HAD_VISIT]->(v_gen:Visit)"
            f"-[:PERFORMED_AT]->({l}:Location)"
        )
    return _WRONG_REGISTERED_AT_RE.sub(replacer, cypher)


# ── Master autofix ─────────────────────────────────────────────────────────────

def autofix_cypher(cypher: str, log: logging.Logger | None = None) -> str:
    """
    Apply all fixes in sequence. Returns corrected Cypher.
    Logs a warning for every rule that fires.
    """
    _logger = log or logger
    original = cypher

    fixes = [
        ("sql_date_functions",      _fix_sql_date_functions),
        ("quarter_function",        _fix_quarter_function),
        ("date_trunc",              _fix_date_trunc),
        ("substring_on_datetime",   _fix_substring_on_datetime),
        ("group_by",                _fix_group_by),
        ("semicolons",              _fix_semicolons),
        ("greater_than",            _fix_greater_than),
        ("fake_source_db",          _fix_fake_source_db),
        ("orderby_lost_var",        _fix_orderby_lost_var),
        ("inline_is_not_null",      _fix_inline_is_not_null),
        ("unaliased_with",          _fix_unaliased_with),
        ("inline_comparison",       _fix_inline_comparison),
        ("wrong_registered_at",     _fix_wrong_registered_at),
    ]

    for rule_name, fix_fn in fixes:
        before = cypher
        cypher = fix_fn(cypher)
        if cypher != before:
            _logger.warning(
                f"autofix [{rule_name}]:\n"
                f"  BEFORE: {before.strip()}\n"
                f"  AFTER:  {cypher.strip()}"
            )

    if cypher != original:
        _logger.info(f"autofix: {sum(1 for _, f in fixes if f(original) != original)} rule(s) fired")

    return cypher.strip()