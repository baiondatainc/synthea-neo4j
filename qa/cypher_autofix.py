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


# ── Fix 9: ORDER BY on pre-aggregation variable after WITH ────────────────────
# WITH cam.name AS campaign_name, avg(...) AS avg_bal
# ORDER BY count(rc) DESC   ← rc no longer in scope
# → Move count into WITH, alias it, ORDER BY the alias
_WITH_ORDERBY_LOST_VAR_RE = re.compile(
    r'(WITH\s+.+?)\s+ORDER\s+BY\s+count\((\w+)\)',
    re.IGNORECASE | re.DOTALL,
)

def _fix_orderby_lost_var(cypher: str) -> str:
    """
    Detects ORDER BY count(var) where var was consumed by a prior WITH aggregation.
    Injects count(var) AS _cnt into the WITH clause and rewrites ORDER BY.
    """
    m = _WITH_ORDERBY_LOST_VAR_RE.search(cypher)
    if not m:
        return cypher
    with_clause = m.group(1)
    var = m.group(2)
    # Add count alias to WITH
    fixed_with = with_clause.rstrip(', ') + f', count({var}) AS _{var}_count'
    cypher = cypher.replace(m.group(0), fixed_with + f'\nORDER BY _{var}_count')
    return cypher


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