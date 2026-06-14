"""
Output guardrail — PHI/PII redaction.

Two surfaces:
  - free-text answers (regex-based redaction of names/phones/emails/SSNs/DOBs)
  - structured result rows (column-aware redaction using the data dictionary
    PII property list)

Per the v2 design "always mask" decision, redaction does not depend on
caller identity. Tighten/loosen by editing `MASK_*` constants and the
`pii` lists in `metadata/data_dictionary.yaml`.
"""
from __future__ import annotations

import re

from metadata.catalog import get_catalog

# ── Regex masks ──────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
DOB_RE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

MASK_EMAIL = "[email-redacted]"
MASK_PHONE = "[phone-redacted]"
MASK_SSN = "[ssn-redacted]"
MASK_DOB = "[dob-redacted]"
MASK_NAME = "[name-redacted]"


def redact_text(text: str) -> str:
    if not text:
        return text
    text = EMAIL_RE.sub(MASK_EMAIL, text)
    text = PHONE_RE.sub(MASK_PHONE, text)
    text = SSN_RE.sub(MASK_SSN, text)
    text = DOB_RE.sub(MASK_DOB, text)
    return text


def _mask_for_property(prop: str) -> str:
    p = prop.lower()
    if "email" in p:
        return MASK_EMAIL
    if "phone" in p or "cell" in p or "fax" in p:
        return MASK_PHONE
    if "dob" in p or "birth" in p:
        return MASK_DOB
    if "name" in p:
        return MASK_NAME
    return "[redacted]"


def redact_rows(rows: list[dict]) -> list[dict]:
    """Walk result rows; mask any column whose key matches a PII property in
    any label's `pii` list. Numeric/null values pass through unchanged.
    """
    if not rows:
        return rows
    catalog = get_catalog()
    pii_props = catalog.all_pii_properties()
    if not pii_props:
        return rows

    # Lowercase lookup so result keys like `phone_norm` match dictionary keys.
    pii_lc = {p.lower() for p in pii_props}

    out: list[dict] = []
    for r in rows:
        masked = {}
        for k, v in r.items():
            base = k.lower()
            # Strip common suffixes like _norm, _str so dictionary still matches.
            base_stripped = re.sub(r"_(norm|str|raw)$", "", base)
            if base in pii_lc or base_stripped in pii_lc:
                masked[k] = None if v is None else _mask_for_property(k)
            elif isinstance(v, str):
                masked[k] = redact_text(v)
            else:
                masked[k] = v
        out.append(masked)
    return out
