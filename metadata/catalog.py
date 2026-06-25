"""
Metadata catalog — loads the data dictionary and exposes lookups for:
  - friendly schema text (injected into the text2cypher prompt)
  - value translation (swap coded values for human-readable labels in answers)
  - PII property list (used by guardrails/output.py for redaction)
  - allowed-topic keywords (used by guardrails/input.py)

TOKEN BUDGET
------------
Target local LLM context: 4096 tokens
Reserved for question + Cypher output: ~800 tokens
Reserved for system prompt boilerplate: ~400 tokens
Available for schema injection: ~900 tokens   ← schema_for_question() targets this

Full schema_addendum() is kept for reference / offline use only.
At runtime always call schema_for_question(question) instead.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DICT_PATH = Path(__file__).parent / "data_dictionary.yaml"

# Maximum tokens to budget for the schema block injected into prompts.
# Rough heuristic: 1 token ≈ 4 chars of English text.
_SCHEMA_CHAR_BUDGET = 900 * 4   # 900 tokens × 4 chars = 3600 chars

# Labels whose keyword triggers imply they should be included
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "Patient":       ["patient", "patients", "cohort", "balance", "payor", "payer",
                      "self-pay", "self pay", "sapa", "bai", "propensity", "bad debt"],
    "Visit":         ["visit", "encounter", "admission", "discharge", "modality",
                      "mri", "ct", "ultrasound", "imaging", "radiology"],
    "Charge":        ["charge", "charges", "cpt", "hcpcs", "procedure", "billed",
                      "line", "outstanding", "balance", "write-off", "void"],
    "Transaction":   ["payment", "transaction", "adjustment", "denial", "bad debt",
                      "collection", "ers", "era", "paid", "refund", "co45", "carc"],
    "Statement":     ["statement", "bill", "notice", "level", "delivery"],
    "InsurancePlan": ["insurance", "plan", "carrier", "payer", "coverage",
                      "primary", "secondary", "hmo", "ppo", "medicare", "medicaid"],
    "Location":      ["location", "site", "facility", "center", "where", "address"],
    "Practice":      ["practice", "source", "radm", "radh", "tri", "nraa",
                      "cirpa", "iai", "rade", "acr", "radz"],
    "RCCall":        ["call", "calls", "ringcentral", "agent", "handle", "queue",
                      "skill", "connected", "voicemail", "outcome"],
    "IVRInbound":    ["ivr", "inbound", "self-service", "pay by phone", "phone payment"],
    "DiallerCall":   ["dialler", "dialer", "outbound", "predictive"],
    "Campaign":      ["campaign", "outreach", "sapa", "mrb", "atlanta 404"],
    "BirdeyeReview": ["review", "rating", "birdeye", "stars", "sentiment", "satisfaction"],
    "DiagnosisCode": ["diagnosis", "icd", "icd-10", "condition", "disease"],
    "ProcedureCode": ["procedure", "cpt", "hcpcs", "code", "rvu", "modality"],
}

# Relationships indexed by the labels they connect — used to filter
# only relevant relationship examples into the prompt.
_REL_LABELS: dict[str, tuple[str, str]] = {
    "REGISTERED_AT":           ("Patient",      "Practice"),
    "BELONGS_TO_PRACTICE":     ("Location",     "Practice"),
    "ISSUED_BY_PRACTICE":      ("InsurancePlan","Practice"),
    "HAD_VISIT":               ("Patient",      "Visit"),
    "PERFORMED_AT":            ("Visit",        "Location"),
    "UNDER_PLAN":              ("Visit",        "InsurancePlan"),
    "HAS_CHARGE":              ("Patient",      "Charge"),
    "PART_OF_VISIT":           ("Charge",       "Visit"),
    "AT_LOCATION":             ("Charge",       "Location"),
    "DIAGNOSED_WITH":          ("Charge",       "DiagnosisCode"),
    "USES_PROCEDURE":          ("Charge",       "ProcedureCode"),
    "SETTLES":                 ("Transaction",  "Charge"),
    "HAS_TRANSACTION":         ("Patient",      "Transaction"),
    "RECEIVED_STATEMENT":      ("Patient",      "Statement"),
    "PART_OF_CAMPAIGN":        ("RCCall",       "Campaign"),
    "ATTRIBUTED_TO_PHONE":     ("RCCall",       "PhoneBridge"),
    "IDENTIFIED_BY_PHONE":     ("Patient",      "PhoneBridge"),
    "CALLED_IVR":              ("Patient",      "IVRInbound"),
    "CONTACTED_BY_DIALLER":    ("Patient",      "DiallerCall"),
    "REVIEWS":                 ("BirdeyeReview","Location"),
    "RUN_BY":                  ("Campaign",     "Practice"),
    "SAME_PERSON_AS":          ("Patient",      "Patient"),
    "DIALLER_PART_OF_CAMPAIGN":("DiallerCall",  "Campaign"),
}

# Common multi-hop path keywords — only inject paths that match the question
_PATH_KEYWORDS: dict[str, list[str]] = {
    "patient_to_carrier":    ["carrier", "insurance", "plan", "payer", "coverage"],
    "patient_to_location":   ["location", "site", "facility", "where", "center"],
    "patient_to_diagnosis":  ["diagnosis", "icd", "condition", "disease"],
    "patient_to_procedure":  ["procedure", "cpt", "modality", "mri", "ct"],
    "patient_to_agent":      ["agent", "call", "ringcentral", "who called"],
    "charge_to_payment":     ["payment", "paid", "collected", "cash"],
    "charge_to_denial":      ["denial", "denied", "rejected", "co45"],
    "cross_practice_patient":["multi-practice", "multiple practices", "same patient"],
    "location_to_reviews":   ["review", "rating", "birdeye", "stars"],
    "patient_full_balance":  ["balance", "outstanding", "owe", "total balance"],
}


class Catalog:
    def __init__(self, doc: dict[str, Any]):
        self._doc = doc
        self._labels: dict[str, dict] = doc.get("labels", {})
        self._rels: dict[str, dict]   = doc.get("relationships", {})
        self._topics: list[str]       = [t.lower() for t in doc.get("allowed_topics", [])]
        self._paths: dict[str, str]   = doc.get("common_paths", {})

        self._pii_by_label: dict[str, set[str]] = {
            label: set(meta.get("pii", []))
            for label, meta in self._labels.items()
        }

        # value_map[label][property][code] -> friendly label
        self._value_map: dict[str, dict[str, dict[str, str]]] = {}
        for label, meta in self._labels.items():
            props = meta.get("properties", {}) or {}
            self._value_map[label] = {}
            for prop_name, prop_meta in props.items():
                vals = (prop_meta or {}).get("values") or {}
                if vals:
                    self._value_map[label][prop_name] = {
                        str(k).lower(): v for k, v in vals.items()
                    }

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def labels(self) -> dict[str, dict]:
        return self._labels

    @property
    def relationships(self) -> dict[str, dict]:
        return self._rels

    @property
    def allowed_topics(self) -> list[str]:
        return self._topics

    # ── PII helpers ──────────────────────────────────────────────────────────

    def pii_properties(self, label: str) -> set[str]:
        return self._pii_by_label.get(label, set())

    def all_pii_properties(self) -> set[str]:
        out: set[str] = set()
        for props in self._pii_by_label.values():
            out.update(props)
        return out

    # ── Value translation ────────────────────────────────────────────────────

    def translate_value(self, label: str, prop: str, code: Any) -> Any:
        if code is None:
            return code
        key = str(code).lower()
        return self._value_map.get(label, {}).get(prop, {}).get(key, code)

    def humanize_row(self, row: dict, label_hint: str | None = None) -> dict:
        """Translate coded values in a result row to human-readable labels."""
        out = {}
        for k, v in row.items():
            if isinstance(v, (int, float)) or v is None:
                out[k] = v
                continue
            translated = v
            for label in self._labels:
                if label_hint and label != label_hint:
                    continue
                t = self.translate_value(label, k, v)
                if t != v:
                    translated = t
                    break
            out[k] = translated
        return out

    def humanize_rows(
        self, rows: list[dict], label_hint: str | None = None
    ) -> list[dict]:
        return [self.humanize_row(r, label_hint) for r in rows]

    # ── Token-aware schema injection (USE THIS AT RUNTIME) ───────────────────

    def schema_for_question(self, question: str) -> str:
        """
        Return a compact schema block relevant to this question.

        Targets ~900 tokens (~3600 chars) to fit inside a 4096-token context
        alongside the system prompt, question, and generated Cypher.

        Algorithm:
          1. Detect relevant node labels from question keywords.
          2. Emit one-line node + key property descriptions for those labels.
          3. Emit only the relationship examples that touch relevant labels.
          4. Emit only common_paths that match question keywords.
          5. Stop adding content once char budget is reached.
        """
        q = question.lower()
        budget = _SCHEMA_CHAR_BUDGET
        parts: list[str] = []

        # ── Step 1: detect relevant labels ───────────────────────────────────
        relevant: list[str] = []
        for label, keywords in _LABEL_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                relevant.append(label)

        # Patient is the hub — always include it
        if "Patient" not in relevant:
            relevant.insert(0, "Patient")

        # ── Step 2: compact node + property block ─────────────────────────────
        node_lines: list[str] = ["NODES:"]
        for label in relevant:
            meta = self._labels.get(label, {})
            desc = meta.get("description", "")
            # Truncate long descriptions to one sentence
            desc = desc.split(".")[0] if desc else ""
            node_lines.append(f"  ({label}) — {desc}")
            props = meta.get("properties", {}) or {}
            for prop_name, prop_meta in props.items():
                if not prop_meta:
                    continue
                pdesc = prop_meta.get("description", "")
                vals  = prop_meta.get("values") or {}
                # Only include properties with descriptions or value maps
                if not pdesc and not vals:
                    continue
                # Compact: skip verbose descriptions, just list value codes
                if vals:
                    val_str = ", ".join(
                        f"{k}={v}" for k, v in list(vals.items())[:4]
                    )
                    node_lines.append(f"    .{prop_name}: {val_str}")
                else:
                    # Trim description to 60 chars
                    short = pdesc[:60].rstrip()
                    node_lines.append(f"    .{prop_name}: {short}")

        node_block = "\n".join(node_lines)
        if len(node_block) <= budget:
            parts.append(node_block)
            budget -= len(node_block)
        else:
            # Over budget — emit label names only, no properties
            parts.append("NODES: " + ", ".join(relevant))
            budget -= len(parts[-1])

        # ── Step 3: relevant relationship examples ────────────────────────────
        rel_lines: list[str] = ["RELATIONSHIPS:"]
        relevant_set = set(relevant)
        for rel, (frm, to) in _REL_LABELS.items():
            if frm in relevant_set or to in relevant_set:
                rel_meta = self._rels.get(rel, {})
                example  = rel_meta.get("example", f"(:{frm})-[:{rel}]->(:{to})")
                rel_lines.append(f"  {example}")

        rel_block = "\n".join(rel_lines)
        if budget > 200 and rel_lines:
            # Truncate if needed
            if len(rel_block) <= budget:
                parts.append(rel_block)
                budget -= len(rel_block)
            else:
                # Fit as many lines as the budget allows
                fitted: list[str] = [rel_lines[0]]  # header
                for line in rel_lines[1:]:
                    if budget - len("\n".join(fitted)) - len(line) - 1 > 200:
                        fitted.append(line)
                    else:
                        break
                parts.append("\n".join(fitted))
                budget = 0

        # ── Step 4: relevant common paths ────────────────────────────────────
        if budget > 150:
            path_lines: list[str] = []
            for name, keywords in _PATH_KEYWORDS.items():
                if any(kw in q for kw in keywords):
                    path = self._paths.get(name, "")
                    if path:
                        path_lines.append(path.strip())

            if path_lines:
                path_block = "PATHS:\n" + "\n".join(path_lines)
                if len(path_block) <= budget:
                    parts.append(path_block)

        return "\n\n".join(parts)

    # ── Full schema dump (offline / debugging only) ──────────────────────────

    def schema_addendum(self) -> str:
        """
        Full human-readable schema dump.

        WARNING: ~4500 tokens. Do NOT inject this into a 4096-token context.
        Use for documentation, offline inspection, or high-context models only.
        Call schema_for_question(question) at runtime instead.
        """
        lines = ["NODE DESCRIPTIONS (FULL):"]
        for label, meta in self._labels.items():
            desc = meta.get("description", "")
            lines.append(f"  {label}: {desc}")
            props = meta.get("properties", {}) or {}
            for prop_name, prop_meta in props.items():
                pdesc = (prop_meta or {}).get("description", "")
                if pdesc:
                    lines.append(f"    .{prop_name}: {pdesc}")
                vals = (prop_meta or {}).get("values") or {}
                if vals:
                    pairs = ", ".join(f"{k}={v}" for k, v in list(vals.items())[:5])
                    lines.append(f"      values: {pairs}")
        lines.append("")
        lines.append("RELATIONSHIP DESCRIPTIONS (FULL):")
        for rel, meta in self._rels.items():
            frm     = meta.get("from", "")
            to      = meta.get("to", "")
            example = meta.get("example", "")
            lines.append(f"  {rel}: {meta.get('description', '')}")
            if example:
                lines.append(f"    example: {example}")
        lines.append("")
        lines.append("COMMON PATHS:")
        for name, path in self._paths.items():
            lines.append(f"  {name}:")
            lines.append(f"    {path.strip()}")
        return "\n".join(lines)

    # ── Token estimator (utility) ────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count: 1 token ≈ 4 chars. Good enough for budget checks."""
        return max(1, len(text) // 4)

    def schema_token_count(self, question: str) -> dict:
        """Return token estimates for both schema modes — useful for debugging."""
        focused = self.schema_for_question(question)
        full    = self.schema_addendum()
        return {
            "focused_chars":  len(focused),
            "focused_tokens": self.estimate_tokens(focused),
            "full_chars":     len(full),
            "full_tokens":    self.estimate_tokens(full),
        }


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    with open(DICT_PATH, "r") as f:
        doc = yaml.safe_load(f)
    return Catalog(doc)