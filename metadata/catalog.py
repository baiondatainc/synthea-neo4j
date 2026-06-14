"""
Metadata catalog — loads the data dictionary and exposes lookups for:
  - friendly schema text (injected into the text2cypher prompt)
  - value translation (swap coded values for human-readable labels in answers)
  - PII property list (used by guardrails/output.py for redaction)
  - allowed-topic keywords (used by guardrails/input.py)
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DICT_PATH = Path(__file__).parent / "data_dictionary.yaml"


class Catalog:
    def __init__(self, doc: dict[str, Any]):
        self._labels: dict[str, dict] = doc.get("labels", {})
        self._rels: dict[str, dict] = doc.get("relationships", {})
        self._topics: list[str] = [t.lower() for t in doc.get("allowed_topics", [])]

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
                    self._value_map[label][prop_name] = vals

    @property
    def labels(self) -> dict[str, dict]:
        return self._labels

    @property
    def relationships(self) -> dict[str, dict]:
        return self._rels

    @property
    def allowed_topics(self) -> list[str]:
        return self._topics

    def pii_properties(self, label: str) -> set[str]:
        return self._pii_by_label.get(label, set())

    def all_pii_properties(self) -> set[str]:
        out: set[str] = set()
        for props in self._pii_by_label.values():
            out.update(props)
        return out

    def translate_value(self, label: str, prop: str, code: Any) -> Any:
        if code is None:
            return code
        key = str(code).lower()
        return self._value_map.get(label, {}).get(prop, {}).get(key, code)

    def schema_addendum(self) -> str:
        """Human-readable description block to append to the text2cypher prompt."""
        lines = ["NODE DESCRIPTIONS:"]
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
        lines.append("RELATIONSHIP DESCRIPTIONS:")
        for rel, meta in self._rels.items():
            lines.append(f"  {rel}: {meta.get('description', '')}")
        return "\n".join(lines)

    def humanize_row(self, row: dict, label_hint: str | None = None) -> dict:
        """Best-effort: translate coded values in a single result row.

        Doesn't know which Label each column came from, so it tries every label
        and falls through on no match.
        """
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

    def humanize_rows(self, rows: list[dict], label_hint: str | None = None) -> list[dict]:
        return [self.humanize_row(r, label_hint) for r in rows]


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    with open(DICT_PATH, "r") as f:
        doc = yaml.safe_load(f)
    return Catalog(doc)
