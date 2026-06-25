"""
Metadata catalog — loads the data dictionary and exposes lookups for:
  - friendly schema text (injected into the text2cypher prompt)
  - value translation (swap coded values for human-readable labels in answers)
  - PII property list (used by guardrails/output.py for redaction)
  - allowed-topic keywords (used by guardrails/input.py)
  - HTML rendering of the full catalog (for review + editing)

TOKEN BUDGET
------------
All tuning is in data_dictionary.yaml under `token_budget`.
At runtime always call schema_for_question(question) — not schema_addendum().
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DICT_PATH = Path(__file__).parent / "data_dictionary.yaml"


class Catalog:
    def __init__(self, doc: dict[str, Any]):
        self._doc = doc
        self._labels: dict[str, dict] = doc.get("labels", {})
        self._rels: dict[str, dict]   = doc.get("relationships", {})
        self._topics: list[str]       = [t.lower() for t in doc.get("allowed_topics", [])]
        self._paths: dict[str, str]   = doc.get("common_paths", {})

        # ── Config loaded from YAML (no hardcoding in Python) ────────────────
        budget_cfg                  = doc.get("token_budget", {})
        self._char_budget: int      = budget_cfg.get("schema_char_budget", 3600)
        self._min_rel_budget: int   = budget_cfg.get("min_rel_budget", 200)
        self._min_path_budget: int  = budget_cfg.get("min_path_budget", 150)
        self._max_values: int       = budget_cfg.get("max_values_shown", 4)
        self._max_prop_chars: int   = budget_cfg.get("max_prop_desc_chars", 60)

        self._label_keywords: dict[str, list[str]] = doc.get("label_keywords", {})
        self._path_keywords: dict[str, list[str]]  = doc.get("path_keywords", {})

        # Relationship endpoint index — built from relationships section
        self._rel_labels: dict[str, tuple[str, str]] = {
            rel: (meta.get("from", ""), meta.get("to", ""))
            for rel, meta in self._rels.items()
            if meta.get("from") and meta.get("to")
        }

        # PII index
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

    # ── Token-aware schema injection (USE THIS AT RUNTIME) ───────────────────

    def schema_for_question(self, question: str) -> str:
        """
        Compact schema block relevant to this question.
        Reads all tuning from data_dictionary.yaml token_budget + label_keywords.
        """
        q      = question.lower()
        budget = self._char_budget
        parts: list[str] = []

        # Step 1 — detect relevant labels
        relevant: list[str] = []
        for label, keywords in self._label_keywords.items():
            if any(kw in q for kw in keywords):
                relevant.append(label)
        if "Patient" not in relevant:
            relevant.insert(0, "Patient")

        # Step 2 — compact node + property block
        node_lines: list[str] = ["NODES:"]
        for label in relevant:
            meta = self._labels.get(label, {})
            desc = (meta.get("description", "") or "").split(".")[0]
            node_lines.append(f"  ({label}) — {desc}")
            for prop_name, prop_meta in (meta.get("properties") or {}).items():
                if not prop_meta:
                    continue
                pdesc = prop_meta.get("description", "")
                vals  = prop_meta.get("values") or {}
                if not pdesc and not vals:
                    continue
                if vals:
                    val_str = ", ".join(
                        f"{k}={v}" for k, v in list(vals.items())[:self._max_values]
                    )
                    node_lines.append(f"    .{prop_name}: {val_str}")
                else:
                    node_lines.append(f"    .{prop_name}: {pdesc[:self._max_prop_chars].rstrip()}")

        node_block = "\n".join(node_lines)
        if len(node_block) <= budget:
            parts.append(node_block)
            budget -= len(node_block)
        else:
            parts.append("NODES: " + ", ".join(relevant))
            budget -= len(parts[-1])

        # Step 3 — relevant relationship examples
        if budget > self._min_rel_budget:
            relevant_set = set(relevant)
            rel_lines: list[str] = ["RELATIONSHIPS:"]
            for rel, (frm, to) in self._rel_labels.items():
                if frm in relevant_set or to in relevant_set:
                    example = self._rels[rel].get("example", f"(:{frm})-[:{rel}]->(:{to})")
                    rel_lines.append(f"  {example}")

            rel_block = "\n".join(rel_lines)
            if len(rel_block) <= budget:
                parts.append(rel_block)
                budget -= len(rel_block)
            else:
                fitted = [rel_lines[0]]
                for line in rel_lines[1:]:
                    if budget - len("\n".join(fitted)) - len(line) - 1 > self._min_rel_budget:
                        fitted.append(line)
                    else:
                        break
                parts.append("\n".join(fitted))
                budget = 0

        # Step 4 — relevant common paths
        if budget > self._min_path_budget:
            path_lines: list[str] = []
            for name, keywords in self._path_keywords.items():
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
        """Full dump — ~4500 tokens. Use schema_for_question() at runtime."""
        lines = ["NODE DESCRIPTIONS (FULL):"]
        for label, meta in self._labels.items():
            lines.append(f"  {label}: {meta.get('description', '')}")
            for prop_name, prop_meta in (meta.get("properties") or {}).items():
                pdesc = (prop_meta or {}).get("description", "")
                if pdesc:
                    lines.append(f"    .{prop_name}: {pdesc}")
                vals = (prop_meta or {}).get("values") or {}
                if vals:
                    pairs = ", ".join(f"{k}={v}" for k, v in list(vals.items())[:5])
                    lines.append(f"      values: {pairs}")
        lines += ["", "RELATIONSHIP DESCRIPTIONS (FULL):"]
        for rel, meta in self._rels.items():
            lines.append(f"  {rel}: {meta.get('description', '')}")
            if meta.get("example"):
                lines.append(f"    example: {meta['example']}")
        lines += ["", "COMMON PATHS:"]
        for name, path in self._paths.items():
            lines.append(f"  {name}:\n    {path.strip()}")
        return "\n".join(lines)

    # ── HTML renderer ────────────────────────────────────────────────────────

    def to_html(self, output_path: Path | str | None = None) -> str:
        """
        Render the full catalog as a self-contained HTML page.

        Sections:
          - Token budget config
          - Node labels (properties + value maps + PII flags)
          - Relationships (with direction + example Cypher)
          - Common paths
          - Label keywords (what triggers each node in schema_for_question)
          - Allowed topics

        Usage:
            catalog.to_html("metadata/catalog.html")   # write file
            html = catalog.to_html()                    # get string
        """
        pii_all = self.all_pii_properties()

        def _badge(text: str, color: str) -> str:
            return (
                f'<span style="background:{color};color:#fff;padding:2px 7px;'
                f'border-radius:4px;font-size:11px;font-weight:600">{text}</span>'
            )

        def _code(text: str) -> str:
            return f'<code style="background:#f4f4f8;padding:1px 5px;border-radius:3px;font-size:12px">{text}</code>'

        lines: list[str] = []

        # ── HTML head ────────────────────────────────────────────────────────
        lines.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RP Knowledge Graph — Data Dictionary</title>
<style>
  :root {
    --ink: #1a1a2e; --ink-soft: #444466; --muted: #888;
    --bg: #f8f8fc; --surface: #fff; --line: #e0e0ee;
    --indigo: #4338ca; --teal: #0e8c6a; --amber: #a6650c;
    --red: #c0392b; --slate: #5b6472;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg);
         color: var(--ink); line-height: 1.6; font-size: 14px; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 24px 28px; }
  h1 { font-size: 26px; font-weight: 700; color: var(--indigo); margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 32px; }
  h2 { font-size: 18px; font-weight: 600; margin: 36px 0 14px;
       padding-bottom: 6px; border-bottom: 2px solid var(--indigo);
       color: var(--indigo); }
  h3 { font-size: 15px; font-weight: 600; margin: 18px 0 6px; color: var(--ink); }
  .node-card { background: var(--surface); border: 1px solid var(--line);
               border-radius: 10px; padding: 16px 20px; margin-bottom: 14px; }
  .node-title { font-size: 16px; font-weight: 700; color: var(--indigo);
                margin-bottom: 4px; }
  .node-desc { color: var(--ink-soft); font-size: 13px; margin-bottom: 10px; }
  .prop-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .prop-table th { text-align: left; padding: 5px 10px; background: #f0f0f8;
                   color: var(--slate); font-weight: 600; border-bottom: 1px solid var(--line); }
  .prop-table td { padding: 5px 10px; border-bottom: 1px solid #f0f0f0;
                   vertical-align: top; }
  .prop-table tr:last-child td { border-bottom: none; }
  .val-list { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 3px; }
  .val-chip { background: #eef; border: 1px solid #ccd; border-radius: 4px;
              padding: 1px 7px; font-size: 11px; color: var(--ink-soft); }
  .rel-card { background: var(--surface); border: 1px solid var(--line);
              border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
              display: grid; grid-template-columns: 200px 1fr; gap: 12px; }
  .rel-name { font-weight: 700; color: var(--teal); font-size: 13px; }
  .rel-arrow { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .rel-desc { font-size: 13px; color: var(--ink-soft); }
  .rel-example { margin-top: 6px; }
  .path-block { background: #f4f8f4; border-left: 3px solid var(--teal);
                border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; }
  .path-name { font-weight: 600; color: var(--teal); margin-bottom: 4px; }
  .kw-section { background: var(--surface); border: 1px solid var(--line);
                border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
  .kw-label { font-weight: 700; color: var(--indigo); margin-bottom: 6px; font-size: 13px; }
  .kw-chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .kw-chip { background: #ede8ff; border: 1px solid #c5b8f5; border-radius: 4px;
             padding: 2px 8px; font-size: 11px; color: var(--indigo); }
  .budget-table { border-collapse: collapse; width: auto; font-size: 13px; }
  .budget-table td, .budget-table th { padding: 6px 14px; border: 1px solid var(--line); }
  .budget-table th { background: #f0f0f8; }
  .topic-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
  .topic-chip { background: #fff8e8; border: 1px solid #e8d8a0; border-radius: 4px;
                padding: 2px 8px; font-size: 11px; color: var(--amber); }
  .pii-badge { background: #fde8e8; color: var(--red); border: 1px solid #f5b8b8;
               border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 700; }
  nav { position: sticky; top: 0; background: var(--indigo); color: #fff;
        padding: 10px 28px; display: flex; gap: 20px; font-size: 13px; z-index: 100; }
  nav a { color: rgba(255,255,255,0.85); text-decoration: none; font-weight: 500; }
  nav a:hover { color: #fff; }
</style>
</head>
<body>
<nav>
  <a href="#nodes">Nodes</a>
  <a href="#relationships">Relationships</a>
  <a href="#paths">Common Paths</a>
  <a href="#keywords">Label Keywords</a>
  <a href="#topics">Allowed Topics</a>
  <a href="#budget">Token Budget</a>
</nav>
<div class="wrap">
<h1>RP Knowledge Graph — Data Dictionary</h1>
<p class="subtitle">Auto-generated from data_dictionary.yaml &nbsp;·&nbsp;
Edit the YAML file and re-run <code>catalog.to_html()</code> to refresh.</p>
""")

        # ── Token budget ──────────────────────────────────────────────────────
        lines.append('<h2 id="budget">Token Budget</h2>')
        lines.append('<table class="budget-table"><tr><th>Setting</th><th>Value</th><th>Purpose</th></tr>')
        budget_cfg = self._doc.get("token_budget", {})
        budget_meta = {
            "schema_char_budget":   "Max chars for schema block (~tokens×4)",
            "min_rel_budget":       "Min remaining chars before adding relationships",
            "min_path_budget":      "Min remaining chars before adding common paths",
            "max_values_shown":     "Max value examples shown per property",
            "max_prop_desc_chars":  "Max chars for property description",
        }
        for k, purpose in budget_meta.items():
            v = budget_cfg.get(k, "—")
            lines.append(f"<tr><td><b>{k}</b></td><td>{v}</td><td>{purpose}</td></tr>")
        lines.append("</table>")

        # ── Node labels ───────────────────────────────────────────────────────
        lines.append('<h2 id="nodes">Node Labels</h2>')
        for label, meta in self._labels.items():
            pii_props = self._pii_by_label.get(label, set())
            lines.append(f'<div class="node-card">')
            lines.append(f'<div class="node-title">:{label}</div>')
            desc = (meta.get("description") or "").replace("\n", " ").strip()
            lines.append(f'<div class="node-desc">{desc}</div>')

            if pii_props:
                pii_list = " ".join(
                    f'<span class="pii-badge">🔒 {p}</span>' for p in sorted(pii_props)
                )
                lines.append(f'<div style="margin-bottom:8px">PII fields: {pii_list}</div>')

            props = meta.get("properties") or {}
            if props:
                lines.append('<table class="prop-table">')
                lines.append("<tr><th>Property</th><th>Description</th><th>Values</th></tr>")
                for prop_name, prop_meta in props.items():
                    if not prop_meta:
                        continue
                    pdesc = (prop_meta.get("description") or "").replace("\n", " ").strip()
                    vals  = prop_meta.get("values") or {}
                    is_pii = prop_name in pii_props
                    pii_marker = ' <span class="pii-badge">🔒 PII</span>' if is_pii else ""

                    val_html = ""
                    if vals:
                        chips = "".join(
                            f'<span class="val-chip">'
                            f'<b>{k}</b> → {v}'
                            f'</span>'
                            for k, v in vals.items()
                        )
                        val_html = f'<div class="val-list">{chips}</div>'

                    lines.append(
                        f"<tr>"
                        f"<td><b>{prop_name}</b>{pii_marker}</td>"
                        f"<td>{pdesc}</td>"
                        f"<td>{val_html}</td>"
                        f"</tr>"
                    )
                lines.append("</table>")
            lines.append("</div>")

        # ── Relationships ─────────────────────────────────────────────────────
        lines.append('<h2 id="relationships">Relationships</h2>')
        for rel, meta in self._rels.items():
            frm     = meta.get("from", "?")
            to      = meta.get("to", "?")
            desc    = (meta.get("description") or "").replace("\n", " ").strip()
            example = meta.get("example", "")
            lines.append('<div class="rel-card">')
            lines.append(
                f'<div>'
                f'<div class="rel-name">:{rel}</div>'
                f'<div class="rel-arrow">({frm}) → ({to})</div>'
                f'</div>'
            )
            lines.append(
                f'<div>'
                f'<div class="rel-desc">{desc}</div>'
            )
            if example:
                lines.append(
                    f'<div class="rel-example">'
                    f'<code style="font-size:12px;color:#0e8c6a">{example}</code>'
                    f'</div>'
                )
            lines.append("</div></div>")

        # ── Common paths ──────────────────────────────────────────────────────
        lines.append('<h2 id="paths">Common Multi-hop Paths</h2>')
        for name, path in self._paths.items():
            kws = self._path_keywords.get(name, [])
            kw_chips = "".join(f'<span class="kw-chip">{k}</span>' for k in kws)
            lines.append(
                f'<div class="path-block">'
                f'<div class="path-name">{name}</div>'
                f'<code style="white-space:pre;font-size:12px">{path.strip()}</code>'
                f'<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">'
                f'<span style="font-size:11px;color:var(--muted)">triggers on:</span> {kw_chips}'
                f'</div>'
                f'</div>'
            )

        # ── Label keywords ────────────────────────────────────────────────────
        lines.append('<h2 id="keywords">Label Keywords</h2>')
        lines.append(
            '<p style="color:var(--muted);font-size:13px;margin-bottom:12px">'
            'These keywords trigger inclusion of each node label in '
            '<code>schema_for_question()</code>. Edit in '
            '<code>label_keywords</code> section of data_dictionary.yaml.</p>'
        )
        for label, keywords in self._label_keywords.items():
            chips = "".join(f'<span class="kw-chip">{k}</span>' for k in keywords)
            lines.append(
                f'<div class="kw-section">'
                f'<div class="kw-label">:{label}</div>'
                f'<div class="kw-chips">{chips}</div>'
                f'</div>'
            )

        # ── Allowed topics ────────────────────────────────────────────────────
        lines.append('<h2 id="topics">Allowed Topics (Guardrail)</h2>')
        lines.append(
            '<p style="color:var(--muted);font-size:13px;margin-bottom:8px">'
            'Questions must contain at least one of these substrings to pass '
            'the input guardrail. Edit in <code>allowed_topics</code> section.</p>'
        )
        chips = "".join(f'<span class="topic-chip">{t}</span>' for t in self._topics)
        lines.append(f'<div class="topic-chips">{chips}</div>')

        lines.append("</div></body></html>")

        html = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    # ── Token estimator ───────────────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def schema_token_count(self, question: str) -> dict:
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


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "catalog.html"
    cat = get_catalog()
    cat.to_html(out)
    print(f"Written: {out}")
    q = "show patients with bad debt at RADM"
    counts = cat.schema_token_count(q)
    print(f"Token check — focused: {counts['focused_tokens']} tokens  "
          f"full: {counts['full_tokens']} tokens")