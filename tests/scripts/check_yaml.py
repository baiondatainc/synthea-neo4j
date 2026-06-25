#!/usr/bin/env python3
"""
Diagnose the YAML syntax error in data_dictionary.yaml.

Run from the repo root:
    python scripts/check_yaml.py
    python scripts/check_yaml.py path/to/other.yaml

The YAML SafeLoader error "could not find expected ':'" at line N usually means:
  - A mapping key longer than 1024 characters (YAML's simple-key limit)
  - A key that spans multiple lines without block-scalar syntax
  - A stray unquoted colon inside a value

This script shows the offending lines with context so you can fix them.
"""
from __future__ import annotations

import sys
from pathlib import Path
print(Path(__file__).resolve().parents[0] )

DEFAULT = Path.cwd() / "metadata" / "data_dictionary.yaml"


def check(path: Path) -> None:
    print(f"Checking: {path}\n")
    lines = path.read_text(encoding="utf-8").splitlines()

    # ── Heuristic 1: keys > 1024 chars ───────────────────────────────────────
    print("=== Keys longer than 1024 characters ===")
    found_long = False
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        # A YAML mapping key line looks like "  some_key: value" or "  some_key:"
        if ":" in stripped:
            key_part = stripped.split(":", 1)[0]
            if len(key_part) > 1024:
                found_long = True
                print(f"  Line {i}: key length={len(key_part)} — {key_part[:120]}...")
    if not found_long:
        print("  (none found — key-length is probably not the issue)")

    # ── Heuristic 2: show lines 375–390 (where the scanner errored) ───────────
    ERROR_LINE = 381  # from the traceback
    CONTEXT = 8
    start = max(0, ERROR_LINE - CONTEXT - 1)
    end = min(len(lines), ERROR_LINE + CONTEXT)
    print(f"\n=== Lines {start + 1}–{end} (error near line {ERROR_LINE}) ===")
    for i, line in enumerate(lines[start:end], start + 1):
        marker = " >>>" if i == ERROR_LINE else "    "
        print(f"{marker} {i:>4}: {line}")

    # ── Heuristic 3: try yaml.safe_load and print the exact error ─────────────
    print("\n=== yaml.safe_load result ===")
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            yaml.safe_load(f)
        print("  ✓ Parsed successfully — no error found.")
    except yaml.YAMLError as e:
        print(f"  ✗ YAMLError: {e}")
        # Give a fix hint based on the error message
        msg = str(e)
        if "simple key" in msg or "could not find expected ':'" in msg:
            print(
                "\n  FIX HINTS:\n"
                "  1. If the key at the error line is very long (>1024 chars),\n"
                "     wrap it in quotes: '\"very long key...\": value'\n"
                "  2. If the key text contains a colon (:), wrap the key in quotes.\n"
                "  3. If the key spans multiple lines, use a block scalar (|) or\n"
                "     move the content into the value portion.\n"
                "  4. Check for missing ':' after the key on that line.\n"
            )


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    check(p)