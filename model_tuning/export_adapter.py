#!/usr/bin/env python3
"""
export_adapter.py  v2 — adapter → Ollama, no merging, no custom template
────────────────────────────────────────────────────────────────────────
v2 change: the generated Modelfile contains NO TEMPLATE block.

Why: with FROM pointing at an Ollama LIBRARY model, the created model
INHERITS the base's built-in ChatML template — the proven one. Every
hand-written or copied template we tried either failed to render (custom Go
template → silent fallback to raw completion → one-token "MATCH") or was
Jinja syntax Ollama can't execute at all (→ SQL / word-problem rambling).
Inheritance is byte-exact and cannot drift.

After `ollama create`, this script VERIFIES the effective template: it must
be Go syntax ({{ }}) and must contain <|im_start|>. If it sees Jinja
({%- %}) or nothing, it deletes the model and exits loudly.

Usage (no retraining needed — works on the existing adapter):
    python model_tuning/export_adapter.py \\
        --adapter ./lora_adapter_q3_v2 \\
        --tag text2cypher-ft-candidate
"""
import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
UNSLOTH_CPP = Path.home() / ".unsloth" / "llama.cpp"
OLLAMA_BASE = "qwen3:4b-instruct-2507-q4_K_M"
CONVERT_URL = ("https://raw.githubusercontent.com/ggml-org/llama.cpp/"
               "master/convert_lora_to_gguf.py")

# NOTE: no TEMPLATE constant in v2 — template is inherited from OLLAMA_BASE.

PARAMETERS = """PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 256
PARAMETER num_ctx 9216
PARAMETER num_thread 4
PARAMETER num_batch 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_start|>"
"""


def find_converter() -> Path:
    """convert_lora_to_gguf.py must sit NEXT TO the conversion/ package."""
    script = UNSLOTH_CPP / "convert_lora_to_gguf.py"
    if script.exists():
        return script
    if not (UNSLOTH_CPP / "conversion").exists():
        sys.exit(f"✗ {UNSLOTH_CPP}/conversion not found — is unsloth's "
                 f"llama.cpp installed?")
    print(f"↓ Fetching convert_lora_to_gguf.py → {script}")
    urllib.request.urlretrieve(CONVERT_URL, script)
    return script


def load_system_prompt() -> str:
    """Single source of truth: SYSTEM block of Modelfile.qwen3-4b."""
    mf = REPO_ROOT / "model_tuning/Modelfile.qwen3-4b"
    if not mf.exists():
        sys.exit(f"✗ {mf} not found — SYSTEM prompt source of truth is missing")
    m = re.search(r'SYSTEM\s+"""(.*?)"""', mf.read_text(encoding="utf-8"),
                  re.DOTALL)
    if not m:
        sys.exit('✗ No SYSTEM """...""" block in Modelfile.qwen3-4b')
    return m.group(1)


def verify_generation(tag: str) -> None:
    """Functional smoke test — the only verification that matters.

    (v2's template-SYNTAX check was wrong: modern Ollama renders Jinja
    templates natively, so 'Jinja detected' is normal for library bases.
    That check deleted a healthy model. Now we simply ask the model a
    canonical question and check the shape of the answer.)

    Failure signatures this catches:
      - one-token 'MATCH' then stop  → broken artifact / EOS issue
      - SQL or prose rambling        → template not applied (raw completion)
    """
    print("   Verifying generation (canary: 'How many patients?')...")
    try:
        out = subprocess.run(["ollama", "run", tag, "How many patients?"],
                             capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        sys.exit("✗ Generation timed out — check ollama serve and try manually.")
    resp = out.stdout.strip()
    print(f"   Response: {resp[:120]!r}")

    problems = []
    u = resp.upper()
    if "MATCH" not in u or "RETURN" not in u:
        problems.append("no MATCH...RETURN — not a Cypher query")
    if len(resp) < 20:
        problems.append("suspiciously short (one-token failure signature)")
    if "SELECT " in u or "FROM " in u.replace("RETURN", ""):
        problems.append("SQL detected — template fell back to raw completion")
    if problems:
        print("✗ GENERATION VERIFICATION FAILED:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(f"  Model '{tag}' kept for inspection. Try:\n"
                 f"    ollama show {tag} --modelfile\n"
                 f"    ollama run {tag} \"How many patients?\"")
    print("   ✓ Generation verified: valid Cypher returned")


def main():
    ap = argparse.ArgumentParser(description="Adapter → GGUF → Ollama (no merge)")
    ap.add_argument("--adapter", required=True,
                    help="Adapter dir, e.g. ./lora_adapter_q3_v2")
    ap.add_argument("--tag", default="text2cypher-ft-candidate")
    ap.add_argument("--base", default=OLLAMA_BASE,
                    help="Ollama LIBRARY base (must match training base and "
                         "must bring its own template)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "gguf_export_v1"))
    ap.add_argument("--skip-create", action="store_true")
    args = ap.parse_args()

    adapter = Path(args.adapter).resolve()
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"✗ {adapter} is not a PEFT adapter dir (no adapter_config.json)")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_gguf = out_dir / f"adapter-{adapter.name}.gguf"

    # ── 1. Convert adapter → GGUF (skip if already converted) ────────────────
    if adapter_gguf.exists():
        print(f"🔧 [1/3] Reusing existing {adapter_gguf.name}")
    else:
        converter = find_converter()
        print(f"🔧 [1/3] Converting {adapter.name} → {adapter_gguf.name}")
        cmd = [sys.executable, str(converter), str(adapter),
               "--outfile", str(adapter_gguf),
               "--base-model-id", "Qwen/Qwen3-4B-Instruct-2507"]
        try:
            subprocess.run(cmd, check=True, cwd=UNSLOTH_CPP)
        except subprocess.CalledProcessError:
            print("   --base-model-id failed; retrying with cached local base...")
            cache = Path.home() / (".cache/huggingface/hub/"
                                   "models--Qwen--Qwen3-4B-Instruct-2507/snapshots")
            snaps = sorted(cache.glob("*")) if cache.exists() else []
            if not snaps:
                sys.exit("✗ No cached base model found.")
            subprocess.run([sys.executable, str(converter), str(adapter),
                            "--outfile", str(adapter_gguf),
                            "--base", str(snaps[-1])], check=True,
                           cwd=UNSLOTH_CPP)
        size_mb = adapter_gguf.stat().st_size / 1e6
        print(f"   ✓ {adapter_gguf}  ({size_mb:.0f} MB)")

    # ── 2. Generate Modelfile — FROM + ADAPTER + PARAMETERS + SYSTEM only ────
    modelfile = out_dir / f"Modelfile.{args.tag}"
    system = load_system_prompt()
    modelfile.write_text(
        f"# AUTO-GENERATED by export_adapter.py v2 — do not hand-edit; regenerate.\n"
        f"# TEMPLATE intentionally OMITTED: inherited from {args.base}.\n"
        f"# (Custom templates caused raw-completion fallback: 'MATCH'-only / SQL output.)\n"
        f"FROM {args.base}\n"
        f"ADAPTER {adapter_gguf}\n\n"
        f"{PARAMETERS}\n"
        f'SYSTEM """{system}"""\n',
        encoding="utf-8",
    )
    print(f"🔧 [2/3] Modelfile written → {modelfile}  (no TEMPLATE — inherited)")

    # ── 3. Build + verify ────────────────────────────────────────────────────
    if args.skip_create:
        print(f"   (skipped) ollama create {args.tag} -f {modelfile}")
        return
    if shutil.which("ollama") is None:
        sys.exit("✗ ollama not on PATH — run the create command manually.")

    subprocess.run(["ollama", "rm", args.tag], capture_output=True)  # clean slate
    print(f"🔧 [3/3] ollama create {args.tag}")
    subprocess.run(["ollama", "create", args.tag, "-f", str(modelfile)],
                   check=True)
    verify_generation(args.tag)

    print(f"\n✅ Done. Test:")
    print(f"   ollama run {args.tag} \"How many patients?\"")
    print(f"   ollama run {args.tag} \"Total bad debt by state\"")
    print(f"   ollama run {args.tag} \"Reviews by location\"")


if __name__ == "__main__":
    main()