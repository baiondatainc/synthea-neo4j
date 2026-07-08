#!/usr/bin/env python3
"""
export_to_ollama.py  v2
───────────────────────
Exports lora_adapter_v3 to GGUF using Unsloth's native approach
(load base + adapter together, not separately).

Usage:
    python export_to_ollama.py --adapter ./lora_adapter_v3 --tag rp-cypher-v3
"""

import argparse, os, re, subprocess, sys
from pathlib import Path

LORA_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def export_gguf(adapter_path: str, gguf_dir: str) -> Path:
    print(f"\n{'='*60}")
    print(f"  Exporting GGUF from {adapter_path}")
    print(f"{'='*60}")

    import unsloth  # noqa
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    os.makedirs(gguf_dir, exist_ok=True)

    # KEY FIX: load the adapter_path directly as model_name
    # Unsloth reads adapter_config.json and loads base + adapter together
    print(f"  Loading adapter directly as model_name ...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = adapter_path,   # ← adapter path, not base model
            max_seq_length = 4096,
            dtype          = None,
            load_in_4bit   = False,          # need full precision for merge
        )
        tokenizer = get_chat_template(tokenizer, "qwen-2.5")
        print(f"  ✓ Loaded via adapter path")
    except Exception as e:
        print(f"  Method 1 failed: {e}")
        print(f"  Trying method 2: load base + merge adapter manually...")
        try:
            from peft import PeftModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name     = LORA_BASE_MODEL,
                max_seq_length = 4096,
                dtype          = None,
                load_in_4bit   = False,
            )
            tokenizer  = get_chat_template(tokenizer, "qwen-2.5")
            # Use PEFT directly without Unsloth wrapper
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()   # merge LoRA into base weights
            print(f"  ✓ Loaded and merged via PeftModel.merge_and_unload()")
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            sys.exit(1)

    # Export to GGUF
    print(f"\n  Saving GGUF q4_k_m to {gguf_dir} ...")
    try:
        model.save_pretrained_gguf(
            gguf_dir, tokenizer,
            quantization_method="q4_k_m",
        )
        gguf_files = sorted(Path(gguf_dir).glob("**/*.gguf"))
        if gguf_files:
            gguf_path = gguf_files[0]
            size_gb   = gguf_path.stat().st_size / 1e9
            print(f"  ✓ GGUF: {gguf_path}  ({size_gb:.2f} GB)")
            return gguf_path
        else:
            print(f"  ✗ No .gguf file found in {gguf_dir}")
            sys.exit(1)
    except Exception as e:
        print(f"  GGUF q4_k_m failed: {e}")
        print(f"  Trying q8_0 quantisation instead...")
        try:
            model.save_pretrained_gguf(
                gguf_dir, tokenizer,
                quantization_method="q8_0",
            )
            gguf_files = sorted(Path(gguf_dir).glob("**/*.gguf"))
            if gguf_files:
                gguf_path = gguf_files[0]
                print(f"  ✓ GGUF (q8_0): {gguf_path}")
                return gguf_path
        except Exception as e2:
            print(f"  q8_0 also failed: {e2}")

        print(f"\n  Saving merged 16-bit weights instead...")
        merged_dir = gguf_dir + "_merged"
        os.makedirs(merged_dir, exist_ok=True)
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"  ✓ Merged weights saved to {merged_dir}")
        print(f"\n  Convert manually with llama.cpp:")
        print(f"    git clone https://github.com/ggerganov/llama.cpp")
        print(f"    cd llama.cpp && pip install -r requirements.txt")
        print(f"    python convert_hf_to_gguf.py {merged_dir} --outtype q4_k_m")
        sys.exit(1)


def update_modelfile(gguf_path: Path, tag: str, modelfile_src: str = None) -> str:
    print(f"\n{'='*60}")
    print(f"  Updating Modelfile")
    print(f"{'='*60}")

    base_modelfile = Path(modelfile_src) if modelfile_src else Path("Modelfile.jp")
    if base_modelfile.exists():
        content = base_modelfile.read_text()
        old_from = re.search(r"^FROM .+$", content, re.MULTILINE)
        if old_from:
            content = content.replace(
                old_from.group(0),
                f"FROM {gguf_path.resolve()}"
            )
            print(f"  Replaced: {old_from.group(0)[:60]}")
        else:
            content = f"FROM {gguf_path.resolve()}\n" + content
    else:
        # Create minimal Modelfile with system prompt
        content = f"""FROM {gguf_path.resolve()}

PARAMETER temperature 0
PARAMETER top_k 1
PARAMETER repeat_penalty 1.05
PARAMETER num_predict 300
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "\\nQ:"

SYSTEM \"""You are a Neo4j Cypher generator for the RP (RP) knowledge graph.
Output ONLY raw Cypher. No markdown. No explanations. ALWAYS include LIMIT. Alias every property.

Labels: Patient, Visit, Charge, Transaction, Statement, RCCall, IVRInbound,
        DiallerCall, PhoneBridge, Campaign, Location, InsurancePlan,
        Practice, DiagnosisCode, ProcedureCode, BirdeyeReview

Key relationships:
  (Patient)-[:HAD_VISIT]->(Visit)  (Patient)-[:HAS_CHARGE]->(Charge)
  (Patient)-[:HAS_TRANSACTION]->(Transaction)  (Patient)-[:RECEIVED_STATEMENT]->(Statement)
  (Patient)-[:IDENTIFIED_BY_PHONE]->(PhoneBridge)  (Patient)-[:CALLED_IVR]->(IVRInbound)
  (Transaction)-[:SETTLES]->(Charge)  (RCCall)-[:ATTRIBUTED_TO_PHONE]->(PhoneBridge)
  (Charge)-[:AT_LOCATION]->(Location)  (Visit)-[:PERFORMED_AT]->(Location)
  (BirdeyeReview)-[:REVIEWS]->(Location)

Q: How many patients?
MATCH (p:Patient) RETURN count(p) AS patient_count LIMIT 1

Q: \"""
"""

    new_modelfile = f"Modelfile.{tag}"
    Path(new_modelfile).write_text(content)
    print(f"  ✓ Written: {new_modelfile}")
    return new_modelfile


def register_with_ollama(modelfile: str, tag: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Registering '{tag}' with Ollama")
    print(f"{'='*60}")

    result = subprocess.run(
        ["ollama", "create", tag, "-f", modelfile],
        text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ Failed. Run manually:")
        print(f"    ollama create {tag} -f {modelfile}")
        return False

    # Verify
    check = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if tag in check.stdout:
        print(f"  ✓ '{tag}' registered successfully")
        return True
    print(f"  ⚠  Not found in ollama list — may need a moment")
    return True


def quick_test(tag: str):
    import json, urllib.request
    print(f"\n{'='*60}")
    print(f"  Quick test: {tag}")
    print(f"{'='*60}")
    q = "How many patients have an outstanding balance?"
    payload = json.dumps({
        "model": tag, "stream": False,
        "options": {"temperature": 0, "num_predict": 150},
        "messages": [{"role": "user", "content": q}],
    }).encode()
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read()).get("message", {}).get("content", "").strip()
        print(f"  Q: {q}")
        print(f"  A: {r[:300]}")
        if "MATCH" in r.upper():
            print(f"  ✓ Generating Cypher correctly")
        else:
            print(f"  ⚠  Doesn't look like Cypher — check Modelfile system prompt")
    except Exception as e:
        print(f"  ✗ Test error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  default="../lora_adapter_v3")
    parser.add_argument("--gguf-dir",   default="../gguf_export_v3")
    parser.add_argument("--modelfile",  default=None,
                        help="Path to Modelfile.jp (auto-detected if not set)")
    parser.add_argument("--tag",      default="rp-cypher-v3")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip GGUF export, use --gguf-path instead")
    parser.add_argument("--gguf-path", default=None,
                        help="Existing GGUF file to use with --skip-export")
    args = parser.parse_args()

    print("=" * 60)
    print("  RP Text2Cypher — Export LoRA → Ollama")
    print(f"  Adapter: {args.adapter}")
    print(f"  Tag:     {args.tag}")
    print("=" * 60)

    if args.skip_export and args.gguf_path:
        gguf_path = Path(args.gguf_path)
        if not gguf_path.exists():
            print(f"  ✗ GGUF not found: {gguf_path}")
            sys.exit(1)
        print(f"  Using existing GGUF: {gguf_path}")
    else:
        gguf_path = export_gguf(args.adapter, args.gguf_dir)

    # Find Modelfile — check arg, project root, script dir
    modelfile_path = args.modelfile
    if not modelfile_path:
        for candidate in ["Modelfile.jp", "../Modelfile.jp", "../../Modelfile.jp"]:
            if Path(candidate).exists():
                modelfile_path = candidate
                print(f"  Found Modelfile: {candidate}")
                break
    modelfile = update_modelfile(gguf_path, args.tag, modelfile_path)
    ok        = register_with_ollama(modelfile, args.tag)

    if ok:
        quick_test(args.tag)

    print(f"\n{'='*60}")
    print(f"  Complete!")
    print(f"{'='*60}")
    print(f"\n  Test interactively:")
    print(f"    ollama run {args.tag}")
    print(f"\n  Use in demo:")
    print(f"    python demo_inference.py --mode ollama --model {args.tag} \\")
    print(f"      --uri bolt://localhost:7687 --user neo4j --password <pw> --demo")


if __name__ == "__main__":
    main()