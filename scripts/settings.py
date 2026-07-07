"""
Patch to apply to settings.py — change LORA_BASE_MODEL to Qwen2.5-Coder.
Run: python settings_patch.py
Or manually change the line in settings.py.
"""
from pathlib import Path
import re

settings_path = Path("settings.py")
if not settings_path.exists():
    print("settings.py not found — run from project root")
    exit(1)

content = settings_path.read_text()

# Replace base model
old = 'LORA_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"'
new = 'LORA_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"'

if old in content:
    content = content.replace(old, new)
    settings_path.write_text(content)
    print(f"✓ Changed: {old}")
    print(f"       to: {new}")
elif "Qwen/Qwen2.5-Coder-7B-Instruct" in content:
    print("✓ Already set to Qwen2.5-Coder-7B-Instruct")
else:
    print("⚠  LORA_BASE_MODEL line not found in expected format")
    print("  Manually set: LORA_BASE_MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'")
    # Try to find it anyway
    matches = [l for l in content.split('\n') if 'LORA_BASE_MODEL' in l]
    print(f"  Current lines: {matches}")