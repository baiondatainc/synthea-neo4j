# remerge_fp16.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "lora_adapter_q3_v1")
model = model.merge_and_unload()
model.save_pretrained("merged_fp16")
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507").save_pretrained("merged_fp16")