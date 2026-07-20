# label_audit.py — run in the training venv
import unsloth
from unsloth import FastLanguageModel
import pandas as pd, sys
sys.path.insert(0, "model_tuning")
from prompt import SYSTEM_PROMPT, RESPONSE_TEMPLATE

_, tok = FastLanguageModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",
    max_seq_length=10240, load_in_4bit=True)
df = pd.read_parquet("data/splits/train.parquet")
msgs = [{"role":"system","content":SYSTEM_PROMPT.strip()},
        {"role":"user","content":df.iloc[0]["question"]},
        {"role":"assistant","content":df.iloc[0]["cypher"]}]
ids = tok.apply_chat_template(msgs, tokenize=True)
tmpl = tok.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
for j in range(len(ids)-len(tmpl)+1):
    if ids[j:j+len(tmpl)] == tmpl:
        print("template at", j, "| unmasked label tokens:", len(ids)-(j+len(tmpl)))
        print("cypher token count should be ≈", len(tok.encode(df.iloc[0]["cypher"])))
        break