# ─────────────────────────────────────────────────────────────────────────────
# BASE MODEL — Qwen3-4B-Instruct-2507  (matches Modelfile.qwen3-4b production)
#
# CHANGED (v6):
#   - Base model: Qwen2.5-Coder-7B  →  Qwen3-4B-Instruct-2507 (non-thinking)
#     Must match the Ollama FROM line: qwen3:4b-instruct-2507-q4_K_M
#   - SYSTEM_PROMPT is no longer duplicated here. It is parsed straight out of
#     Modelfile.qwen3-4b so training and serving can never drift apart.
#   - WRONG_PROPS updated for the new schema:
#       * c.service_date REMOVED from the blocklist — it is now a VALID
#         property (Charge.service_date, used in the Modelfile examples)
#       * p.patient_id ADDED — does not exist, must be p.patientId
#       * v.visit_date ADDED — Visit has admit_date / discharge_date only
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
from pathlib import Path

LORA_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Alternative: unsloth pre-quantized mirror (faster download, same weights):
# LORA_BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — single source of truth is Modelfile.qwen3-4b
#
# The Modelfile SYSTEM block is ~7-8k tokens. Duplicating it here caused the
# v5 drift (compact 2324-token prompt in training vs full prompt in serving,
# and wrong property names like patient_id / total_charged on Patient).
# We now parse the SYSTEM """...""" block directly from the Modelfile.
# ─────────────────────────────────────────────────────────────────────────────

def _find_modelfile() -> Path:
    # 1. explicit override
    env = os.environ.get("MODELFILE_PATH")
    if env and Path(env).exists():
        return Path(env)
    # 2. common locations relative to this file (scripts/prompt.py → repo root)
    here = Path(__file__).resolve().parent
    for candidate in (
        here / "Modelfile.qwen3-4b",
        here.parent / "Modelfile.qwen3-4b",
        Path.cwd() / "Modelfile.qwen3-4b",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Modelfile.qwen3-4b not found. Set MODELFILE_PATH env var or place it "
        "in the repo root. SYSTEM_PROMPT is parsed from it so training and "
        "Ollama serving stay identical."
    )


def _load_system_prompt() -> str:
    text = _find_modelfile().read_text(encoding="utf-8")
    m = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    if not m:
        raise RuntimeError('No SYSTEM """...""" block found in Modelfile.qwen3-4b')
    return m.group(1)


SYSTEM_PROMPT = _load_system_prompt()

TEST_QUESTIONS = [
    "How many patients have an outstanding balance greater than zero?",
    "What is the total outstanding balance grouped by payor cohort?",
    "Give me patient visit trends",
    "Which locations have the highest Birdeye average rating?",
    "How many unique patients visited in the state of TX in 2023?",
]

# Qwen3-4B-Instruct-2507 uses the same ChatML template as Qwen2.5
# (<|im_start|>role ... <|im_end|>), and the 2507 *instruct* variant emits NO
# <think> blocks — so the response template is unchanged.
# Loss is computed only on tokens after this marker.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GATE
# ─────────────────────────────────────────────────────────────────────────────

WRITE_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|DETACH)\b", re.I)

# Aggregate detector — used to decide whether a LIMIT is mandatory.
# The Modelfile only requires LIMIT for non-aggregate result sets
# ("How many patients?" has no LIMIT and is correct).
AGG_RE = re.compile(r"\b(count|sum|avg|min|max|collect)\s*\(", re.I)

# Wrong property names — catch and reject before training.
# NOTE: c.service_date was removed (now valid); p.patient_id / v.visit_date added.
WRONG_PROPS = re.compile(
    r"\b(c\.amount|c\.agent\b|c\.call_type|c\.duration|c\.successful_calls"
    r"|c\.failed_calls|c\.successful\b|iv\.amount_collected|rc\.agent\b"
    r"|p\.patient_id\b|v\.visit_date\b|l\.code\b)",
    re.I,
)