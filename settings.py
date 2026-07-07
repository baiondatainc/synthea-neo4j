"""
LoRA training settings and guardrail configuration.
Source of truth for model thresholds, row caps, and guardrail rules.
"""

# LoRA Training Configuration (Gemma 2 9B)
LORA_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_MODEL_PATH = "../text2cypher-gemma-2-9b-it-finetuned-2024v1"  # Local base if available
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention heads
    "gate_proj", "up_proj", "down_proj",      # MLP layers
]
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_USE_GRADIENT_CHECKPOINTING = True
LORA_USE_RSLORA = True  # rsLoRA scaling for better stability

# Training hyperparameters
TRAIN_BATCH_SIZE = 2  # Gemma 9B needs smaller batch on GPU
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 2e-4  # Standard for LoRA
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.05  # 5% of total steps
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 16
LR_SCHEDULER_TYPE = "cosine"

# Model generation
GENERATION_MAX_LENGTH = 512
GENERATION_TEMPERATURE = 0.0  # Deterministic at inference
GENERATION_TOP_K = 1
GENERATION_TOP_P = 1.0

# Cypher execution guardrails
CYPHER_ROW_LIMIT = 100
CYPHER_TIMEOUT_SECONDS = 15
CYPHER_EXPLAIN_TIMEOUT_SECONDS = 5

# Model paths and format
CYPHER_MODEL_TAG = "rp-cypher-gemma-v1"  # Ollama tag for production
CYPHER_MODEL_FORMAT = "gemma"  # Prompt format (gemma vs qwen ChatML)
CHECKPOINT_DIR = "./lora_checkpoints"
ADAPTER_DIR = "./lora_adapter_v1"
GGUF_EXPORT_DIR = "./gguf_export_v1"
MODELS_DIR = "./models"
MAX_SEQ_LENGTH = 2048

# Validation thresholds
GENERATION_RATE_FLOOR = 0.869  # 86.9% minimum
GUARDRAIL_PASS_RATE_FLOOR = 0.869
EXECUTABLE_RATE_FLOOR = 0.99

# Category weak floors (never regress below these)
CATEGORY_FLOORS = {
    "Contact Center Operations": 10,
    "Provider & Referral Patterns": 10,
    "Executive / KPI Dashboard": 12,
    "Trend & Temporal Analysis": 12,
    "Context Chain": 29,
}

# PII/PHI detection patterns (for redaction)
PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "mrn": r"\bMRN[:\s]+(\d+)\b",
    "dob": r"\b\d{1,2}/\d{1,2}/\d{4}\b",
}

# Jailbreak keywords / unsafe topics
UNSAFE_KEYWORDS = [
    "DROP", "DELETE", "TRUNCATE", "CREATE", "ALTER", "GRANT", "REVOKE",
    "CALL", "apoc", "dbms",  # Dangerous procedures
    "eval", "exec", "execute", "__import__",  # Code execution
]

# Safe relationship types (allowlist for schema queries)
SAFE_RELATIONSHIPS = [
    "HAS_VISIT", "VISITED_BY", "AT_PROVIDER", "AT_ORGANIZATION",
    "DIAGNOSED_WITH", "TREATED_WITH", "PRESCRIBED_BY", "UNDERWENT",
    "HAD_OBSERVATION", "WORKS_AT", "PART_OF_VISIT", "PART_OF_CAMPAIGN",
    # Add more as needed
]

# Safe node labels (allowlist)
SAFE_NODES = [
    "Patient", "Visit", "Provider", "Organization", "Condition", "Medication",
    "Procedure", "Observation", "Campaign", "DiallerCall", "PhoneBridge",
]

def invalidate_chain_cache():
    """Called when training completes to clear cached few-shot exemplars."""
    import os
    cache_dir = "cache"
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".pkl") or f.endswith(".json"):
                os.remove(os.path.join(cache_dir, f))
    print(f"✓ Cache invalidated: {cache_dir}")
