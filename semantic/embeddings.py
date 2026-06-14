"""
Sentence-transformer embeddings + Neo4j vector index.

Local-only: model is `sentence-transformers/all-MiniLM-L6-v2` (384 dims, 80MB).
No cloud calls. The model is downloaded once at first use and cached at
~/.cache/torch/sentence_transformers (or HF_HOME).

Three jobs live here:
  1. `get_embedder()` — lazy-load and cache the model in-process
  2. `embed(text)` / `embed_many(texts)` — single + batch
  3. `vectorize_patients(...)` — offline pass that writes embeddings
     to every Patient node + creates the vector index

The runtime retriever (`qa.hybrid_retriever`) reuses (1) and (2) at request time.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

from config import get_settings
from graph.connection import Neo4jConnection
from metadata.catalog import get_catalog

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
PATIENT_INDEX = "patient_embedding"


# ── Model loader ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder():
    """Lazy-loads the sentence-transformer model. ~80MB download on first call."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model {MODEL_NAME} (one-time, ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Embedding model ready (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed(text: str) -> list[float]:
    """Single-text embedding. Returns a Python list (Neo4j-friendly)."""
    if not text:
        return [0.0] * EMBED_DIM
    vec = get_embedder().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_many(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Batched embedding for offline use."""
    if not texts:
        return []
    vecs = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


# ── Patient profile builder ──────────────────────────────────────────────────

def build_patient_profile(row: dict) -> str:
    """Render a Patient node's attributes into a natural-language string the
    embedder can semantically index. Values from the data dictionary are
    swapped in (e.g. payor_cohort=sapa -> "Self-pay") so the embedding sees
    human terms, not codes.
    """
    catalog = get_catalog()
    parts = []

    # Demographics + geography
    state = row.get("state")
    city = row.get("city")
    zip_ = row.get("zip")
    if state or city:
        loc = ", ".join(p for p in [city, state] if p)
        parts.append(f"located in {loc}")
    if zip_:
        parts.append(f"zip {zip_}")

    # Payor + propensity
    payor = catalog.translate_value("Patient", "payor_cohort", row.get("payor_cohort"))
    if payor:
        parts.append(f"payor cohort: {payor}")
    tier = catalog.translate_value("Patient", "call_tier", row.get("call_tier"))
    if tier:
        parts.append(f"call tier: {tier}")
    grade = row.get("propensity_grade")
    if grade:
        parts.append(f"propensity grade {grade}")

    # Insurance
    if row.get("carrier_name"):
        parts.append(f"carrier {row['carrier_name']}")
    if row.get("plan_type"):
        parts.append(f"plan type {row['plan_type']}")

    # Behavioral flags (only the True ones)
    flag_labels = {
        "is_self_pay": "self-pay",
        "is_bai": "BCBS affiliated",
        "is_catastrophe": "catastrophe case",
        "is_friction": "friction case",
        "is_clean": "clean account",
        "is_fully_covered": "fully covered",
        "multi_practice_flag": "multi-practice",
        "has_any_calls": "previously contacted",
        "has_insurance": "insured",
    }
    flags_on = [label for key, label in flag_labels.items() if row.get(key)]
    if flags_on:
        parts.append(f"profile: {', '.join(flags_on)}")

    # Financial bucket — coarse, not exact
    bal = row.get("outstanding_balance") or 0
    try:
        bal = float(bal)
    except (TypeError, ValueError):
        bal = 0
    if bal == 0:
        parts.append("balance: zero")
    elif bal < 200:
        parts.append("balance: low")
    elif bal < 2000:
        parts.append("balance: medium")
    elif bal < 10000:
        parts.append("balance: high")
    else:
        parts.append("balance: very high")

    bd = row.get("adj_bad_debt") or 0
    try:
        bd = float(bd)
    except (TypeError, ValueError):
        bd = 0
    if bd > 0:
        parts.append("has bad debt")

    return ". ".join(parts) if parts else "patient with no enriched profile"


# ── Vector index management ──────────────────────────────────────────────────

def ensure_vector_index() -> None:
    """Create the Patient vector index if it doesn't exist. Idempotent.

    Requires Neo4j 5.11+ — confirmed by the 5.26 image in compose.
    Uses cosine similarity (matches normalize_embeddings=True at encode time).
    """
    cypher = f"""
        CREATE VECTOR INDEX {PATIENT_INDEX} IF NOT EXISTS
        FOR (p:Patient) ON (p.embedding)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBED_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
    """
    Neo4jConnection.run_query(cypher)
    logger.info(f"Ensured vector index {PATIENT_INDEX} (dim={EMBED_DIM}, cosine)")


# ── Offline vectorization job ────────────────────────────────────────────────

PATIENT_FETCH_CYPHER = """
MATCH (p:Patient)
WHERE p.embedding IS NULL OR $force = true
RETURN p.`patientId:ID(Patient)` AS id_v1,
       p.patientId               AS id_v2,
       p.state                   AS state,
       p.city                    AS city,
       p.zip                     AS zip,
       p.payor_cohort            AS payor_cohort,
       p.call_tier               AS call_tier,
       p.propensity_grade        AS propensity_grade,
       p.carrier_name            AS carrier_name,
       p.plan_type               AS plan_type,
       p.is_self_pay             AS is_self_pay,
       p.is_bai                  AS is_bai,
       p.is_catastrophe          AS is_catastrophe,
       p.is_friction             AS is_friction,
       p.is_clean                AS is_clean,
       p.is_fully_covered        AS is_fully_covered,
       p.multi_practice_flag     AS multi_practice_flag,
       p.has_any_calls           AS has_any_calls,
       p.has_insurance           AS has_insurance,
       p.outstanding_balance     AS outstanding_balance,
       p.adj_bad_debt            AS adj_bad_debt
LIMIT $limit
"""

UPSERT_EMBED_CYPHER = """
UNWIND $rows AS row
MATCH (p:Patient)
WHERE p.patientId = row.id OR p.`patientId:ID(Patient)` = row.id
CALL db.create.setNodeVectorProperty(p, 'embedding', row.embedding)
"""


def _patient_id(row: dict) -> str | None:
    return row.get("id_v2") or row.get("id_v1")


def vectorize_patients(
    batch_size: int = 256,
    limit: int = 1_000_000,
    force: bool = False,
) -> dict:
    """One-shot offline pass: fetch patients without embeddings, build
    profile strings, embed in batches, write back to Neo4j.
    """
    ensure_vector_index()

    fetched = Neo4jConnection.run_query(
        PATIENT_FETCH_CYPHER,
        {"force": force, "limit": limit},
    )
    if not fetched:
        logger.info("vectorize_patients: nothing to do (all patients have embeddings)")
        return {"patients": 0, "embedded": 0}

    logger.info(f"vectorize_patients: {len(fetched):,} patients to embed")
    total = 0
    for i in range(0, len(fetched), batch_size):
        chunk = fetched[i:i + batch_size]
        profiles = [build_patient_profile(r) for r in chunk]
        vectors = embed_many(profiles, batch_size=batch_size)
        rows = []
        for r, v in zip(chunk, vectors):
            pid = _patient_id(r)
            if not pid:
                continue
            rows.append({"id": pid, "embedding": v})
        Neo4jConnection.run_query(UPSERT_EMBED_CYPHER, {"rows": rows})
        total += len(rows)
        logger.info(f"  embedded {total:,} / {len(fetched):,}")

    logger.info(f"vectorize_patients: done. embedded={total:,}")
    return {"patients": len(fetched), "embedded": total}
