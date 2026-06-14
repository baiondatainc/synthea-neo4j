"""
Focus entity tracker. Persists the entity IDs the conversation is currently
"on" so elliptical follow-ups ("those patients", "narrow that down") can be
rewritten into standalone questions.

Stored under `focus:{conversation_id}` as a JSON list, TTL = session_ttl.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Iterable

from config import get_settings

logger = logging.getLogger(__name__)

# Result-column names known to carry node identifiers. Order matters: prefer
# fully qualified composite IDs over bare counts/strings.
ID_KEYS = [
    "patientId", "patient_id", "patientid",
    "practiceId", "practice_id", "code",
    "locationId", "location_id", "location_node_id",
    "visitId", "visit_id",
    "chargeId", "charge_id",
    "transactionId", "transaction_id",
    "insuranceId", "insurance_id",
    "statementId", "statement_id",
    "campaignId", "campaign_name",
    "diagnosisId", "diagnosis_code",
    "procedureId", "procedure_code",
]

MAX_FOCUS_IDS = 50


def extract_entity_ids(rows: list[dict]) -> list[str]:
    """Pull a usable set of focus IDs from Neo4j result rows.

    Strategy: look for a column whose key matches a known ID-bearing name.
    Return the first matching column's values (deduped, capped at MAX_FOCUS_IDS).
    """
    if not rows:
        return []

    keys = list(rows[0].keys())
    chosen_key: str | None = None

    for candidate in ID_KEYS:
        for k in keys:
            if k.lower() == candidate.lower():
                chosen_key = k
                break
        if chosen_key:
            break

    if not chosen_key:
        return []

    seen: set[str] = set()
    ids: list[str] = []
    for r in rows:
        v = r.get(chosen_key)
        if v is None:
            continue
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        ids.append(s)
        if len(ids) >= MAX_FOCUS_IDS:
            break
    return ids


class FocusStore:
    def __init__(self, redis_url: str, ttl: int):
        self._ttl = ttl
        self._fallback: dict[str, list[str]] = {}
        self._client = None
        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info("FocusStore: connected to Redis")
        except Exception as e:
            logger.warning(f"FocusStore: Redis unavailable ({e}); using in-process fallback")
            self._client = None

    def _key(self, conv_id: str) -> str:
        return f"focus:{conv_id}"

    def get(self, conv_id: str) -> list[str]:
        if self._client is None:
            return list(self._fallback.get(conv_id, []))
        try:
            raw = self._client.get(self._key(conv_id))
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning(f"FocusStore.get failed: {e}")
            return list(self._fallback.get(conv_id, []))

    def set(self, conv_id: str, ids: Iterable[str]) -> None:
        payload = list(dict.fromkeys(ids))[:MAX_FOCUS_IDS]
        if self._client is None:
            self._fallback[conv_id] = payload
            return
        try:
            self._client.set(self._key(conv_id), json.dumps(payload), ex=self._ttl)
        except Exception as e:
            logger.warning(f"FocusStore.set failed: {e}")
            self._fallback[conv_id] = payload

    def clear(self, conv_id: str) -> None:
        if self._client is None:
            self._fallback.pop(conv_id, None)
            return
        try:
            self._client.delete(self._key(conv_id))
        except Exception as e:
            logger.warning(f"FocusStore.clear failed: {e}")


@lru_cache(maxsize=1)
def get_focus_store() -> FocusStore:
    settings = get_settings()
    return FocusStore(redis_url=settings.redis_url, ttl=settings.session_ttl_seconds)
