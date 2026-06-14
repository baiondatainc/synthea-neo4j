"""
Answer cache.

Stores the full answer bundle for a (rewritten question, schema version)
pair so that repeat questions skip the LLM and Neo4j entirely.

Bundle shape:
  {
    "question": str,        # the (rewritten) question this entry answers
    "cypher":   str,        # the executed Cypher
    "results":  list[dict], # already-redacted result rows
    "answer":   str,        # the qa_llm answer text (already redacted)
  }

Keyed by `cache:answer:{sha256(question|schema_version)}`. TTL from settings.
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache

from config import get_settings

logger = logging.getLogger(__name__)


def make_cache_key(question: str, schema_version: str) -> str:
    """Stable cache key — normalises whitespace + case so cosmetic differences
    in the same question still hit."""
    norm = " ".join(question.lower().split())
    h = hashlib.sha256(f"{norm}|{schema_version}".encode()).hexdigest()
    return f"cache:answer:{h[:32]}"


class AnswerCache:
    def __init__(self, redis_url: str, ttl: int):
        self._ttl = ttl
        self._client = None
        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info("AnswerCache: connected to Redis")
        except Exception as e:
            logger.warning(f"AnswerCache: Redis unavailable ({e}); cache disabled")
            self._client = None

    def get(self, key: str) -> dict | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"AnswerCache.get failed: {e}")
            return None

    def set(self, key: str, bundle: dict) -> None:
        if self._client is None:
            return
        try:
            self._client.set(key, json.dumps(bundle, default=str), ex=self._ttl)
        except Exception as e:
            logger.warning(f"AnswerCache.set failed: {e}")

    def clear_all(self) -> int:
        """Wipe every answer-cache key. Returns count deleted. Use sparingly."""
        if self._client is None:
            return 0
        deleted = 0
        try:
            for k in self._client.scan_iter("cache:answer:*"):
                self._client.delete(k)
                deleted += 1
        except Exception as e:
            logger.warning(f"AnswerCache.clear_all failed: {e}")
        return deleted


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCache:
    settings = get_settings()
    return AnswerCache(redis_url=settings.redis_url, ttl=settings.cache_ttl_seconds)
