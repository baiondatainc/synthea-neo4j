"""
Redis-backed conversation history.

Wraps `langchain_redis.RedisChatMessageHistory` with a TTL refresh and a
transcript-formatting helper. Keyed by `chat:{conversation_id}`.

Falls back to an in-process dict store if Redis is unreachable so the agent
still serves requests (without persistent memory) during outages.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import get_settings

logger = logging.getLogger(__name__)


class _InMemoryFallback:
    """Last-resort store when Redis is unreachable. Process-local, no TTL."""

    def __init__(self):
        self._store: dict[str, list[BaseMessage]] = {}

    def get(self, conv_id: str) -> List[BaseMessage]:
        return list(self._store.get(conv_id, []))

    def append(self, conv_id: str, msg: BaseMessage) -> None:
        self._store.setdefault(conv_id, []).append(msg)

    def clear(self, conv_id: str) -> None:
        self._store.pop(conv_id, None)


class SessionStore:
    """Per-conversation message history."""

    def __init__(self, redis_url: str, ttl: int):
        self._redis_url = redis_url
        self._ttl = ttl
        self._fallback = _InMemoryFallback()
        self._using_fallback = False

        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            # langchain_community's version uses plain Redis LIST commands —
            # no RediSearch / FT.* required, unlike langchain_redis.
            from langchain_community.chat_message_histories import RedisChatMessageHistory
            self._history_cls = RedisChatMessageHistory
            logger.info(f"SessionStore: connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"SessionStore: Redis unavailable ({e}); using in-process fallback")
            self._client = None
            self._history_cls = None
            self._using_fallback = True

    # ── basic ops ────────────────────────────────────────────────────────

    def _history(self, conv_id: str):
        return self._history_cls(
            session_id=conv_id,
            url=self._redis_url,
            ttl=self._ttl,
            key_prefix="chat:",
        )

    def get_messages(self, conv_id: str) -> List[BaseMessage]:
        if self._using_fallback:
            return self._fallback.get(conv_id)
        try:
            return list(self._history(conv_id).messages)
        except Exception as e:
            logger.warning(f"SessionStore.get_messages failed: {e}; using fallback")
            return self._fallback.get(conv_id)

    def append_user(self, conv_id: str, text: str) -> None:
        self._append(conv_id, HumanMessage(content=text))

    def append_assistant(self, conv_id: str, text: str) -> None:
        self._append(conv_id, AIMessage(content=text))

    def _append(self, conv_id: str, msg: BaseMessage) -> None:
        if self._using_fallback:
            self._fallback.append(conv_id, msg)
            return
        try:
            self._history(conv_id).add_message(msg)
        except Exception as e:
            logger.warning(f"SessionStore.append failed: {e}; using fallback")
            self._fallback.append(conv_id, msg)

    def clear(self, conv_id: str) -> None:
        if self._using_fallback:
            self._fallback.clear(conv_id)
            return
        try:
            self._history(conv_id).clear()
        except Exception as e:
            logger.warning(f"SessionStore.clear failed: {e}")

    # ── transcript helper ────────────────────────────────────────────────

    def transcript(self, conv_id: str, max_turns: int = 6) -> str:
        """Recent transcript formatted for prompt injection. Empty string
        when there's no prior turns."""
        msgs = self.get_messages(conv_id)
        if not msgs:
            return ""
        recent = msgs[-(max_turns * 2):]  # one turn = user + assistant
        lines = []
        for m in recent:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            text = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{role}: {text}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    settings = get_settings()
    return SessionStore(redis_url=settings.redis_url, ttl=settings.session_ttl_seconds)
