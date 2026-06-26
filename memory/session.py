"""
Redis-backed conversation history.

Wraps langchain_community RedisChatMessageHistory with a TTL refresh and a
transcript-formatting helper. Keyed by `chat:{conversation_id}`.

Falls back to an in-process dict store if Redis is unreachable so the agent
still serves requests (without persistent memory) during outages.

Changes vs previous version:
  - _append: logs full traceback (exc_info=True) so silent failures are visible
  - _append: logs which fallback path was taken
  - transcript: logs how many messages were read
  - Added explicit add_message error visibility
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
            from langchain_community.chat_message_histories import RedisChatMessageHistory
            self._history_cls = RedisChatMessageHistory

            # Smoke-test: try an actual write+read to catch schema/version issues
            # at startup rather than silently at runtime.
            _test_h = RedisChatMessageHistory(
                session_id="__healthcheck__",
                url=redis_url,
                ttl=10,
                key_prefix="chat:",
            )
            _test_h.add_message(HumanMessage(content="ping"))
            _msgs = _test_h.messages
            _test_h.clear()
            logger.info(
                f"SessionStore: connected to Redis at {redis_url} "
                f"(smoke-test write+read OK, got {len(_msgs)} msg)"
            )
        except Exception as e:
            logger.warning(
                f"SessionStore: Redis unavailable or write failed ({e}); "
                f"using in-process fallback",
                exc_info=True,
            )
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
            msgs = list(self._history(conv_id).messages)
            logger.info(f"SessionStore.get_messages: conv_id={conv_id!r} → {len(msgs)} messages")
            return msgs
        except Exception as e:
            logger.warning(
                f"SessionStore.get_messages failed: {e}; falling back to in-process",
                exc_info=True,
            )
            return self._fallback.get(conv_id)

    def append_user(self, conv_id: str, text: str) -> None:
        self._append(conv_id, HumanMessage(content=text))

    def append_assistant(self, conv_id: str, text: str) -> None:
        self._append(conv_id, AIMessage(content=text))

    def _append(self, conv_id: str, msg: BaseMessage) -> None:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if self._using_fallback:
            self._fallback.append(conv_id, msg)
            logger.info(f"SessionStore._append ({role}): wrote to IN-PROCESS fallback conv_id={conv_id!r}")
            return
        try:
            self._history(conv_id).add_message(msg)
            logger.info(f"SessionStore._append ({role}): wrote to Redis conv_id={conv_id!r}")
        except Exception as e:
            # Log with full traceback so the root cause is visible in server logs.
            logger.warning(
                f"SessionStore._append ({role}) FAILED for conv_id={conv_id!r}: {e}; "
                f"falling back to in-process store",
                exc_info=True,
            )
            self._fallback.append(conv_id, msg)

    def clear(self, conv_id: str) -> None:
        if self._using_fallback:
            self._fallback.clear(conv_id)
            return
        try:
            self._history(conv_id).clear()
        except Exception as e:
            logger.warning(f"SessionStore.clear failed: {e}", exc_info=True)

    # ── transcript helper ────────────────────────────────────────────────

    def transcript(self, conv_id: str, max_turns: int = 6) -> str:
        """Recent transcript formatted for prompt injection."""
        msgs = self.get_messages(conv_id)
        if not msgs:
            logger.info(f"SessionStore.transcript: conv_id={conv_id!r} → empty")
            return ""
        recent = msgs[-(max_turns * 2):]
        lines = []
        for m in recent:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            text = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{role}: {text}")
        transcript = "\n".join(lines)
        logger.info(
            f"SessionStore.transcript: conv_id={conv_id!r} → "
            f"{len(msgs)} msgs, {len(transcript)} chars"
        )
        return transcript


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    settings = get_settings()
    return SessionStore(redis_url=settings.redis_url, ttl=settings.session_ttl_seconds)