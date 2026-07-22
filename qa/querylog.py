"""
Postgres query logging for the KG QA pipeline.

Logs one row per request: question, rewritten question, generated Cypher,
cypher-generation duration, execution duration, row count, status, timestamp.

Design:
  - asyncpg pool, created lazily on first log call.
  - Fire-and-forget: log_query() schedules the INSERT on the event loop and
    returns immediately — a slow/down Postgres never delays the answer stream.
  - Fails soft: any DB error is logged at WARNING and swallowed.
  - Config comes from config.Settings (which loads .env), NOT os.getenv —
    pydantic-settings parses .env without exporting to os.environ, so reading
    the environment directly silently disabled logging. os.getenv is kept only
    as a fallback for contexts where Settings isn't importable.

Setup:
  pip install asyncpg
  .env:
    QUERYLOG_ENABLED=true
    QUERYLOG_DSN=postgresql://postgres:...@localhost:5432/rpdb

  And in config.Settings add:
    querylog_enabled: bool = True
    querylog_dsn: str = ""

  The table is auto-created on first use.
"""
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:  # keep the app importable without the dependency
    asyncpg = None

_POOL: "asyncpg.Pool | None" = None
_POOL_LOCK = asyncio.Lock()
_SCHEMA_READY = False

# Strong references to in-flight fire-and-forget tasks. The event loop only
# keeps weak refs to tasks, so without this an insert can be garbage-collected
# mid-write. Also lets close_pool() drain pending writes on shutdown.
_TASKS: set = set()

_DDL = """
CREATE TABLE IF NOT EXISTS query_log (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    conversation_id TEXT,
    question        TEXT NOT NULL,
    rewritten       TEXT,
    cypher          TEXT,
    cypher_gen_ms   INTEGER,
    exec_ms         INTEGER,
    total_ms        INTEGER,
    row_count       INTEGER,
    status          TEXT NOT NULL DEFAULT 'ok',   -- ok | cache_hit | blocked | error | hybrid
    detail          TEXT,                          -- block reason / error message
    answer_chars    INTEGER,
    model           TEXT
);
CREATE INDEX IF NOT EXISTS idx_query_log_ts     ON query_log (ts DESC);
CREATE INDEX IF NOT EXISTS idx_query_log_status ON query_log (status);
"""

_INSERT = """
INSERT INTO query_log
    (conversation_id, question, rewritten, cypher,
     cypher_gen_ms, exec_ms, total_ms, row_count,
     status, detail, answer_chars, model)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
"""


def _get_config() -> tuple[bool, str]:
    """Return (enabled, dsn) — Settings first, env vars as fallback."""
    try:
        from config import get_settings
        s = get_settings()
        enabled = bool(getattr(s, "querylog_enabled", False))
        dsn = getattr(s, "querylog_dsn", "") or ""
        if dsn:
            return enabled, dsn
    except Exception as e:
        logger.debug(f"querylog: Settings unavailable ({e}) — falling back to env")

    enabled = os.getenv("QUERYLOG_ENABLED", "true").lower() in ("1", "true", "yes")
    return enabled, os.getenv("QUERYLOG_DSN", "")


def _enabled() -> bool:
    if asyncpg is None:
        return False
    enabled, dsn = _get_config()
    return enabled and bool(dsn)


async def _get_pool() -> "asyncpg.Pool | None":
    global _POOL, _SCHEMA_READY
    if _POOL is not None:
        return _POOL
    async with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        _, dsn = _get_config()
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=5)
            if not _SCHEMA_READY:
                async with pool.acquire() as conn:
                    await conn.execute(_DDL)
                _SCHEMA_READY = True
            _POOL = pool
            logger.info("querylog: Postgres pool initialised")
        except Exception as e:
            logger.warning(f"querylog: could not connect to Postgres ({e}) — logging disabled for now")
            return None
    return _POOL


async def _write(row: dict) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                _INSERT,
                row.get("conversation_id"),
                row.get("question", ""),
                row.get("rewritten"),
                row.get("cypher"),
                row.get("cypher_gen_ms"),
                row.get("exec_ms"),
                row.get("total_ms"),
                row.get("row_count"),
                row.get("status", "ok"),
                (row.get("detail") or "")[:2000] or None,
                row.get("answer_chars"),
                row.get("model"),
            )
    except Exception as e:
        logger.warning(f"querylog: insert failed — {e}")


def log_query(
    *,
    question: str,
    conversation_id: str | None = None,
    rewritten: str | None = None,
    cypher: str | None = None,
    cypher_gen_ms: int | None = None,
    exec_ms: int | None = None,
    total_ms: int | None = None,
    row_count: int | None = None,
    status: str = "ok",
    detail: str | None = None,
    answer_chars: int | None = None,
    model: str | None = None,
) -> None:
    """Fire-and-forget: schedules the insert and returns immediately.

    Safe to call from any coroutine. No-ops if disabled/unconfigured.
    """
    if not _enabled():
        # One-time breadcrumb so a misconfig is visible instead of silent.
        if not getattr(log_query, "_warned", False):
            enabled, dsn = _get_config()
            logger.warning(
                f"querylog: DISABLED — asyncpg={asyncpg is not None}, "
                f"enabled={enabled}, dsn_set={bool(dsn)}. "
                "Check Settings.querylog_dsn / QUERYLOG_DSN."
            )
            log_query._warned = True
        return
    row = {
        "question": question,
        "conversation_id": conversation_id,
        "rewritten": rewritten,
        "cypher": cypher,
        "cypher_gen_ms": cypher_gen_ms,
        "exec_ms": exec_ms,
        "total_ms": total_ms,
        "row_count": row_count,
        "status": status,
        "detail": detail,
        "answer_chars": answer_chars,
        "model": model,
    }
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_write(row))
        _TASKS.add(task)
        task.add_done_callback(_TASKS.discard)
    except RuntimeError:
        # No running loop (e.g. sync CLI context) — run it synchronously.
        try:
            asyncio.run(_write(row))
        except Exception as e:
            logger.warning(f"querylog: sync write failed — {e}")


async def close_pool() -> None:
    """Call on app shutdown: drain pending writes, then close the pool."""
    global _POOL
    if _TASKS:
        await asyncio.gather(*_TASKS, return_exceptions=True)
    if _POOL is not None:
        await _POOL.close()
        _POOL = None