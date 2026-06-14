from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Neo4j Aura
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")

    # LLM
    llm_provider: str = Field("anthropic", env="LLM_PROVIDER")
    llm_model: str = Field("claude-sonnet-4-20250514", env="LLM_MODEL")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2", env="OLLAMA_MODEL")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Data
    synthea_data_dir: str = Field("/home/siva/work/codebase/RP/synthea-neo4j/generator/input", env="SYNTHEA_DATA_DIR")
    batch_size: int = Field(500, env="BATCH_SIZE")

    # ── v2: Redis + memory + cache ───────────────────────────────────────
    redis_url: str = Field("redis://rp-redis:6379/0", env="REDIS_URL")
    session_ttl_seconds: int = Field(86400, env="SESSION_TTL")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL")
    memory_enabled: bool = Field(True, env="MEMORY_ENABLED")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    # Bump when graph schema or ingestion changes to invalidate the cache.
    schema_version: str = Field("v1", env="SCHEMA_VERSION")

    # ── v2: Guardrails ───────────────────────────────────────────────────
    guardrails_enabled: bool = Field(True, env="GUARDRAILS_ENABLED")
    guardrails_redact_output: bool = Field(True, env="GUARDRAILS_REDACT_OUTPUT")
    cypher_row_limit: int = Field(100, env="CYPHER_ROW_LIMIT")
    cypher_timeout_seconds: int = Field(15, env="CYPHER_TIMEOUT")

    # ── v2: Hybrid retriever (reserved for phase D) ──────────────────────
    hybrid_retriever_enabled: bool = Field(False, env="HYBRID_RETRIEVER_ENABLED")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
