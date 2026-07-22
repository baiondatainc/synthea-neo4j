FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV UV_LINK_MODE=copy

# 1) Dependencies only — cached until pyproject.toml / uv.lock change
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2) Application code
COPY main.py config.py ./
COPY api ./api
COPY graph ./graph
COPY ingest ./ingest
COPY qa ./qa
COPY metadata ./metadata
COPY guardrails ./guardrails
COPY memory ./memory
COPY cache ./cache
COPY semantic ./semantic

# 3) Install the project itself
RUN uv sync --frozen --no-dev --no-cache

ENV PATH="/app/.venv/bin:$PATH" \
    APP_HOST=0.0.0.0 \
    APP_PORT=8001 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

CMD ["python", "main.py", "serve"]