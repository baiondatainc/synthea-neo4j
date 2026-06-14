FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md main.py config.py ./
COPY api ./api
COPY graph ./graph
COPY ingest ./ingest
COPY qa ./qa

RUN uv pip install --system --no-cache .

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8001 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001

CMD ["python", "main.py", "serve"]
