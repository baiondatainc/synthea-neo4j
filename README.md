# Synthea → Neo4j Aura Knowledge Graph QA

A full-stack healthcare knowledge graph demo:
- **Synthea** synthetic patient data (CSVs)
- **Neo4j Aura** (cloud graph database)
- **LangChain** RAG — natural language → Cypher → answer
- **FastAPI + WebSocket** for streaming responses
- **Vanilla JS frontend** chat UI

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Neo4j Aura free account: https://neo4j.com/cloud/platform/aura-graph-database/
- Anthropic API key (or Ollama running locally)

---

## 1. Setup

```bash
# Clone / enter project
cd synthea-neo4j

# Install all dependencies with uv
uv sync

# Copy and fill in your credentials
cp .env.example .env
nano .env
```

### .env (fill these in)
```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-aura-password
ANTHROPIC_API_KEY=your-key
LLM_PROVIDER=anthropic
```

---

## 2. Get Synthea Data

**Option A — Download pre-generated (fastest for demo):**
```bash
mkdir -p data/synthea
cd data/synthea

# 1000-patient sample from Synthea GitHub
wget https://github.com/synthetichealth/synthea/raw/master/src/test/resources/generic/observations.csv
# Or generate your own:
```

**Option B — Generate with Synthea:**
```bash
# Requires Java 11+
wget https://github.com/synthetichealth/synthea/releases/latest/download/synthea-with-dependencies.jar
java -jar synthea-with-dependencies.jar -p 1000 --exporter.csv.export=true
cp output/csv/*.csv data/synthea/
```

Required CSV files: `patients.csv`, `encounters.csv`, `conditions.csv`, `medications.csv`
Optional: `procedures.csv`, `observations.csv`, `providers.csv`, `organizations.csv`

---

## 3. Create Schema + Ingest Data

```bash
# Create constraints and indexes
uv run main.py schema

# Ingest all CSV files
uv run main.py ingest

# Check counts
uv run main.py stats
```

---

## 4. Start the Server

```bash
uv run main.py serve
# → API running at http://localhost:8000
# → WebSocket at  ws://localhost:8000/ws/qa
```

---

## 5. Open the Frontend

Open `frontend/index.html` in your browser (double-click or serve it):
```bash
python3 -m http.server 3000 --directory frontend
# → http://localhost:3000
```

---

## 6. Test via CLI

```bash
# One-off question with streaming output
uv run main.py ask "Which patients have diabetes?"
uv run main.py ask "What are the most common conditions?"
uv run main.py ask "Show me patients with both hypertension and diabetes"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + Neo4j status |
| GET | `/stats` | Node/relationship counts |
| GET | `/sample-questions` | Suggested questions |
| POST | `/ingest?drop_first=false` | Trigger ingestion |
| WS | `/ws/qa` | Streaming QA WebSocket |

### WebSocket Protocol
```json
// Send:
{"question": "Which patients have diabetes?"}

// Receive (in order):
{"type": "thinking", "data": "Generating Cypher query..."}
{"type": "cypher",   "data": "MATCH (p:Patient)-[:HAS_CONDITION]->..."}
{"type": "token",    "data": "Based on "}
{"type": "token",    "data": "the graph data..."}
{"type": "end",      "data": ""}
```

---

## Graph Schema

```
(Patient)-[:HAS_ENCOUNTER]->(Encounter)-[:PERFORMED_BY]->(Provider)-[:BELONGS_TO]->(Organization)
(Patient)-[:HAS_CONDITION]->(Condition)-[:DIAGNOSED_IN]->(Encounter)
(Patient)-[:PRESCRIBED]->(Medication)-[:PRESCRIBED_IN]->(Encounter)
(Patient)-[:HAD_PROCEDURE]->(Procedure)-[:PERFORMED_IN]->(Encounter)
(Patient)-[:HAS_OBSERVATION]->(Observation)-[:RECORDED_IN]->(Encounter)
```

---

## Using Ollama Instead of Anthropic

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

You already have `llama3.2-vision` and `qwen3:30b` — either works great.

---

## Project Structure

```
synthea-neo4j/
├── pyproject.toml        # uv/pip dependencies
├── .env.example
├── config.py             # pydantic settings
├── main.py               # CLI entry point
├── ingest/
│   ├── schema.py         # Neo4j constraints + indexes
│   └── ingestion.py      # CSV → Neo4j MERGE pipeline
├── graph/
│   ├── connection.py     # Neo4j driver singleton
│   └── schema_text.py    # Graph schema for LLM prompt
├── qa/
│   ├── llm.py            # LLM factory (Anthropic/OpenAI/Ollama)
│   └── chain.py          # LangChain GraphCypherQAChain + streaming
├── api/
│   └── websocket_server.py  # FastAPI app
├── frontend/
│   └── index.html        # Chat UI
└── data/
    └── synthea/          # Put CSV files here
```
