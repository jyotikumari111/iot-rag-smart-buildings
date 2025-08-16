# Smart Building RAG (IoT Sensor Data + Manuals)

A deployable Retrieval-Augmented Generation (RAG) demo for smart buildings. It ingests IoT telemetry, indexes maintenance manuals/specs, detects anomalies, and answers operator queries with citations and live KPIs.

## Quickstart

```bash
# 1) Copy and edit .env
cp .env.example .env

# 2) Start core services (DB, Chroma, UI+API)
docker compose up -d --build

# 3) Initialize DB schema
docker compose exec timescaledb psql -U postgres -d smart_building -f /docker-entrypoint-initdb.d/init_db.sql

# 4) Index some documents (place sample PDFs/text in data/manuals/)
docker compose exec api python -m rag.indexer --root /app/data/manuals

# 5) Open the app (local): http://localhost:8501
```

### Notes
- **LLM**: Uses OpenAI (if key provided) or tries local Ollama (`OLLAMA_HOST`) as fallback. If neither is available, a safe stub responds.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (can be changed via env).
- **Vector DB**: Chroma server container.
- **Time-series**: TimescaleDB (Postgres).

## Project Layout
```
smart-building-rag/
├── docker-compose.yml
├── .env.example
├── README.md
├── data/
│   ├── manuals/            # Put PDFs or .txt here (mounted into API container)
│   └── samples/
├── infra/
│   └── init_db.sql
└── services/
    ├── api/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── app.py
    │   ├── rag/
    │   │   ├── __init__.py
    │   │   ├── retriever.py
    │   │   ├── indexer.py
    │   │   ├── prompts.py
    │   │   ├── utils.py
    │   │   └── eval_ragas.py
    │   └── ml/
    │       ├── __init__.py
    │       ├── features.py
    │       ├── anomalies.py
    │       └── predict.py
    ├── ui/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── Streamlit_app.py
    └── simulator/
        └── produce_telemetry.py
```
