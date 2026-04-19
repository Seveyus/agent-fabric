# Agent Fabric V1 Hardened Skeleton

Local-first V1 for a **Project Audit / Risk Detection Engine**.

## Included
- FastAPI API
- Redis-backed worker
- Postgres persistence via SQLAlchemy
- Qdrant vector store wrapper
- Ollama local model adapter
- Explicit pipeline orchestration
- Evidence-backed findings
- Job -> Run -> Findings -> Report flow
- Docker Compose for local deployment

## Quick start
```bash
cp .env.example .env
make dev-up
make migrate
make seed
```

API docs: `http://localhost:8000/docs`

## Main flow
1. Create a job with uploaded files
2. Worker creates a run
3. Pipeline parses documents
4. Extractor identifies tasks, owners, dates
5. Chunks are embedded and indexed in Qdrant
6. Risk analyst generates findings with evidence references
7. Report is persisted as markdown artifact
8. API exposes run detail, findings, and final report
```
