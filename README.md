# Project Risk Intelligence V0

Local-first V0 for a **Project Risk Intelligence Engine**.

## Included
- FastAPI API
- Redis-backed worker
- Postgres persistence via SQLAlchemy
- Qdrant vector store wrapper
- Ollama local model adapter
- Explicit risk pipeline orchestration
- Evidence-backed findings
- Job -> Run -> Snapshot -> Findings -> Report flow
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
4. Extractor identifies actions, owners, dates, blockers, dependencies and risk mentions
5. Chunks are embedded and indexed in Qdrant
6. Risk engine computes evidence-backed findings and a project risk score
7. A project snapshot is persisted for historical tracking
8. Report is persisted as markdown artifact
9. API exposes run detail, findings, snapshots, and final report
```
