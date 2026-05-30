# Project Risk Intelligence V0

Local-first V0 for a **Project Risk Intelligence Engine**.

Note: integration tokens are currently stored in plaintext in the database. This is acceptable only for local/dev testing in the current V0.

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

## Telemetry flow
1. Create an integration connection for GitHub or Jira
2. Trigger telemetry collection for a repo (`owner/repo`) or Jira project key (`ENG`)
3. Persist a telemetry snapshot with normalized metrics and a risk score
4. Read snapshots back through the API and compare them over time

## Example API usage
Create a GitHub connection:
```bash
curl -X POST http://localhost:8000/integrations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "github",
    "name": "GitHub Prod",
    "base_url": "https://api.github.com",
    "auth_token": "ghp_xxx",
    "config": {}
  }'
```

Collect GitHub telemetry:
```bash
curl -X POST http://localhost:8000/telemetry/collect \
  -H "Content-Type: application/json" \
  -d '{
    "connection_id": "conn_123",
    "project_ref": "owner/repo"
  }'
```

Collect Jira telemetry:
```bash
curl -X POST http://localhost:8000/integrations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "jira",
    "name": "Jira Prod",
    "base_url": "https://your-company.atlassian.net",
    "auth_token": "jira_api_token_or_bearer",
    "config": {}
  }'

curl -X POST http://localhost:8000/telemetry/collect \
  -H "Content-Type: application/json" \
  -d '{
    "connection_id": "conn_456",
    "project_ref": "ENG"
  }'
```
