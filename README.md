# Project Risk Intelligence

Local-first engine for **project risk detection, historical memory, forecasting, simulation, and experimental world-state modeling**.

Note: integration tokens are currently stored in plaintext in the database. This is acceptable only for local/dev testing in the current V0.

## Product map
- `V1`: Project risk dashboard
  Unstructured data -> knowledge discovery
- `V2`: Graph memory + hybrid retrieval
  Graph RAG / Hybrid RAG
- `V3`: Historical project memory + delay prediction
  Graph history + prediction
- `V4`: Scenario simulation
  Graph-based simulation / agentic workflows
- `V5`: Experimental world state
  World-model scaffold, not a trained JEPA-like model

## Included
- FastAPI API
- Redis-backed worker
- Postgres persistence via SQLAlchemy
- Qdrant vector store wrapper
- Ollama local model adapter
- Explicit risk pipeline orchestration
- Evidence-backed findings
- Job -> Run -> Snapshot -> Findings -> Report flow
- GitHub and Jira telemetry collection
- Knowledge graph memory
- Hybrid knowledge search
- Delay forecasting
- Scenario simulation
- Experimental world-state generation
- Docker Compose for local deployment

## Quick start
```bash
cp .env.example .env
make dev-up
make migrate
make seed
```

API docs: `http://localhost:8000/docs`
Dashboard: `http://localhost:8000/dashboard`

## What is implemented
### V1
1. Create a job with uploaded files
2. Worker creates a run
3. Pipeline parses documents
4. Extractor identifies actions, owners, dates, blockers, dependencies and risk mentions
5. Chunks are embedded and indexed in Qdrant
6. Risk engine computes evidence-backed findings and a project risk score
7. A project snapshot is persisted for historical tracking
8. Report is persisted as markdown artifact
9. API exposes run detail, findings, snapshots, and final report

### V2
1. Create an integration connection for GitHub or Jira
2. Trigger telemetry collection for a repo (`owner/repo`) or Jira project key (`ENG`)
3. Persist a telemetry snapshot with normalized metrics and a risk score
4. Ingest telemetry and document evidence into a knowledge graph
5. Query graph memory, telemetry, and document snapshots through hybrid search

### V3
- Store telemetry snapshots over time
- Generate a delay probability forecast from recent risk history

### V4
- Run a scenario on the latest project baseline
- Test changes like more contributors, fewer stale PRs, fewer blocked tickets
- Return simulated risk score and recommendations

### V5
- Build an experimental world state from graph density and recent telemetry
- This is a research scaffold only
- It is not a trained JEPA-like model and should not be presented as such

## Dashboard
The dashboard is a lightweight UI served directly by FastAPI. It lets you:
- inspect the latest telemetry and document risk snapshots
- create GitHub or Jira integrations
- trigger one telemetry collection manually
- search knowledge across graph memory
- generate delay forecasts
- run simulations
- build experimental world states
- review recent pipeline runs

After `make seed`, the dashboard includes demo data for telemetry, forecasting, simulation, and world state so the UI is not empty.

## API surfaces
- `/files`, `/jobs`, `/runs`
- `/integrations`, `/telemetry`
- `/knowledge/search`
- `/knowledge/forecast/{project_ref}`
- `/knowledge/simulate`
- `/knowledge/world-state/{project_ref}`
- `/dashboard`, `/dashboard/overview`

## How to test
### 1. Boot the stack
```bash
cp .env.example .env
make dev-up
make migrate
make seed
```

### 2. Open the product
- Dashboard: `http://localhost:8000/dashboard`
- Swagger: `http://localhost:8000/docs`

### 3. Verify the seeded demo
When the dashboard opens, you should already see:
- one demo telemetry project: `demo/project-risk`
- one forecast
- one simulation
- one experimental world state

If the screen is empty, inspect logs:
```bash
make logs
```

### 4. Test V1 manually from the dashboard
1. Open `/dashboard`
2. Create a GitHub integration
3. Collect telemetry for a real repo like `owner/repo`
4. Verify a new active project card appears
5. Run a forecast for the same project
6. Run a simulation with adjustments like:
   `{"active_contributors_30d": 2, "stale_pull_requests": -2, "commits_last_7d": 5}`
7. Build the world state

### 5. Test the document pipeline
Upload a project status file through Swagger or curl:
```bash
curl -X POST http://localhost:8000/files \
  -F "file=@storage/raw/demo_project_status.txt"
```

Create a job with the returned document id:
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": "project_risk",
    "objective": "Detect delivery risk and ownership gaps",
    "document_ids": ["doc_xxx"]
  }'
```

Then reload the dashboard and check:
- recent runs
- document snapshots
- report artifact in the run detail

### 6. Test the API directly
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

Create a delay forecast:
```bash
curl -X POST http://localhost:8000/knowledge/forecast/owner%2Frepo
```

Run a simulation:
```bash
curl -X POST http://localhost:8000/knowledge/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "project_ref": "owner/repo",
    "scenario_name": "Add contributors",
    "adjustments": {
      "active_contributors_30d": 2,
      "stale_pull_requests": -2,
      "commits_last_7d": 5
    }
  }'
```

Build a world state:
```bash
curl -X POST http://localhost:8000/knowledge/world-state/owner%2Frepo
```

Search knowledge:
```bash
curl -X POST http://localhost:8000/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "stale pull requests blocked delivery",
    "project_ref": "owner/repo",
    "limit": 10
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

## Truth in advertising
- V1 through V4 are pragmatic product features.
- V5 is only an experimental scaffold.
- There is no trained world model and no JEPA-like training pipeline in this repo today.
