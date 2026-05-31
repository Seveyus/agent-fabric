from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.integrations import (
    BatchSyncResultOut,
    ConnectorSyncRunOut,
    IntegrationConnectionCreateRequest,
    IntegrationConnectionOut,
    RecentProjectCandidateOut,
    TelemetryCollectRequest,
)
from core.services.connector_sync_service import discover_recent_projects, run_connector_sync, sync_recent_projects
from core.services.integration_service import create_integration
from db.models.connector_sync_run import ConnectorSyncRun
from db.models.integration import Integration

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.post("", response_model=IntegrationConnectionOut)
def create_connection(payload: IntegrationConnectionCreateRequest, db: Session = Depends(get_db)):
    return IntegrationConnectionOut.model_validate(create_integration(db, payload))


@router.get("", response_model=list[IntegrationConnectionOut])
def list_connections(db: Session = Depends(get_db)):
    rows = db.query(Integration).order_by(Integration.created_at.desc()).all()
    return [IntegrationConnectionOut.model_validate(row) for row in rows]


@router.post("/sync", response_model=ConnectorSyncRunOut)
def sync_integration(payload: TelemetryCollectRequest, db: Session = Depends(get_db)):
    row = db.get(Integration, payload.integration_id)
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found.")
    try:
        sync_run = run_connector_sync(db, row, payload.project_ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConnectorSyncRunOut.model_validate(sync_run)


@router.get("/sync-runs", response_model=list[ConnectorSyncRunOut])
def list_sync_runs(project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ConnectorSyncRun).order_by(ConnectorSyncRun.started_at.desc())
    if project_ref:
        query = query.filter(ConnectorSyncRun.project_ref == project_ref)
    rows = query.limit(100).all()
    return [ConnectorSyncRunOut.model_validate(row) for row in rows]


@router.get("/{integration_id}/recent-projects", response_model=list[RecentProjectCandidateOut])
def get_recent_projects(integration_id: str, limit: int = 8, db: Session = Depends(get_db)):
    row = db.get(Integration, integration_id)
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found.")
    try:
        projects = discover_recent_projects(row, limit=max(1, min(limit, 20)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not discover recent projects: {exc}") from exc
    return [RecentProjectCandidateOut.model_validate(item) for item in projects]


@router.post("/{integration_id}/sync-recent", response_model=BatchSyncResultOut)
def sync_recent(integration_id: str, limit: int = 8, db: Session = Depends(get_db)):
    row = db.get(Integration, integration_id)
    if not row:
        raise HTTPException(status_code=404, detail="Integration not found.")
    try:
        result = sync_recent_projects(db, row, limit=max(1, min(limit, 20)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not sync recent projects: {exc}") from exc
    return BatchSyncResultOut(
        integration_id=result["integration_id"],
        provider=result["provider"],
        discovered_count=result["discovered_count"],
        synced_count=result["synced_count"],
        failed_count=result["failed_count"],
        projects=[RecentProjectCandidateOut.model_validate(item) for item in result["projects"]],
        sync_runs=[ConnectorSyncRunOut.model_validate(item) for item in result["sync_runs"]],
    )


@router.get("/{connection_id}", response_model=IntegrationConnectionOut)
def get_connection(connection_id: str, db: Session = Depends(get_db)):
    row = db.get(Integration, connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Integration connection not found.")
    return IntegrationConnectionOut.model_validate(row)
