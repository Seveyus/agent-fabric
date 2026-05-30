from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.integrations import MetricSnapshotOut, TelemetryCollectRequest, UnifiedProjectRiskScoreOut
from core.services.connector_sync_service import run_connector_sync
from core.services.project_risk_score_service import compute_project_risk_score
from db.models.connector_sync_run import ConnectorSyncRun
from db.models.integration import Integration
from db.models.metric_snapshot import MetricSnapshot

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("/collect", response_model=UnifiedProjectRiskScoreOut)
def collect_snapshot(payload: TelemetryCollectRequest, db: Session = Depends(get_db)):
    integration = db.get(Integration, payload.integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found.")
    try:
        run_connector_sync(db, integration, payload.project_ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UnifiedProjectRiskScoreOut(**compute_project_risk_score(db, payload.project_ref))


@router.get("/snapshots", response_model=list[MetricSnapshotOut])
def list_snapshots(integration_id: str | None = None, project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(MetricSnapshot).order_by(MetricSnapshot.captured_at.desc())
    if integration_id:
        query = query.join(ConnectorSyncRun, ConnectorSyncRun.id == MetricSnapshot.sync_run_id).filter(
            ConnectorSyncRun.integration_id == integration_id
        )
    if project_ref:
        query = query.filter(MetricSnapshot.project_ref == project_ref)
    rows = query.limit(100).all()
    return [MetricSnapshotOut.model_validate(row) for row in rows]
