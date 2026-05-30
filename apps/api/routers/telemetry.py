from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.integrations import TelemetryCollectRequest, TelemetrySnapshotOut
from core.services.telemetry_service import collect_telemetry_snapshot
from db.models.integration_connection import IntegrationConnection
from db.models.telemetry_snapshot import TelemetrySnapshot

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.post("/collect", response_model=TelemetrySnapshotOut)
def collect_snapshot(payload: TelemetryCollectRequest, db: Session = Depends(get_db)):
    connection = db.get(IntegrationConnection, payload.connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Integration connection not found.")
    snapshot = collect_telemetry_snapshot(db, connection, payload.project_ref)
    return TelemetrySnapshotOut.model_validate(snapshot)


@router.get("/snapshots", response_model=list[TelemetrySnapshotOut])
def list_snapshots(connection_id: str | None = None, project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(TelemetrySnapshot).order_by(TelemetrySnapshot.created_at.desc())
    if connection_id:
        query = query.filter(TelemetrySnapshot.connection_id == connection_id)
    if project_ref:
        query = query.filter(TelemetrySnapshot.project_ref == project_ref)
    rows = query.limit(100).all()
    return [TelemetrySnapshotOut.model_validate(row) for row in rows]
