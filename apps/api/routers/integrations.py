from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.integrations import IntegrationConnectionCreateRequest, IntegrationConnectionOut
from core.services.integration_service import create_integration_connection
from db.models.integration_connection import IntegrationConnection

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.post("", response_model=IntegrationConnectionOut)
def create_connection(payload: IntegrationConnectionCreateRequest, db: Session = Depends(get_db)):
    return IntegrationConnectionOut.model_validate(create_integration_connection(db, payload))


@router.get("", response_model=list[IntegrationConnectionOut])
def list_connections(db: Session = Depends(get_db)):
    rows = db.query(IntegrationConnection).order_by(IntegrationConnection.created_at.desc()).all()
    return [IntegrationConnectionOut.model_validate(row) for row in rows]


@router.get("/{connection_id}", response_model=IntegrationConnectionOut)
def get_connection(connection_id: str, db: Session = Depends(get_db)):
    row = db.get(IntegrationConnection, connection_id)
    if not row:
        raise HTTPException(status_code=404, detail="Integration connection not found.")
    return IntegrationConnectionOut.model_validate(row)
