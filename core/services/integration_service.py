from uuid import uuid4

from db.models.integration import Integration
from db.models.integration_connection import IntegrationConnection


def create_integration_connection(db, payload) -> IntegrationConnection:
    row = IntegrationConnection(
        id=f"conn_{uuid4().hex[:12]}",
        provider=payload.provider,
        name=payload.name,
        base_url=payload.base_url,
        auth_token=payload.auth_token,
        config=payload.config,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_integration(db, payload) -> Integration:
    row = Integration(
        id=f"int_{uuid4().hex[:12]}",
        provider=payload.provider,
        name=payload.name,
        base_url=payload.base_url,
        auth_token=payload.auth_token,
        config=payload.config,
        status="connected",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
