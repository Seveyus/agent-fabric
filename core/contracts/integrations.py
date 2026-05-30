from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IntegrationConnectionCreateRequest(BaseModel):
    provider: Literal["github", "jira"]
    name: str
    base_url: str
    auth_token: str
    config: dict = Field(default_factory=dict)


class IntegrationConnectionOut(BaseModel):
    id: str
    provider: str
    name: str
    base_url: str
    config: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class TelemetryCollectRequest(BaseModel):
    connection_id: str
    project_ref: str


class TelemetrySnapshotOut(BaseModel):
    id: str
    connection_id: str
    provider: str
    project_ref: str
    risk_score: float
    risk_level: str
    metrics: dict
    evidence: dict
    created_at: datetime

    model_config = {"from_attributes": True}
