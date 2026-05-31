from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IntegrationConnectionCreateRequest(BaseModel):
    provider: Literal["github", "jira", "notion"]
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
    status: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class TelemetryCollectRequest(BaseModel):
    integration_id: str
    project_ref: str


class RecentProjectCandidateOut(BaseModel):
    project_ref: str
    display_name: str
    provider: str
    last_activity_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class BatchSyncResultOut(BaseModel):
    integration_id: str
    provider: str
    discovered_count: int
    synced_count: int
    failed_count: int
    projects: list[RecentProjectCandidateOut]
    sync_runs: list["ConnectorSyncRunOut"]


class ConnectorSyncRunOut(BaseModel):
    id: str
    integration_id: str
    provider: str
    project_ref: str
    status: str
    raw_count: int
    entity_count: int
    relation_count: int
    snapshot_count: int
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class MetricSnapshotOut(BaseModel):
    id: str
    sync_run_id: str
    project_ref: str
    metric_source: str
    metric_name: str
    metric_value: float
    metric_unit: str | None = None
    dimensions: dict
    captured_at: datetime

    model_config = {"from_attributes": True}


class UnifiedProjectRiskScoreOut(BaseModel):
    project_ref: str
    risk_score: float
    risk_level: str
    top_reasons: list[str]
    recommended_actions: list[str]
    evidence: list[dict]

    model_config = {"from_attributes": True}


BatchSyncResultOut.model_rebuild()
