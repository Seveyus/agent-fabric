from datetime import datetime
from pydantic import BaseModel

from core.contracts.findings import FindingOut
from core.contracts.project_snapshots import ProjectSnapshotOut


class ArtifactOut(BaseModel):
    id: str
    kind: str
    path: str
    content_type: str

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id: str
    job_id: str
    pipeline: str
    status: str
    current_step: str | None = None
    started_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class RunDetailOut(RunOut):
    findings: list[FindingOut] = []
    artifacts: list[ArtifactOut] = []
    snapshots: list[ProjectSnapshotOut] = []
