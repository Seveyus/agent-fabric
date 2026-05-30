from datetime import date, datetime

from pydantic import BaseModel


class ProjectSnapshotOut(BaseModel):
    id: str
    run_id: str
    snapshot_date: date
    total_documents: int
    total_chunks: int
    owners_detected: int
    actions_detected: int
    dates_detected: int
    risks_detected: int
    blockers_detected: int
    dependencies_detected: int
    owner_coverage_ratio: float
    risk_score: float
    risk_level: str
    created_at: datetime

    model_config = {"from_attributes": True}
