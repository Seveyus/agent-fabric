from datetime import datetime
from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    pipeline: str = Field(default="project_risk")
    project_ref: str | None = None
    objective: str
    document_ids: list[str]


class JobOut(BaseModel):
    id: str
    pipeline: str
    project_ref: str | None = None
    objective: str
    status: str
    input_document_ids: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}
