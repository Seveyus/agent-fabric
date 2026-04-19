from pydantic import BaseModel


class ProjectAuditInput(BaseModel):
    job_id: str
    objective: str
    document_ids: list[str]


class ProjectAuditFinding(BaseModel):
    kind: str
    severity: str
    title: str
    summary: str
    confidence: float
    evidence: list[dict] = []
