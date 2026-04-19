from pydantic import BaseModel

from core.contracts.evidence import EvidenceRefOut


class FindingOut(BaseModel):
    id: str
    run_id: str
    kind: str
    severity: str
    title: str
    summary: str
    confidence: float
    evidence: list[EvidenceRefOut] = []

    model_config = {"from_attributes": True}
