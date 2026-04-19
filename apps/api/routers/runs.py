from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.evidence import EvidenceRefOut
from core.contracts.findings import FindingOut
from core.contracts.runs import ArtifactOut, RunDetailOut, RunOut
from db.models.artifact import Artifact
from db.models.evidence_ref import EvidenceRef
from db.models.finding import Finding
from db.models.run import Run

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("/{run_id}", response_model=RunDetailOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    finding_rows = db.query(Finding).filter(Finding.run_id == run_id).all()
    findings = []
    for finding in finding_rows:
        evidence_rows = db.query(EvidenceRef).filter(EvidenceRef.finding_id == finding.id).all()
        findings.append(
            FindingOut(
                id=finding.id,
                run_id=finding.run_id,
                kind=finding.kind,
                severity=finding.severity,
                title=finding.title,
                summary=finding.summary,
                confidence=finding.confidence,
                evidence=[EvidenceRefOut.model_validate(x) for x in evidence_rows],
            )
        )

    artifacts = db.query(Artifact).filter(Artifact.run_id == run_id).all()

    return RunDetailOut(
        **RunOut.model_validate(run).model_dump(),
        findings=findings,
        artifacts=[ArtifactOut.model_validate(x) for x in artifacts],
    )
