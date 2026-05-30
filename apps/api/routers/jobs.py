from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from apps.worker.job_queue import enqueue_pipeline
from core.contracts.jobs import JobCreateRequest, JobOut
from db.models.document import Document
from db.models.job import Job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobOut)
def create_job(payload: JobCreateRequest, db: Session = Depends(get_db)):
    documents = db.query(Document).filter(Document.id.in_(payload.document_ids)).all()
    if len(documents) != len(payload.document_ids):
        raise HTTPException(status_code=404, detail="One or more documents were not found.")

    job = Job(
        id=f"job_{uuid4().hex[:12]}",
        pipeline=payload.pipeline,
        project_ref=payload.project_ref,
        objective=payload.objective,
        status="queued",
        input_document_ids=payload.document_ids,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    enqueue_pipeline({"job_id": job.id, "pipeline": job.pipeline})
    return JobOut.model_validate(job)


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobOut.model_validate(job)
