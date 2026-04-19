from datetime import datetime, timezone
from uuid import uuid4

from db.models.job import Job
from db.models.run import Run


def create_run(db, job: Job) -> Run:
    run = Run(
        id=f"run_{uuid4().hex[:12]}",
        job_id=job.id,
        pipeline=job.pipeline,
        status="running",
        current_step="initializing",
    )
    db.add(run)
    job.status = "running"
    db.commit()
    db.refresh(run)
    return run


def mark_run_step(db, run: Run, step: str):
    run.current_step = step
    db.commit()


def complete_run(db, run: Run, job: Job):
    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    run.current_step = "completed"
    job.status = "completed"
    db.commit()


def fail_run(db, run: Run, job: Job):
    run.status = "failed"
    run.completed_at = datetime.now(timezone.utc)
    run.current_step = "failed"
    job.status = "failed"
    db.commit()
