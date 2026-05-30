from sqlalchemy.orm import Session

from core.orchestrator.pipeline_runner import ProjectRiskPipelineRunner
from core.orchestrator.registry import PipelineRegistry
from core.services.run_service import complete_run, create_run, fail_run
from db.models.job import Job
from db.session import SessionLocal


class OrchestratorEngine:
    def run_job(self, job_id: str, pipeline: str):
        if not PipelineRegistry.exists(pipeline):
            raise ValueError(f"Unknown pipeline: {pipeline}")

        db: Session = SessionLocal()
        try:
            job = db.get(Job, job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            run = create_run(db, job)
            ProjectRiskPipelineRunner().run(db, run, job)
            complete_run(db, run, job)
        except Exception:
            if "db" in locals() and "run" in locals() and "job" in locals():
                fail_run(db, run, job)
            raise
        finally:
            db.close()
