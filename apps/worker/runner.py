import logging
import time

from apps.worker.queue import dequeue_pipeline
from core.orchestrator.engine import OrchestratorEngine

logger = logging.getLogger("agent_fabric.worker")


def run_forever():
    engine = OrchestratorEngine()
    while True:
        job = dequeue_pipeline()
        if not job:
            time.sleep(1)
            continue

        try:
            logger.info("received_job job_id=%s pipeline=%s", job["job_id"], job["pipeline"])
            engine.run_job(job["job_id"], job["pipeline"])
        except Exception:
            logger.exception("pipeline_failed job_id=%s", job.get("job_id"))
