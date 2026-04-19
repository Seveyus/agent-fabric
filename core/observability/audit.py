from uuid import uuid4

from db.models.agent_step import AgentStep


def build_step_record(run_id: str, step_name: str, status: str, duration_ms: int, confidence=None, model_used=None):
    return AgentStep(
        id=f"step_{uuid4().hex[:12]}",
        run_id=run_id,
        step_name=step_name,
        status=status,
        duration_ms=duration_ms,
        confidence=confidence,
        model_used=model_used,
    )
