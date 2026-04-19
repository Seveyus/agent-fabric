from core.observability.audit import build_step_record
from core.observability.metrics import Timer


def execute_step(db, run_id: str, step_name: str, fn, *args, model_used=None, **kwargs):
    with Timer() as t:
        result = fn(*args, **kwargs)
    confidence = getattr(result, "confidence", None)
    db.add(build_step_record(run_id, step_name, "completed", t.duration_ms, confidence, model_used))
    db.commit()
    return result
