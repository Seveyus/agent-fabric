def step_event(step_name: str, status: str, meta: dict | None = None) -> dict:
    return {"step_name": step_name, "status": status, "meta": meta or {}}
