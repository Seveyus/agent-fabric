from uuid import uuid4


def new_trace_id() -> str:
    return f"trace_{uuid4().hex[:12]}"
