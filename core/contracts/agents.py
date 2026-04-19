from pydantic import BaseModel


class AgentResult(BaseModel):
    ok: bool = True
    confidence: float = 0.0
    payload: dict = {}
    warnings: list[str] = []
