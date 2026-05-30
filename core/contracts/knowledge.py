from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeSearchRequest(BaseModel):
    query: str
    project_ref: str | None = None
    limit: int = Field(default=8, ge=1, le=25)


class KnowledgeSearchResultOut(BaseModel):
    result_type: str
    ref: str
    title: str
    score: float
    summary: str
    metadata: dict = {}


class KnowledgeSearchResponse(BaseModel):
    results: list[KnowledgeSearchResultOut]


class ForecastOut(BaseModel):
    id: str
    project_ref: str
    source: str
    delay_probability: float
    risk_trend: float
    forecast: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class SimulationRequest(BaseModel):
    project_ref: str
    scenario_name: str
    adjustments: dict = {}


class SimulationRunOut(BaseModel):
    id: str
    project_ref: str
    scenario_name: str
    baseline: dict
    adjustments: dict
    outcome: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class WorldStateOut(BaseModel):
    id: str
    project_ref: str
    state_kind: str
    latent_state: dict
    transitions: dict
    created_at: datetime

    model_config = {"from_attributes": True}
