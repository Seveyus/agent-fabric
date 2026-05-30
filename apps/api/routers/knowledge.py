from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.deps import get_db
from core.contracts.knowledge import (
    ForecastOut,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResultOut,
    SimulationRequest,
    SimulationRunOut,
    WorldStateOut,
)
from core.services.forecast_service import create_delay_forecast
from core.services.knowledge_search_service import hybrid_search
from core.services.project_risk_score_service import compute_project_risk_score
from core.services.simulation_service import run_project_simulation
from core.services.world_model_service import build_experimental_world_state
from db.models.forecast_record import ForecastRecord
from db.models.simulation_run import SimulationRun
from db.models.world_model_state import WorldModelState

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/project-risk/{project_ref}")
def project_risk(project_ref: str, db: Session = Depends(get_db)):
    return compute_project_risk_score(db, project_ref)


@router.post("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(payload: KnowledgeSearchRequest, db: Session = Depends(get_db)):
    results = hybrid_search(db, payload.query, payload.project_ref, payload.limit)
    return KnowledgeSearchResponse(results=[KnowledgeSearchResultOut(**item) for item in results])


@router.post("/forecast/{project_ref}", response_model=ForecastOut)
def forecast_project_delay(project_ref: str, db: Session = Depends(get_db)):
    try:
        record = create_delay_forecast(db, project_ref)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ForecastOut.model_validate(record)


@router.get("/forecasts", response_model=list[ForecastOut])
def list_forecasts(project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ForecastRecord).order_by(ForecastRecord.created_at.desc())
    if project_ref:
        query = query.filter(ForecastRecord.project_ref == project_ref)
    rows = query.limit(50).all()
    return [ForecastOut.model_validate(row) for row in rows]


@router.post("/simulate", response_model=SimulationRunOut)
def simulate_project(payload: SimulationRequest, db: Session = Depends(get_db)):
    try:
        row = run_project_simulation(db, payload.project_ref, payload.scenario_name, payload.adjustments)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SimulationRunOut.model_validate(row)


@router.get("/simulations", response_model=list[SimulationRunOut])
def list_simulations(project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(SimulationRun).order_by(SimulationRun.created_at.desc())
    if project_ref:
        query = query.filter(SimulationRun.project_ref == project_ref)
    rows = query.limit(50).all()
    return [SimulationRunOut.model_validate(row) for row in rows]


@router.post("/world-state/{project_ref}", response_model=WorldStateOut)
def create_world_state(project_ref: str, db: Session = Depends(get_db)):
    try:
        row = build_experimental_world_state(db, project_ref)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WorldStateOut.model_validate(row)


@router.get("/world-states", response_model=list[WorldStateOut])
def list_world_states(project_ref: str | None = None, db: Session = Depends(get_db)):
    query = db.query(WorldModelState).order_by(WorldModelState.created_at.desc())
    if project_ref:
        query = query.filter(WorldModelState.project_ref == project_ref)
    rows = query.limit(50).all()
    return [WorldStateOut.model_validate(row) for row in rows]
