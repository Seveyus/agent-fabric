from sqlalchemy import func

from db.models.canonical_relation import CanonicalRelation
from db.models.connector_sync_run import ConnectorSyncRun
from db.models.forecast_record import ForecastRecord
from db.models.integration import Integration
from db.models.metric_snapshot import MetricSnapshot
from db.models.project_snapshot import ProjectSnapshot
from db.models.run import Run
from db.models.simulation_run import SimulationRun
from db.models.world_model_state import WorldModelState
from core.services.project_risk_score_service import compute_project_risk_score


def build_dashboard_overview(db) -> dict:
    integrations_count = db.query(func.count(Integration.id)).scalar() or 0
    telemetry_count = db.query(func.count(MetricSnapshot.id)).scalar() or 0
    document_snapshot_count = db.query(func.count(ProjectSnapshot.id)).scalar() or 0
    run_count = db.query(func.count(Run.id)).scalar() or 0
    forecasts_count = db.query(func.count(ForecastRecord.id)).scalar() or 0
    simulations_count = db.query(func.count(SimulationRun.id)).scalar() or 0
    world_states_count = db.query(func.count(WorldModelState.id)).scalar() or 0
    sync_runs_count = db.query(func.count(ConnectorSyncRun.id)).scalar() or 0

    latest_metrics = db.query(MetricSnapshot).order_by(MetricSnapshot.captured_at.desc()).limit(50).all()
    latest_project_snapshots = (
        db.query(ProjectSnapshot)
        .order_by(ProjectSnapshot.created_at.desc())
        .limit(12)
        .all()
    )
    recent_runs = db.query(Run).order_by(Run.started_at.desc()).limit(10).all()
    latest_sync_runs = db.query(ConnectorSyncRun).order_by(ConnectorSyncRun.started_at.desc()).limit(12).all()
    latest_forecasts = db.query(ForecastRecord).order_by(ForecastRecord.created_at.desc()).limit(12).all()
    latest_simulations = db.query(SimulationRun).order_by(SimulationRun.created_at.desc()).limit(12).all()
    latest_world_states = db.query(WorldModelState).order_by(WorldModelState.created_at.desc()).limit(12).all()
    graph_relations = db.query(CanonicalRelation).order_by(CanonicalRelation.created_at.desc()).limit(20).all()
    integrations = db.query(Integration).order_by(Integration.created_at.desc()).limit(20).all()
    forecasts_by_project = {item.project_ref: item for item in latest_forecasts}
    latest_sync_by_project = {}
    relations_by_project: dict[str, list[dict]] = {}

    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    provider_distribution: dict[str, int] = {}
    project_refs = [ref for (ref,) in db.query(MetricSnapshot.project_ref).distinct().all()]

    for item in latest_sync_runs:
        latest_sync_by_project.setdefault(item.project_ref, item)

    for relation in graph_relations:
        if not relation.project_ref:
            continue
        relations_by_project.setdefault(relation.project_ref, []).append(
            {
                "relation_type": relation.relation_type,
                "source_ref": relation.source_ref,
                "target_ref": relation.target_ref,
            }
        )

    active_projects = []
    for project_ref in project_refs[:12]:
        score = compute_project_risk_score(db, project_ref)
        score["provider"] = "unified"
        latest_sync = latest_sync_by_project.get(project_ref)
        forecast = forecasts_by_project.get(project_ref)
        score["last_sync_status"] = latest_sync.status if latest_sync else "unknown"
        score["last_sync_at"] = latest_sync.completed_at.isoformat() if latest_sync and latest_sync.completed_at else None
        score["top_relationships"] = relations_by_project.get(project_ref, [])[:4]
        score["forecast_status"] = forecast.forecast["projected_status"] if forecast else None
        score["delay_probability"] = forecast.delay_probability if forecast else None
        active_projects.append(score)
        risk_distribution[score["risk_level"]] = risk_distribution.get(score["risk_level"], 0) + 1

    for item in latest_metrics:
        provider_distribution[item.metric_source] = provider_distribution.get(item.metric_source, 0) + 1

    return {
        "summary": {
            "integrations_count": integrations_count,
            "telemetry_count": telemetry_count,
            "document_snapshot_count": document_snapshot_count,
            "run_count": run_count,
            "active_projects_count": len(project_refs),
            "forecasts_count": forecasts_count,
            "simulations_count": simulations_count,
            "world_states_count": world_states_count,
            "sync_runs_count": sync_runs_count,
        },
        "risk_distribution": risk_distribution,
        "provider_distribution": provider_distribution,
        "active_projects": active_projects[:8],
        "integrations": [
            {
                "id": item.id,
                "provider": item.provider,
                "name": item.name,
                "status": item.status,
                "created_at": item.created_at.isoformat(),
            }
            for item in integrations
        ],
        "sync_runs": [
            {
                "id": item.id,
                "integration_id": item.integration_id,
                "provider": item.provider,
                "project_ref": item.project_ref,
                "status": item.status,
                "raw_count": item.raw_count,
                "entity_count": item.entity_count,
                "relation_count": item.relation_count,
                "snapshot_count": item.snapshot_count,
                "started_at": item.started_at.isoformat(),
                "completed_at": item.completed_at.isoformat() if item.completed_at else None,
            }
            for item in latest_sync_runs
        ],
        "telemetry_snapshots": [
            {
                "id": item.id,
                "provider": item.metric_source,
                "project_ref": item.project_ref,
                "risk_score": None,
                "risk_level": None,
                "metrics": {item.metric_name: item.metric_value},
                "evidence": {"reasons": []},
                "created_at": item.captured_at.isoformat(),
            }
            for item in latest_metrics[:12]
        ],
        "document_snapshots": [
            {
                "id": item.id,
                "run_id": item.run_id,
                "project_ref": item.project_ref,
                "risk_score": item.risk_score,
                "risk_level": item.risk_level,
                "actions_detected": item.actions_detected,
                "owners_detected": item.owners_detected,
                "blockers_detected": item.blockers_detected,
                "dependencies_detected": item.dependencies_detected,
                "created_at": item.created_at.isoformat(),
            }
            for item in latest_project_snapshots
        ],
        "recent_runs": [
            {
                "id": item.id,
                "job_id": item.job_id,
                "pipeline": item.pipeline,
                "status": item.status,
                "current_step": item.current_step,
                "started_at": item.started_at.isoformat(),
                "completed_at": item.completed_at.isoformat() if item.completed_at else None,
            }
            for item in recent_runs
        ],
        "forecasts": [
            {
                "id": item.id,
                "project_ref": item.project_ref,
                "source": item.source,
                "delay_probability": item.delay_probability,
                "risk_trend": item.risk_trend,
                "forecast": item.forecast,
                "created_at": item.created_at.isoformat(),
            }
            for item in latest_forecasts
        ],
        "simulations": [
            {
                "id": item.id,
                "project_ref": item.project_ref,
                "scenario_name": item.scenario_name,
                "baseline": item.baseline,
                "adjustments": item.adjustments,
                "outcome": item.outcome,
                "created_at": item.created_at.isoformat(),
            }
            for item in latest_simulations
        ],
        "world_states": [
            {
                "id": item.id,
                "project_ref": item.project_ref,
                "state_kind": item.state_kind,
                "latent_state": item.latent_state,
                "transitions": item.transitions,
                "created_at": item.created_at.isoformat(),
            }
            for item in latest_world_states
        ],
        "graph_relationships": [
            {
                "relation_type": item.relation_type,
                "source_ref": item.source_ref,
                "target_ref": item.target_ref,
                "project_ref": item.project_ref,
            }
            for item in graph_relations
        ],
    }
