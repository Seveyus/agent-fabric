from sqlalchemy import func

from db.models.integration_connection import IntegrationConnection
from db.models.project_snapshot import ProjectSnapshot
from db.models.run import Run
from db.models.telemetry_snapshot import TelemetrySnapshot


def build_dashboard_overview(db) -> dict:
    integrations_count = db.query(func.count(IntegrationConnection.id)).scalar() or 0
    telemetry_count = db.query(func.count(TelemetrySnapshot.id)).scalar() or 0
    document_snapshot_count = db.query(func.count(ProjectSnapshot.id)).scalar() or 0
    run_count = db.query(func.count(Run.id)).scalar() or 0

    latest_telemetry = (
        db.query(TelemetrySnapshot)
        .order_by(TelemetrySnapshot.created_at.desc())
        .limit(12)
        .all()
    )
    latest_project_snapshots = (
        db.query(ProjectSnapshot)
        .order_by(ProjectSnapshot.created_at.desc())
        .limit(12)
        .all()
    )
    recent_runs = db.query(Run).order_by(Run.started_at.desc()).limit(10).all()

    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    provider_distribution: dict[str, int] = {}
    for item in latest_telemetry:
        risk_distribution[item.risk_level] = risk_distribution.get(item.risk_level, 0) + 1
        provider_distribution[item.provider] = provider_distribution.get(item.provider, 0) + 1

    active_projects: dict[str, dict] = {}
    for item in latest_telemetry:
        if item.project_ref not in active_projects:
            active_projects[item.project_ref] = {
                "project_ref": item.project_ref,
                "provider": item.provider,
                "risk_score": item.risk_score,
                "risk_level": item.risk_level,
                "updated_at": item.created_at.isoformat(),
            }

    return {
        "summary": {
            "integrations_count": integrations_count,
            "telemetry_count": telemetry_count,
            "document_snapshot_count": document_snapshot_count,
            "run_count": run_count,
            "active_projects_count": len(active_projects),
        },
        "risk_distribution": risk_distribution,
        "provider_distribution": provider_distribution,
        "active_projects": list(active_projects.values())[:8],
        "telemetry_snapshots": [
            {
                "id": item.id,
                "provider": item.provider,
                "project_ref": item.project_ref,
                "risk_score": item.risk_score,
                "risk_level": item.risk_level,
                "metrics": item.metrics,
                "evidence": item.evidence,
                "created_at": item.created_at.isoformat(),
            }
            for item in latest_telemetry
        ],
        "document_snapshots": [
            {
                "id": item.id,
                "run_id": item.run_id,
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
    }
