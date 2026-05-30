from uuid import uuid4

from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation
from db.models.metric_snapshot import MetricSnapshot
from db.models.world_model_state import WorldModelState


def build_experimental_world_state(db, project_ref: str) -> WorldModelState:
    project_nodes = db.query(CanonicalEntity).filter(CanonicalEntity.project_ref == project_ref).all()
    related_edges = db.query(CanonicalRelation).filter(CanonicalRelation.project_ref == project_ref).all()
    metrics = (
        db.query(MetricSnapshot)
        .filter(MetricSnapshot.project_ref == project_ref)
        .order_by(MetricSnapshot.captured_at.desc())
        .limit(20)
        .all()
    )
    if not metrics:
        raise ValueError(f"No metric snapshot found for project '{project_ref}'")

    latest_metric = metrics[0]
    avg_risk = sum(item.metric_value for item in metrics) / len(metrics)
    latent_state = {
        "node_count": len(project_nodes),
        "edge_count": len(related_edges),
        "recent_risk_mean": round(avg_risk, 2),
        "current_risk": latest_metric.metric_value,
        "provider_mix": sorted({item.metric_source for item in metrics}),
        "signal_density": len(metrics),
    }
    transitions = {
        "interpretation": (
            "Experimental latent state built from graph density and recent telemetry. "
            "This is a research scaffold, not a trained JEPA-like model."
        ),
        "expected_next_state": "risk_increase" if latest_metric.metric_value >= avg_risk else "risk_stable",
        "transition_drivers": [f"{latest_metric.metric_name}={latest_metric.metric_value}"],
    }

    row = WorldModelState(
        id=f"world_{uuid4().hex[:12]}",
        project_ref=project_ref,
        latent_state=latent_state,
        transitions=transitions,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
