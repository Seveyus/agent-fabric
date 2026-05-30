from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import db.models  # noqa: F401
from core.services.forecast_service import create_delay_forecast
from core.services.project_risk_score_service import compute_project_risk_score
from core.services.simulation_service import run_project_simulation
from core.services.world_model_service import build_experimental_world_state
from db.base import Base
from db.models.canonical_entity import CanonicalEntity
from db.models.canonical_relation import CanonicalRelation
from db.models.connector_sync_run import ConnectorSyncRun
from db.models.forecast_record import ForecastRecord
from db.models.integration import Integration
from db.models.metric_snapshot import MetricSnapshot
from db.models.simulation_run import SimulationRun
from db.models.world_model_state import WorldModelState
from db.session import SessionLocal, engine

Base.metadata.create_all(bind=engine)

demo_dir = Path("storage/raw")
demo_dir.mkdir(parents=True, exist_ok=True)

sample = demo_dir / "demo_project_status.txt"
sample.write_text(
    """Weekly status
ProjectRef: demo/project-risk
Action: finalize database migration
Owner: Alice Martin
Action: confirm API contract with mobile team
Blocked: waiting on data warehouse access
Dependency: pending vendor API approval
Risk: delivery may slip if migration is delayed
Decision: postpone release if warehouse access slips again
""",
    encoding="utf-8",
)
print(f"Seeded demo file at {sample}")

session = SessionLocal()
try:
    integration = session.query(Integration).filter(Integration.name == "Demo GitHub").one_or_none()
    if integration is None:
        integration = Integration(
            id=f"int_{uuid4().hex[:12]}",
            provider="github",
            name="Demo GitHub",
            base_url="https://api.github.com",
            auth_token="demo-token",
            config={"demo": True},
            status="connected",
        )
        session.add(integration)
        session.flush()

    existing_syncs = session.query(ConnectorSyncRun).filter(ConnectorSyncRun.project_ref == "demo/project-risk").count()
    if existing_syncs == 0:
        syncs = [
            {
                "started_at": datetime.now(timezone.utc) - timedelta(days=10),
                "metrics": {
                    "commit_velocity_7d": 4.0,
                    "stale_prs": 2.0,
                    "open_issues": 14.0,
                    "closed_issues": 9.0,
                    "issue_age_avg_days": 5.0,
                    "pr_review_delay_avg_hours": 18.0,
                    "release_recency_days": 9.0,
                    "backlog_size": 18.0,
                    "blocked_tickets": 3.0,
                    "overdue_tickets": 1.0,
                    "unassigned_tickets": 2.0,
                    "cycle_time_avg_days": 8.0,
                    "velocity_proxy": 7.0,
                    "recent_decisions": 1.0,
                    "undocumented_projects": 0.0,
                    "decisions_linked_to_projects": 1.0,
                    "stale_docs": 1.0,
                },
            },
            {
                "started_at": datetime.now(timezone.utc) - timedelta(days=5),
                "metrics": {
                    "commit_velocity_7d": 2.0,
                    "stale_prs": 3.0,
                    "open_issues": 17.0,
                    "closed_issues": 8.0,
                    "issue_age_avg_days": 8.0,
                    "pr_review_delay_avg_hours": 26.0,
                    "release_recency_days": 15.0,
                    "backlog_size": 22.0,
                    "blocked_tickets": 6.0,
                    "overdue_tickets": 3.0,
                    "unassigned_tickets": 4.0,
                    "cycle_time_avg_days": 10.0,
                    "velocity_proxy": 5.0,
                    "recent_decisions": 2.0,
                    "undocumented_projects": 0.0,
                    "decisions_linked_to_projects": 2.0,
                    "stale_docs": 2.0,
                },
            },
            {
                "started_at": datetime.now(timezone.utc) - timedelta(days=1),
                "metrics": {
                    "commit_velocity_7d": 0.0,
                    "stale_prs": 4.0,
                    "open_issues": 19.0,
                    "closed_issues": 6.0,
                    "issue_age_avg_days": 12.0,
                    "pr_review_delay_avg_hours": 42.0,
                    "release_recency_days": 28.0,
                    "backlog_size": 31.0,
                    "blocked_tickets": 12.0,
                    "overdue_tickets": 5.0,
                    "unassigned_tickets": 7.0,
                    "cycle_time_avg_days": 14.0,
                    "velocity_proxy": 3.0,
                    "recent_decisions": 3.0,
                    "undocumented_projects": 1.0,
                    "decisions_linked_to_projects": 3.0,
                    "stale_docs": 5.0,
                },
            },
        ]

        for item in syncs:
            sync_run = ConnectorSyncRun(
                id=f"sync_{uuid4().hex[:12]}",
                integration_id=integration.id,
                provider="github",
                project_ref="demo/project-risk",
                status="completed",
                raw_count=12,
                entity_count=10,
                relation_count=12,
                snapshot_count=len(item["metrics"]),
                started_at=item["started_at"],
                completed_at=item["started_at"] + timedelta(minutes=4),
            )
            session.add(sync_run)
            session.flush()
            for name, value in item["metrics"].items():
                source = "github" if name in {
                    "commit_velocity_7d", "stale_prs", "open_issues", "closed_issues",
                    "issue_age_avg_days", "pr_review_delay_avg_hours", "release_recency_days"
                } else "jira" if name in {
                    "backlog_size", "blocked_tickets", "overdue_tickets", "unassigned_tickets",
                    "cycle_time_avg_days", "velocity_proxy"
                } else "notion"
                session.add(
                    MetricSnapshot(
                        id=f"ms_{uuid4().hex[:12]}",
                        sync_run_id=sync_run.id,
                        project_ref="demo/project-risk",
                        metric_source=source,
                        metric_name=name,
                        metric_value=value,
                        metric_unit="count",
                        dimensions={},
                        captured_at=item["started_at"],
                    )
                )

        canonical_entities = [
            ("Project", "project:demo/project-risk", "demo/project-risk"),
            ("Repository", "repo:demo/project-risk", "demo/project-risk"),
            ("Customer", "customer:acme", "Acme Corp"),
            ("Decision", "decision:release-delay", "Postpone release if warehouse access slips"),
            ("Risk", "risk:warehouse-blocker", "Warehouse access blocks delivery"),
            ("Person", "person:alice-martin", "Alice Martin"),
            ("Team", "team:platform", "Platform Team"),
        ]
        for entity_type, entity_ref, name in canonical_entities:
            if session.query(CanonicalEntity).filter(CanonicalEntity.entity_ref == entity_ref).one_or_none() is None:
                session.add(
                    CanonicalEntity(
                        id=f"cent_{uuid4().hex[:12]}",
                        sync_run_id=sync_run.id,
                        source_provider="seed",
                        entity_type=entity_type,
                        entity_ref=entity_ref,
                        workspace_ref="workspace:seed:demo",
                        project_ref="demo/project-risk",
                        name=name,
                        attributes={},
                    )
                )

        canonical_relations = [
            ("linked_to_customer", "project:demo/project-risk", "customer:acme"),
            ("mentions", "decision:release-delay", "project:demo/project-risk"),
            ("affects", "risk:warehouse-blocker", "project:demo/project-risk"),
            ("blocks", "risk:warehouse-blocker", "repo:demo/project-risk"),
            ("owns", "team:platform", "project:demo/project-risk"),
            ("works_on", "person:alice-martin", "project:demo/project-risk"),
        ]
        for relation_type, source_ref, target_ref in canonical_relations:
            if (
                session.query(CanonicalRelation)
                .filter(
                    CanonicalRelation.relation_type == relation_type,
                    CanonicalRelation.source_ref == source_ref,
                    CanonicalRelation.target_ref == target_ref,
                )
                .one_or_none()
                is None
            ):
                session.add(
                    CanonicalRelation(
                        id=f"crel_{uuid4().hex[:12]}",
                        sync_run_id=sync_run.id,
                        source_provider="seed",
                        relation_type=relation_type,
                        source_ref=source_ref,
                        target_ref=target_ref,
                        project_ref="demo/project-risk",
                        attributes={},
                    )
                )
        session.commit()

    if session.query(ForecastRecord).filter(ForecastRecord.project_ref == "demo/project-risk").count() == 0:
        create_delay_forecast(session, "demo/project-risk")
    if session.query(SimulationRun).filter(SimulationRun.project_ref == "demo/project-risk").count() == 0:
        run_project_simulation(
            session,
            "demo/project-risk",
            "Add contributors and reduce blockers",
            {"commit_velocity_7d": 5, "stale_prs": -2, "blocked_tickets": -6, "unassigned_tickets": -3},
        )
    if session.query(WorldModelState).filter(WorldModelState.project_ref == "demo/project-risk").count() == 0:
        build_experimental_world_state(session, "demo/project-risk")

    summary = compute_project_risk_score(session, "demo/project-risk")
    print(f"Unified demo project score: {summary['risk_score']} ({summary['risk_level']})")
finally:
    session.close()
