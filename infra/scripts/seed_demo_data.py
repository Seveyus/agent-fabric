from pathlib import Path
from uuid import uuid4

from db.base import Base
import db.models  # noqa: F401
from db.models.integration_connection import IntegrationConnection
from db.models.telemetry_snapshot import TelemetrySnapshot
from db.session import SessionLocal, engine

Base.metadata.create_all(bind=engine)

demo_dir = Path("storage/raw")
demo_dir.mkdir(parents=True, exist_ok=True)

sample = demo_dir / "demo_project_status.txt"
sample.write_text(
    """Weekly status
Action: finalize database migration
Owner: Alice Martin
Action: confirm API contract with mobile team
Blocked: waiting on data warehouse access
Dependency: pending vendor API approval
Risk: delivery may slip if migration is delayed
""",
    encoding="utf-8",
)
print(f"Seeded demo file at {sample}")

session = SessionLocal()
try:
    connection = (
        session.query(IntegrationConnection)
        .filter(IntegrationConnection.name == "Demo GitHub")
        .one_or_none()
    )
    if connection is None:
        connection = IntegrationConnection(
            id=f"conn_{uuid4().hex[:12]}",
            provider="github",
            name="Demo GitHub",
            base_url="https://api.github.com",
            auth_token="demo-token",
            config={"demo": True},
        )
        session.add(connection)
        session.flush()

    existing = (
        session.query(TelemetrySnapshot)
        .filter(TelemetrySnapshot.connection_id == connection.id, TelemetrySnapshot.project_ref == "demo/project-risk")
        .first()
    )
    if existing is None:
        session.add(
            TelemetrySnapshot(
                id=f"telemetry_{uuid4().hex[:12]}",
                connection_id=connection.id,
                provider="github",
                project_ref="demo/project-risk",
                risk_score=72.0,
                risk_level="high",
                metrics={
                    "repo_name": "demo/project-risk",
                    "open_pull_requests": 11,
                    "commits_last_7d": 0,
                    "active_contributors_30d": 1,
                    "last_commit_age_days": 12,
                    "stale_pull_requests": 4,
                },
                evidence={
                    "reasons": [
                        "No recent code activity for 12 days",
                        "4 pull requests have been stale for over 7 days",
                        "Delivery activity is concentrated on one contributor or fewer",
                    ]
                },
            )
        )

    session.commit()
finally:
    session.close()
