from datetime import date
from uuid import uuid4

from db.models.project_snapshot import ProjectSnapshot


def persist_project_snapshot(db, run_id: str, snapshot: dict) -> ProjectSnapshot:
    row = ProjectSnapshot(
        id=f"snapshot_{uuid4().hex[:12]}",
        run_id=run_id,
        snapshot_date=snapshot.get("snapshot_date", date.today()),
        total_documents=snapshot["total_documents"],
        total_chunks=snapshot["total_chunks"],
        owners_detected=snapshot["owners_detected"],
        actions_detected=snapshot["actions_detected"],
        dates_detected=snapshot["dates_detected"],
        risks_detected=snapshot["risks_detected"],
        blockers_detected=snapshot["blockers_detected"],
        dependencies_detected=snapshot["dependencies_detected"],
        owner_coverage_ratio=snapshot["owner_coverage_ratio"],
        risk_score=snapshot["risk_score"],
        risk_level=snapshot["risk_level"],
    )
    db.add(row)
    db.commit()
    return row
