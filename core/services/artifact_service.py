from pathlib import Path
from uuid import uuid4

from apps.api.config import settings
from db.models.artifact import Artifact


def write_report_artifact(db, run_id: str, markdown: str) -> Artifact:
    out_dir = Path(settings.storage_root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(markdown, encoding="utf-8")
    artifact = Artifact(
        id=f"artifact_{uuid4().hex[:12]}",
        run_id=run_id,
        kind="report_markdown",
        path=str(path),
        content_type="text/markdown",
    )
    db.add(artifact)
    db.commit()
    return artifact
