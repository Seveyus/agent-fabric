from pathlib import Path

from db.base import Base
from db.session import engine

Base.metadata.create_all(bind=engine)

demo_dir = Path("storage/raw")
demo_dir.mkdir(parents=True, exist_ok=True)

sample = demo_dir / "demo_project_status.txt"
sample.write_text(
    """Weekly status
Action: finalize database migration
Owner: Alice Martin
Action: confirm API contract with mobile team
Risk: delivery may slip if migration is delayed
""",
    encoding="utf-8",
)
print(f"Seeded demo file at {sample}")
