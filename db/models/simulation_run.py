from datetime import datetime, timezone

from sqlalchemy import DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    scenario_name: Mapped[str] = mapped_column(String(255), nullable=False)
    baseline: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    adjustments: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    outcome: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
