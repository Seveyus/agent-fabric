from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ProjectSnapshot(Base):
    __tablename__ = "project_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owners_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    actions_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dates_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risks_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blockers_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dependencies_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_coverage_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
