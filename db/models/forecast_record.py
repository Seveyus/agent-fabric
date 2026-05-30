from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class ForecastRecord(Base):
    __tablename__ = "forecast_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    delay_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_trend: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    forecast: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
