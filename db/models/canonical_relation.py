from datetime import datetime, timezone

from sqlalchemy import DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class CanonicalRelation(Base):
    __tablename__ = "canonical_relations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sync_run_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    source_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    target_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    project_ref: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
