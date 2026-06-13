from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class CanonicalRelation(Base):
    """
    Append-only temporal edge in the delivery knowledge graph.

    Relations are NEVER overwritten. When a relationship changes (e.g. a PR that
    was blocking a ticket gets merged), we set valid_to on the old edge and insert a
    new one. This preserves the full causal history of the project.

    Causal relation_types:
      triggered_by    — this entity exists because of another event/entity
      constrained_by  — this entity's behaviour is governed by another
      leads_to        — this entity causally produces another
      overruled_by    — a previous decision was reversed by this one
      blocks          — standard blocking dependency
      depends_on      — softer dependency
      observes        — monitoring/integration relationship
    """

    __tablename__ = "canonical_relations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    sync_run_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    source_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    target_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    project_ref: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # Temporal validity — NULL valid_to means the edge is currently active
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
