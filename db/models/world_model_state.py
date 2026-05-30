from datetime import datetime, timezone

from sqlalchemy import DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class WorldModelState(Base):
    __tablename__ = "world_model_states"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    project_ref: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    state_kind: Mapped[str] = mapped_column(String(64), nullable=False, default="experimental_latent_state")
    latent_state: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    transitions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
