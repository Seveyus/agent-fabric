from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class EvidenceRef(Base):
    __tablename__ = "evidence_refs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    finding_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    document_id: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
