from pydantic import BaseModel


class EvidenceRefOut(BaseModel):
    id: str
    finding_id: str
    document_id: str
    chunk_id: str | None = None
    quote: str
    source_file: str
    page_number: int | None = None

    model_config = {"from_attributes": True}
