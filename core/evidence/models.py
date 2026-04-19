from pydantic import BaseModel


class EvidenceCandidate(BaseModel):
    document_id: str
    chunk_id: str
    quote: str
    source_file: str
    page_number: int | None = None
