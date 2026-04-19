from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    original_name: str
    storage_path: str
    mime_type: str
    size_bytes: int
    status: str

    model_config = {"from_attributes": True}
