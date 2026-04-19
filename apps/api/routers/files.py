from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from apps.api.config import settings
from apps.api.deps import get_db
from core.contracts.documents import DocumentOut
from db.models.document import Document

router = APIRouter(prefix="/files", tags=["files"])


@router.post("", response_model=DocumentOut)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    document_id = f"doc_{uuid4().hex[:12]}"
    destination_dir = Path(settings.storage_root) / "raw"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{document_id}{suffix}"
    content = await file.read()
    destination.write_bytes(content)

    doc = Document(
        id=document_id,
        original_name=file.filename,
        storage_path=str(destination),
        mime_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        status="uploaded",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return DocumentOut.model_validate(doc)
