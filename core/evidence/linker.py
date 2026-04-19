from db.models.chunk import Chunk
from db.models.document import Document


def link_evidence(chunks: list[Chunk], documents_by_id: dict[str, Document], max_items: int = 2) -> list[dict]:
    linked = []
    for chunk in chunks[:max_items]:
        doc = documents_by_id.get(chunk.document_id)
        linked.append(
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "quote": chunk.text[:300],
                "source_file": doc.original_name if doc else "unknown",
                "page_number": chunk.page_number,
            }
        )
    return linked
