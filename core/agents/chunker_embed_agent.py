from uuid import uuid4

from qdrant_client.http.models import PointStruct

from core.retrieval.chunking import chunk_text
from core.retrieval.embeddings import embed_texts
from core.retrieval.qdrant_store import QdrantStore
from db.models.chunk import Chunk


class ChunkerEmbedAgent:
    name = "chunker_embed"

    def run(self, parsed_docs: list[dict], db) -> list[Chunk]:
        store = QdrantStore()
        chunks_created = []
        points = []
        for item in parsed_docs:
            chunks = chunk_text(item["text"])
            if not chunks:
                continue
            vectors = embed_texts(chunks)
            for idx, (text, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = f"chunk_{uuid4().hex[:12]}"
                point_id = f"point_{uuid4().hex[:12]}"
                chunk = Chunk(
                    id=chunk_id,
                    document_id=item["document_id"],
                    ordinal=idx,
                    text=text,
                    page_number=None,
                    qdrant_point_id=point_id,
                )
                db.add(chunk)
                chunks_created.append(chunk)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk_id,
                            "document_id": item["document_id"],
                            "text": text,
                        },
                    )
                )
        db.flush()
        if points:
            store.upsert(points)
        return chunks_created
