from core.retrieval.embeddings import embed_texts
from core.retrieval.qdrant_store import QdrantStore
from core.retrieval.rerank import simple_rerank
from db.models.chunk import Chunk


class RetrieverAgent:
    name = "retriever"

    def run(self, query: str, db, limit: int = 8) -> list[Chunk]:
        store = QdrantStore()
        vector = embed_texts([query])[0]
        hits = store.search(vector, limit=limit)
        items = []
        for hit in hits:
            payload = hit.payload or {}
            items.append({"chunk_id": payload.get("chunk_id"), "text": payload.get("text", "")})
        reranked = simple_rerank(items, query)
        chunk_ids = [x["chunk_id"] for x in reranked if x.get("chunk_id")]
        if not chunk_ids:
            return []
        return db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
