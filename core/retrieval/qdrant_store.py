from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from apps.api.config import settings


class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = "project_audit_chunks"

    def ensure_collection(self, vector_size: int):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert(self, points: list[PointStruct]):
        if not points:
            return
        self.ensure_collection(len(points[0].vector))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int = 5):
        return self.client.search(collection_name=self.collection, query_vector=vector, limit=limit)
