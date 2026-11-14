from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStorage:
    """Small wrapper around Qdrant: create the collection, store chunks, search them."""

    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # A collection is like a table. Every vector in it must have the same size
        # (3072 = size of OpenAI's text-embedding-3-large vectors).
        # COSINE = compare vectors by the angle between them (similar meaning -> small angle).
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        """Insert points, or overwrite them if a point with the same id already exists."""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        """Return the text of the top_k most similar chunks and the PDFs they came from."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources = set()
        for r in results:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def list_sources(self) -> dict[str, int]:
        """Return {pdf name: number of chunks} for every document stored."""
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=["source"],
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            for p in points:
                src = (p.payload or {}).get("source", "unknown")
                counts[src] = counts.get(src, 0) + 1
            if offset is None:
                return counts
