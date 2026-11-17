"""Qdrant vector store (local on-disk for dev, Qdrant Cloud/server for prod)."""
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .config import settings
from .llm import get_embeddings


@lru_cache
def get_store() -> QdrantVectorStore:
    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    else:
        client = QdrantClient(path=settings.qdrant_path)

    if not client.collection_exists(settings.collection):
        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=get_embeddings(),
    )
