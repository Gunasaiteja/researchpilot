"""Vertex AI model clients (Gemini for generation, text-embedding for vectors)."""
from functools import lru_cache

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from .config import settings


@lru_cache
def get_llm() -> ChatVertexAI:
    return ChatVertexAI(
        model_name=settings.llm_model,
        project=settings.gcp_project or None,
        location=settings.gcp_location,
        temperature=0.1,
    )


@lru_cache
def get_embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        model_name=settings.embedding_model,
        project=settings.gcp_project or None,
        location=settings.gcp_location,
    )


def get_callbacks() -> list:
    """Langfuse tracing is enabled only when keys are configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        try:
            from langfuse.callback import CallbackHandler  # langfuse v2
        except ImportError:
            from langfuse.langchain import CallbackHandler  # langfuse v3
        return [CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )]
    except Exception:
        return []
