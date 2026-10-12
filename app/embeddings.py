"""Ollama embeddings client (local inference, zero cost), shared by ingestion
and query rewriting. Requires `ollama pull nomic-embed-text` and `ollama serve`.
"""
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings_client() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embeddings_client().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embeddings_client().embed_query(text)
