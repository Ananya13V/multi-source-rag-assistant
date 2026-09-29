"""Azure OpenAI embeddings client, shared by ingestion and query rewriting."""
from functools import lru_cache

from langchain_openai import AzureOpenAIEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings_client() -> AzureOpenAIEmbeddings:
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return get_embeddings_client().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embeddings_client().embed_query(text)
