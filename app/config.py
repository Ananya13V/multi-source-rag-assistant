"""Centralized settings, loaded from environment / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Azure OpenAI (generation + embeddings)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_api_version: str = "2024-08-01-preview"

    # Supabase (Postgres + pgvector)
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_db_url: str = ""  # direct Postgres connection string, used by psycopg2

    # Retrieval / ingestion tuning
    embedding_dim: int = 1536
    chunk_size_words: int = 400
    chunk_overlap_words: int = 60
    retrieval_top_k: int = 20
    final_top_k: int = 5
    vector_weight: float = 0.6
    text_weight: float = 0.4

    # Table name
    documents_table: str = "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
