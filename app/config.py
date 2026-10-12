"""Centralized settings, loaded from environment / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama (local inference, generation + embeddings) - zero-cost, no API key
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # Supabase (Postgres + pgvector)
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_db_url: str = ""  # direct Postgres connection string, used by psycopg2

    # Retrieval / ingestion tuning
    embedding_dim: int = 768  # nomic-embed-text's output dimension; must match sql/schema.sql's vector() column
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
