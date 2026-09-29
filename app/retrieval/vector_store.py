"""Thin wrapper around the two Supabase RPCs defined in sql/schema.sql."""
from app.db import get_supabase_client


def vector_search(embedding: list[float], match_count: int) -> list[dict]:
    client = get_supabase_client()
    resp = client.rpc(
        "match_documents_vector",
        {"query_embedding": embedding, "match_count": match_count},
    ).execute()
    return resp.data or []


def text_search(query_text: str, match_count: int) -> list[dict]:
    client = get_supabase_client()
    resp = client.rpc(
        "match_documents_text",
        {"query_text": query_text, "match_count": match_count},
    ).execute()
    return resp.data or []
