"""Hybrid retrieval: vector similarity + Postgres full-text search, fused
with Reciprocal Rank Fusion (RRF).

Fusion is pure Python (no DB round-trip needed to test it) so it's covered
by unit tests independent of a live Supabase instance.
"""
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.embeddings import embed_query
from app.retrieval.vector_store import text_search, vector_search


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source_path: str
    source_type: str
    score: float
    pdf_page_number: Optional[int] = None
    code_function_name: Optional[str] = None


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF: score(d) = sum_over_rankers( 1 / (k + rank_in_that_ranker) ).

    Documents absent from a given ranker's list simply contribute 0 for it.
    k damps the influence of any single high rank, per the standard RRF
    formulation (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_results(
    vector_results: list[dict],
    text_results: list[dict],
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    vector_ids = [r["chunk_id"] for r in vector_results]
    text_ids = [r["chunk_id"] for r in text_results]
    fused_scores = reciprocal_rank_fusion([vector_ids, text_ids], k=rrf_k)

    rows_by_id: dict[str, dict] = {r["chunk_id"]: r for r in vector_results}
    for r in text_results:
        rows_by_id.setdefault(r["chunk_id"], r)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=cid,
            content=rows_by_id[cid]["content"],
            source_path=rows_by_id[cid]["source_path"],
            source_type=rows_by_id[cid]["source_type"],
            pdf_page_number=rows_by_id[cid].get("pdf_page_number"),
            code_function_name=rows_by_id[cid].get("code_function_name"),
            score=fused_scores[cid],
        )
        for cid in ranked_ids
    ]


def hybrid_search(query_text: str) -> list[RetrievedChunk]:
    settings = get_settings()
    embedding = embed_query(query_text)
    vector_results = vector_search(embedding, settings.retrieval_top_k)
    text_results = text_search(query_text, settings.retrieval_top_k)
    return fuse_results(vector_results, text_results, top_k=settings.final_top_k)
