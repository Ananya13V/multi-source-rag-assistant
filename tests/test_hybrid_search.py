from app.retrieval.hybrid_search import fuse_results, reciprocal_rank_fusion


def test_rrf_rewards_documents_ranked_highly_in_both_lists():
    vector_ids = ["a", "b", "c"]
    text_ids = ["b", "a", "d"]

    scores = reciprocal_rank_fusion([vector_ids, text_ids])

    # "b" is rank 2 in vector but rank 1 in text; "a" is rank 1 in vector but
    # rank 2 in text -> both should beat "c"/"d", which only appear once.
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    assert scores["d"] > 0  # still present, just from one ranker


def test_fuse_results_merges_rows_and_respects_top_k():
    vector_results = [
        {"chunk_id": "a", "content": "vector-a", "source_path": "x.py", "source_type": "code"},
        {"chunk_id": "b", "content": "vector-b", "source_path": "x.py", "source_type": "code"},
    ]
    text_results = [
        {"chunk_id": "b", "content": "text-b", "source_path": "x.py", "source_type": "code"},
        {"chunk_id": "c", "content": "text-c", "source_path": "y.pdf", "source_type": "pdf", "pdf_page_number": 4},
    ]

    fused = fuse_results(vector_results, text_results, top_k=2)

    assert len(fused) == 2
    assert fused[0].chunk_id == "b"  # appears in both lists -> highest fused score
    # row data for a chunk found via the vector leg comes from that leg
    assert fused[0].content == "vector-b"


def test_fuse_results_falls_back_to_text_row_when_only_found_via_text():
    vector_results = []
    text_results = [
        {"chunk_id": "c", "content": "text-c", "source_path": "y.pdf", "source_type": "pdf", "pdf_page_number": 4},
    ]

    fused = fuse_results(vector_results, text_results, top_k=5)

    assert len(fused) == 1
    assert fused[0].content == "text-c"
    assert fused[0].pdf_page_number == 4
