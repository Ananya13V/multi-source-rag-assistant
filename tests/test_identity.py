from app.ingestion.identity import make_chunk_id, make_content_hash


def test_chunk_id_is_deterministic():
    a = make_chunk_id("docs/setup.pdf", 3)
    b = make_chunk_id("docs/setup.pdf", 3)
    assert a == b


def test_chunk_id_depends_on_path_and_index():
    base = make_chunk_id("docs/setup.pdf", 0)
    diff_index = make_chunk_id("docs/setup.pdf", 1)
    diff_path = make_chunk_id("docs/other.pdf", 0)
    assert base != diff_index
    assert base != diff_path


def test_content_hash_changes_with_content():
    h1 = make_content_hash("hello world")
    h2 = make_content_hash("hello world!")
    assert h1 != h2
    assert make_content_hash("hello world") == h1
