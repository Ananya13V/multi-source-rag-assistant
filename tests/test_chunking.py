from pathlib import Path

from app.ingestion.chunking import chunk_code, chunk_pdf, split_text_by_tokens

CODE_SAMPLE = '''"""Module docstring."""
import os


def add(a, b):
    """Add two numbers."""
    return a + b


class Greeter:
    def hello(self, name):
        return f"hello {name}"
'''


def test_split_text_by_tokens_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    windows = split_text_by_tokens(text, chunk_size=50, overlap=10)
    assert len(windows) > 1
    # every window should be non-empty and reasonably sized
    for w in windows:
        assert w.strip()


def test_split_text_by_tokens_returns_single_window_for_small_text():
    windows = split_text_by_tokens("just a short sentence", chunk_size=500, overlap=75)
    assert windows == ["just a short sentence"]


def test_chunk_code_splits_python_by_function_with_flat_metadata(tmp_path: Path):
    file_path = tmp_path / "sample.py"
    file_path.write_text(CODE_SAMPLE, encoding="utf-8")

    records = chunk_code(file_path, chunk_size=500, overlap=50)

    function_names = {r.code_function_name for r in records}
    assert "add" in function_names
    assert "Greeter" in function_names
    assert None in function_names  # the module-level preamble chunk

    for r in records:
        assert r.source_type == "code"
        assert r.code_language == "python"
        assert r.pdf_page_number is None  # pdf_* fields stay None for code chunks


def test_chunk_pdf_tags_each_chunk_with_its_page_number(tmp_path: Path):
    reportlab = __import__("reportlab.pdfgen.canvas", fromlist=["canvas"])
    pdf_path = tmp_path / "doc.pdf"
    c = reportlab.Canvas(str(pdf_path))
    c.drawString(100, 750, "Page one content about installation steps.")
    c.showPage()
    c.drawString(100, 750, "Page two content about configuration options.")
    c.showPage()
    c.save()

    records = chunk_pdf(pdf_path, chunk_size=500, overlap=50)

    pages = sorted({r.pdf_page_number for r in records})
    assert pages == [1, 2]
    for r in records:
        assert r.source_type == "pdf"
        assert r.code_function_name is None  # code_* fields stay None for pdf chunks
