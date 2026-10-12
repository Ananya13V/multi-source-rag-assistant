"""Source-aware chunking.

Two source types are supported today (see DECISIONS.md): PDF technical docs
and source-code files. Each produces `ChunkRecord`s carrying only the flat,
source-type-prefixed metadata fields relevant to it (pdf_* / code_*) -
irrelevant fields stay None rather than living in a nested sub-object.
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

CODE_LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".md": "markdown",
}

# Word-count based length function rather than a model-specific tokenizer
# (e.g. tiktoken): it needs no network access to download vocab files and
# ingestion doesn't need token-exact precision, just a consistent, roughly
# proportional chunk size across PDF prose and source code.
_WORD_RE = re.compile(r"\S+")


@dataclass
class ChunkRecord:
    source_path: str
    source_type: str  # 'pdf' | 'code'
    chunk_index: int
    content: str
    pdf_page_number: Optional[int] = None
    pdf_title: Optional[str] = None
    code_language: Optional[str] = None
    code_function_name: Optional[str] = None
    code_start_line: Optional[int] = None
    code_end_line: Optional[int] = None


def split_text_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, sized in whitespace-delimited words."""
    text = text.strip()
    if not text:
        return []
    words = list(_WORD_RE.finditer(text))
    if len(words) <= chunk_size:
        return [text]

    step = max(chunk_size - overlap, 1)
    windows = []
    for start in range(0, len(words), step):
        window_words = words[start : start + chunk_size]
        if not window_words:
            break
        windows.append(text[window_words[0].start() : window_words[-1].end()])
        if start + chunk_size >= len(words):
            break
    return windows


def chunk_pdf(path: Path, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    reader = PdfReader(str(path))
    title = path.stem
    records: list[ChunkRecord] = []
    chunk_index = 0
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for window in split_text_by_tokens(page_text, chunk_size, overlap):
            records.append(
                ChunkRecord(
                    source_path=str(path),
                    source_type="pdf",
                    chunk_index=chunk_index,
                    content=window,
                    pdf_page_number=page_number,
                    pdf_title=title,
                )
            )
            chunk_index += 1
    return records


def _python_top_level_units(source: str) -> list[tuple[str, str, int, int]]:
    """Return (name, segment, start_line, end_line) for each top-level
    function/class def, plus a leading "<module>" unit for everything above
    the first def (imports, module docstring, constants)."""
    tree = ast.parse(source)
    units: list[tuple[str, str, int, int]] = []

    def_nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    first_def_line = def_nodes[0].lineno if def_nodes else None
    lines = source.splitlines()
    if first_def_line and first_def_line > 1:
        preamble = "\n".join(lines[: first_def_line - 1]).strip()
        if preamble:
            units.append(("<module>", preamble, 1, first_def_line - 1))
    elif not def_nodes:
        units.append(("<module>", source, 1, len(lines)))

    for node in def_nodes:
        segment = ast.get_source_segment(source, node) or ""
        units.append((node.name, segment, node.lineno, node.end_lineno or node.lineno))

    return units


def chunk_code(path: Path, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    language = CODE_LANGUAGE_BY_EXTENSION.get(path.suffix, path.suffix.lstrip("."))
    source = path.read_text(encoding="utf-8", errors="replace")
    records: list[ChunkRecord] = []
    chunk_index = 0

    if path.suffix == ".py":
        try:
            units = _python_top_level_units(source)
        except SyntaxError:
            units = [("<module>", source, 1, len(source.splitlines()))]
    else:
        units = [("<module>", source, 1, len(source.splitlines()))]

    for name, segment, start_line, end_line in units:
        windows = split_text_by_tokens(segment, chunk_size, overlap)
        for window in windows:
            records.append(
                ChunkRecord(
                    source_path=str(path),
                    source_type="code",
                    chunk_index=chunk_index,
                    content=window,
                    code_language=language,
                    code_function_name=None if name == "<module>" else name,
                    code_start_line=start_line,
                    code_end_line=end_line,
                )
            )
            chunk_index += 1
    return records


def chunk_file(path: Path, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    if path.suffix.lower() == ".pdf":
        return chunk_pdf(path, chunk_size, overlap)
    return chunk_code(path, chunk_size, overlap)
