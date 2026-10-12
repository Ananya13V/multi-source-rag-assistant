"""Prompt construction: retrieved chunks become numbered, citable sources."""
from app.retrieval.hybrid_search import RetrievedChunk

SYSTEM_PROMPT = """You are a technical documentation assistant. Answer only \
using the numbered SOURCES below - never from prior knowledge.

Rules:
- Every factual sentence must end with the tag(s) of the source(s) it came \
from, e.g. "Retries use exponential backoff [S2]."
- If the sources don't contain the answer, say so plainly instead of \
guessing. Do not invent a citation to cover a guess.
- Never cite a source number that isn't listed below.
"""


def source_tag(index: int) -> str:
    return f"[S{index}]"


def format_sources(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.source_type == "pdf":
            origin = f"{chunk.source_path} (page {chunk.pdf_page_number})"
        else:
            fn = chunk.code_function_name or "module-level"
            origin = f"{chunk.source_path} ({fn})"
        blocks.append(f"{source_tag(i)} from {origin}:\n{chunk.content}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"SOURCES:\n\n{format_sources(chunks)}\n\nQUESTION: {question}"
