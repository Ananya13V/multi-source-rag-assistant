"""Ties retrieval + prompting + citation validation into one answer call."""
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from app.config import get_settings
from app.generation.citations import CitationReport, validate_citations
from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.hybrid_search import RetrievedChunk


@dataclass
class SourceRef:
    tag: str
    source_path: str
    source_type: str
    pdf_page_number: int | None
    code_function_name: str | None


@dataclass
class AnswerResult:
    answer: str
    sources: list[SourceRef]
    citation_report: CitationReport


@lru_cache
def _get_llm() -> AzureChatOpenAI:
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment_name,
        temperature=0,
    )


def _source_refs(chunks: list[RetrievedChunk]) -> list[SourceRef]:
    return [
        SourceRef(
            tag=f"S{i}",
            source_path=chunk.source_path,
            source_type=chunk.source_type,
            pdf_page_number=chunk.pdf_page_number,
            code_function_name=chunk.code_function_name,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
    if not chunks:
        return AnswerResult(
            answer="I don't have any indexed documentation that covers this question.",
            sources=[],
            citation_report=validate_citations("", num_sources=0),
        )

    user_prompt = build_user_prompt(question, chunks)
    response = _get_llm().invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    raw_answer = response.content if isinstance(response.content, str) else str(response.content)

    report = validate_citations(raw_answer, num_sources=len(chunks))
    return AnswerResult(
        answer=report.cleaned_text.strip(),
        sources=_source_refs(chunks),
        citation_report=report,
    )
