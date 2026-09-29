"""FastAPI entry point.

    uvicorn app.main:app --reload
"""
import logging
import time

from fastapi import FastAPI, Request

from app.generation.answer import generate_answer
from app.models import QueryRequest, QueryResponse, SourceRefModel
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.query_rewrite import rewrite_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Source Technical Docs RAG Assistant")


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    history = [(turn.role, turn.content) for turn in request.chat_history]
    standalone_question = rewrite_query(request.question, history)

    chunks = hybrid_search(standalone_question)
    result = generate_answer(standalone_question, chunks)

    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        answer=result.answer,
        sources=[SourceRefModel(**vars(s)) for s in result.sources],
        citations_total=result.citation_report.total_citations,
        citations_invalid=len(result.citation_report.invalid_tags),
        latency_ms=latency_ms,
    )
