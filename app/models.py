"""API request/response schemas."""
from pydantic import BaseModel


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: list[ChatTurn] = []


class SourceRefModel(BaseModel):
    tag: str
    source_path: str
    source_type: str
    pdf_page_number: int | None = None
    code_function_name: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRefModel]
    citations_total: int
    citations_invalid: int
    latency_ms: float
