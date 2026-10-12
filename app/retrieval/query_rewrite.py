"""Condense a follow-up question + chat history into a standalone query.

Skipped entirely for the common single-turn case (empty history) so most
queries never pay for the extra LLM round-trip.
"""
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings

_SYSTEM_PROMPT = (
    "Rewrite the user's latest question as a standalone question that makes "
    "sense without the prior conversation. Preserve their intent exactly. "
    "Do not answer it, and do not add information that wasn't implied by "
    "the conversation. Reply with only the rewritten question."
)


@lru_cache
def _get_llm() -> ChatOllama:
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=0,
    )


def rewrite_query(question: str, chat_history: list[tuple[str, str]]) -> str:
    if not chat_history:
        return question

    transcript = "\n".join(f"{role}: {text}" for role, text in chat_history)
    prompt = f"Conversation so far:\n{transcript}\n\nLatest question: {question}"
    response = _get_llm().invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    rewritten = response.content.strip() if isinstance(response.content, str) else question
    return rewritten or question
