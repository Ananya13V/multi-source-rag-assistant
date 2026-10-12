"""RAGAS evaluation over eval/testset.jsonl, run against the live pipeline.

Requires a populated Supabase table (run scripts/ingest_cli.py first), a
working .env (Supabase credentials), and Ollama running locally with the
configured chat + embedding models pulled (`ollama pull llama3.2`,
`ollama pull nomic-embed-text`). Not runnable without Ollama serving:
RAGAS's faithfulness/context-precision/context-recall/answer-relevancy
metrics are themselves LLM judgments, so this call needs a working local model.

    python scripts/ingest_cli.py data/sample_docs
    python eval/run_ragas.py

Writes eval/results.json (per-metric means) and prints a summary table,
alongside our own citation-hallucination-rate and latency stats, which
aren't RAGAS metrics but come straight out of app.generation.answer.
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from langchain_ollama import ChatOllama  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embeddings_client  # noqa: E402
from app.generation.answer import generate_answer  # noqa: E402
from app.retrieval.hybrid_search import hybrid_search  # noqa: E402

TESTSET_PATH = Path(__file__).parent / "testset.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_testset() -> list[dict]:
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline_over_testset(testset: list[dict]) -> tuple[dict, list[float], list[float]]:
    """Returns (ragas_dataset_dict, latencies_ms, hallucination_rates)."""
    questions, answers, contexts, ground_truths = [], [], [], []
    latencies_ms: list[float] = []
    hallucination_rates: list[float] = []

    for item in testset:
        start = time.perf_counter()
        chunks = hybrid_search(item["question"])
        result = generate_answer(item["question"], chunks)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        hallucination_rates.append(result.citation_report.hallucination_rate)

        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append([c.content for c in chunks])
        ground_truths.append(item["ground_truth"])

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    return dataset_dict, latencies_ms, hallucination_rates


def main() -> None:
    testset = load_testset()
    print(f"Running {len(testset)} test queries through the live pipeline...")
    dataset_dict, latencies_ms, hallucination_rates = run_pipeline_over_testset(testset)

    ragas_dataset = Dataset.from_dict(dataset_dict)

    settings = get_settings()
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings_client())

    ragas_result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    summary = {
        "n_queries": len(testset),
        "ragas": ragas_result.to_pandas()[
            ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        ].mean().to_dict(),
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms),
        "avg_citation_hallucination_rate": sum(hallucination_rates) / len(hallucination_rates),
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
