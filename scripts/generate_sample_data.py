"""Generates the demo corpus under data/sample_docs/pdf/.

Run once to (re)create the sample PDFs from the prose below, so the corpus
ships as source text in this repo instead of opaque binary blobs.

    python scripts/generate_sample_data.py
"""
import sys
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "pdf"

# Each entry is one PDF: title, then a list of pages, each page a list of
# paragraphs. Kept short and synthetic (no scraped content) but structured
# like real internal engineering docs.
DOCUMENTS: dict[str, list[list[str]]] = {
    "resilience_patterns": [
        [
            "Resilience Patterns for HTTP Clients",
            "This guide covers the retry and backoff conventions used by "
            "internal services when calling downstream APIs.",
            "All outbound HTTP calls must use exponential backoff with jitter. "
            "The base delay is 1 second, doubling on each attempt, capped at "
            "30 seconds, for a maximum of 5 attempts before the call is "
            "considered failed.",
        ],
        [
            "Only retry on transient failures: HTTP 429, 502, 503, 504, and "
            "connection timeouts. Do not retry on 4xx client errors other "
            "than 429, since retrying a malformed request will never "
            "succeed.",
            "Jitter is applied by multiplying the computed delay by a random "
            "factor between 0.8 and 1.2. This prevents thundering-herd retry "
            "storms when many clients fail at the same time.",
        ],
        [
            "Circuit breakers sit in front of the retry layer. After 5 "
            "consecutive failures to a given host, the breaker opens for 60 "
            "seconds and short-circuits further calls immediately, returning "
            "a fast failure instead of waiting for a timeout.",
        ],
    ],
    "rate_limiting": [
        [
            "API Rate Limiting Design",
            "Our public API enforces per-API-key rate limits using a token "
            "bucket algorithm rather than a fixed window counter.",
            "Each key is allocated a bucket with a capacity of 100 tokens, "
            "refilling at 10 tokens per second. A request consumes one "
            "token; if the bucket is empty the request is rejected with "
            "HTTP 429 and a Retry-After header.",
        ],
        [
            "Token buckets were chosen over fixed windows because they "
            "smooth out bursts: a client that has been idle can burst up to "
            "its full bucket capacity, but sustained traffic is capped at "
            "the refill rate rather than allowing a new burst at every "
            "window boundary.",
            "Rate limit state is stored in Redis with a Lua script so the "
            "check-and-decrement is atomic across concurrent requests for "
            "the same key.",
        ],
    ],
    "vector_search_concepts": [
        [
            "Vector Search and Hybrid Retrieval",
            "This document explains how semantic (vector) search is combined "
            "with keyword search in our retrieval pipeline.",
            "Embeddings are stored in Postgres using the pgvector extension. "
            "An ivfflat index approximates nearest-neighbor search over "
            "cosine distance, trading a small amount of recall for much "
            "faster queries at scale compared to an exact scan.",
        ],
        [
            "Pure vector search misses exact keyword matches, such as an "
            "error code or a function name a user pastes verbatim. To cover "
            "this, results from a Postgres full-text search (tsvector/GIN "
            "index) are combined with the vector results.",
            "The two ranked lists are merged with Reciprocal Rank Fusion "
            "(RRF): each document's score is the sum of 1/(k + rank) across "
            "every list it appears in, with k=60. Documents that rank well "
            "in both searches float to the top even if neither list alone "
            "ranked them first.",
        ],
    ],
    "deployment_guide": [
        [
            "Service Deployment and Timeouts",
            "This guide documents default timeout and worker configuration "
            "for services deployed behind our API gateway.",
            "The gateway enforces a hard 30 second upstream timeout. "
            "Services should configure their own internal timeouts well "
            "below that, typically 10-15 seconds, so a service can return "
            "its own error response instead of the gateway returning a "
            "generic 504.",
        ],
        [
            "Worker processes are sized based on expected concurrency: CPU-"
            "bound services run one worker per core, while I/O-bound "
            "services (most API services) run 2-4 workers per core since "
            "workers spend most of their time waiting on network calls.",
        ],
    ],
}


def write_pdf(name: str, pages: list[list[str]]) -> Path:
    path = OUTPUT_DIR / f"{name}.pdf"
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER

    for page_paragraphs in pages:
        y = height - 72
        for paragraph in page_paragraphs:
            for line in textwrap.wrap(paragraph, width=90):
                c.drawString(72, y, line)
                y -= 14
            y -= 10  # blank line between paragraphs
        c.showPage()
    c.save()
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCUMENTS.items():
        path = write_pdf(name, pages)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
