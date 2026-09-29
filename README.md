# Multi-Source Technical Docs RAG Assistant

A retrieval-augmented generation assistant over heterogeneous technical
documentation - PDFs and source code - built with LangChain, Azure OpenAI,
and Supabase/pgvector, served via FastAPI and evaluated with RAGAS.

Ask it a question and it answers **only** from what's actually indexed,
with every claim tagged back to the source chunk that supports it
(`[S1]`, `[S2]`, ...). Citations the model invents that don't map to a real
retrieved source are detected and stripped before the answer is returned.

## Architecture

```
data/sample_docs/          sample corpus: PDFs + Python source files
        |
        v
app/ingestion/             load -> chunk (source-aware) -> embed -> upsert
  chunking.py                 PDF: page-based windows
                               code: one chunk per top-level function/class (via ast),
                                     falling back to fixed windows for other languages
  identity.py                 chunk_id = sha256(source_path + chunk_index)   (stable across re-ingestion)
                               content_hash = sha256(content)                (skip re-embedding unchanged chunks)
  pipeline.py                  orchestrates the above, diffing against Supabase before calling the embeddings API
        |
        v
Supabase / Postgres (pgvector)
  sql/schema.sql             flat, source-type-prefixed metadata columns (pdf_*, code_*)
                               + ivfflat vector index + GIN full-text index
                               + two narrow RPCs: match_documents_vector, match_documents_text
        |
        v
app/retrieval/              hybrid search: vector leg + full-text leg, fused with
  hybrid_search.py             Reciprocal Rank Fusion (RRF) - pure Python, unit-tested without a DB
  query_rewrite.py             condenses a follow-up question + chat history into a standalone query
        |
        v
app/generation/              retrieved chunks become numbered [Sn] sources in the prompt;
  prompt.py / answer.py        the model must cite one for every factual sentence
  citations.py                  citation validator strips any [Sn] tag outside the retrieved set
        |
        v
app/main.py (FastAPI)        POST /query, GET /health, per-request latency logged + returned

eval/                        RAGAS evaluation harness (faithfulness, answer relevancy,
                               context precision/recall) + our own citation-hallucination-rate and latency stats
```

Design rationale for each of these choices lives in `DECISIONS.md`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in Azure OpenAI + Supabase credentials
```

Then, in the Supabase SQL editor (or via `psql "$SUPABASE_DB_URL"`), run
`sql/schema.sql` once to create the `documents` table, its indexes, and the
two retrieval RPCs.

## Running it

```bash
# 1. Generate the sample corpus (4 synthetic PDFs) - already checked into
#    data/sample_docs/pdf/, re-run only if you edit scripts/generate_sample_data.py
python scripts/generate_sample_data.py

# 2. Ingest the sample corpus (PDFs + code) into Supabase
python scripts/ingest_cli.py data/sample_docs

# 3. Serve
uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/query -X POST -H 'content-type: application/json' \
  -d '{"question": "What is the retry backoff policy?"}' | python3 -m json.tool
```

Example response shape:

```json
{
  "answer": "The base delay is 1 second, doubling on each attempt, capped at 30 seconds, for up to 5 attempts [S1]. The same policy is implemented in the retry decorator, which only retries on transient status codes [S2].",
  "sources": [
    {"tag": "S1", "source_path": "data/sample_docs/pdf/resilience_patterns.pdf", "source_type": "pdf", "pdf_page_number": 1},
    {"tag": "S2", "source_path": "data/sample_docs/code/retry.py", "source_type": "code", "code_function_name": "with_retries"}
  ],
  "citations_total": 2,
  "citations_invalid": 0,
  "latency_ms": 812.4
}
```

## Evaluation

`eval/testset.jsonl` holds 13 test queries with ground-truth answers drawn
from the sample corpus (12 answerable, 1 deliberately not - to check the
assistant declines instead of guessing). After ingesting the sample corpus:

```bash
python eval/run_ragas.py
```

This computes RAGAS faithfulness / answer relevancy / context precision /
context recall (each itself an LLM judgment, so it costs real Azure OpenAI
calls), plus our own average citation-hallucination-rate and latency, and
writes `eval/results.json`.

**Note:** this repo ships the pipeline and the eval harness, but not
pre-computed numbers - those depend on your own Azure OpenAI deployment and
model choice. Run `eval/run_ragas.py` against your own setup and use the
`results.json` it produces for any metrics you want to put on a resume or
report.

## Testing

```bash
pytest
```

The test suite (chunk-id determinism, source-aware chunking, RRF fusion
math, citation validation) runs offline - no Supabase or Azure OpenAI
credentials required.

## Tech stack

LangChain · Azure OpenAI (chat + embeddings) · Supabase (Postgres + pgvector)
· FastAPI · RAGAS · pypdf
