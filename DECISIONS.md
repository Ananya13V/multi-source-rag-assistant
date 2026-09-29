## Decision: LangChain over LlamaIndex
**Date:** 2026-08-17
**Context:** Choosing orchestration framework for multi-source RAG assistant.
**Decision:** LangChain
**Rationale:** Broader multi-source document loader ecosystem; better fit 
for heterogeneous ingestion (docs, code, web) vs. LlamaIndex's strength 
in pure retrieval-optimization.
**Alternatives considered:** LlamaIndex (stronger indexing/retrieval 
primitives, but narrower loader support out of the box)

## Decision: Flat metadata schema with source-type prefixes
**Date:** 2026-08-17
**Context:** Chunks come from two structurally different sources (PDF, code), 
each needing different metadata fields, stored as JSONB in Supabase.
**Decision:** Flat schema — all fields at top level, prefixed by source type 
(pdf_page_number, code_function_name, etc.) rather than nested sub-objects.
**Rationale:** Simpler, consistent SQL filtering (no null-checking nested paths); 
less null-handling surface area in Python retrieval code. Tradeoff: schema 
grows with each new source type added, but acceptable at current scope 
(2 source types, ~10-50 docs).
**Alternatives considered:** Nested metadata by source_type — cleaner-looking 
but requires conditional logic (CASE statements in SQL, nested dict access 
in Python) every time mixed-source chunks are queried together.

## Decision: Deterministic chunk_id generation
**Date:** 2026-08-17
**Context:** Need unique IDs for chunks that support safe re-ingestion.
**Decision:** chunk_id = hash(source_path + chunk_index), not random UUID.
**Rationale:** Re-running ingestion on an updated file produces the same 
chunk_ids for unchanged content, enabling upsert-based re-indexing instead 
of full wipe-and-reload. Paired with a content_hash field to detect which 
chunks actually changed and skip re-embedding the rest.
**Alternatives considered:** Random UUID — simpler to generate but loses 
the ability to detect/update specific chunks on re-ingestion.

## Decision: Supabase + pgvector over Azure AI Search / Qdrant for vector storage
**Date:** 2026-09-29
**Context:** Needed a vector store for embeddings; requirements.txt/.env.example 
already pointed at Supabase before this build started.
**Decision:** Supabase (managed Postgres) + the pgvector extension.
**Rationale:** One database holds both the vector column and the relational 
metadata used for hybrid search (see next decision), so retrieval is plain 
SQL/RPC with no second system to keep in sync. Free tier is sufficient at 
this project's scale (~10-50 docs). Azure AI Search and Qdrant are both 
stronger at large-scale ANN search, but that headroom isn't needed here.
**Alternatives considered:** Azure AI Search (native hybrid search, but a 
separate service from the metadata store, and paid at any meaningful scale); 
Qdrant (good local/open-source option, but adds a service to self-host for 
a personal project already using Supabase for everything else).

## Decision: Hybrid search via Reciprocal Rank Fusion, not a single ranked list
**Date:** 2026-09-29
**Context:** Pure vector similarity misses exact keyword matches (an error 
code, a function name pasted verbatim); pure full-text search misses 
paraphrases and cross-source concept matches.
**Decision:** Run vector similarity (pgvector, `match_documents_vector`) and 
Postgres full-text search (`tsvector`/GIN, `match_documents_text`) as two 
independent ranked lists, then fuse them in Python with RRF 
(score = sum of 1/(k+rank), k=60).
**Rationale:** RRF needs no score normalization between the two very 
different scoring scales (cosine similarity vs. `ts_rank`), and fusing in 
Python rather than SQL keeps the fusion logic covered by unit tests that 
don't require a live database. The SQL RPCs stay narrow (one retrieval leg 
each), which also makes each easy to test/tune independently.
**Alternatives considered:** A single SQL query combining both scores with 
a fixed weighted sum — simpler, but weights need re-tuning per scale change 
in either score, and it can't be unit-tested without a live Postgres/pgvector 
instance.

## Decision: Citation-tag grounding + a validator, not just "cite your sources" prompting
**Date:** 2026-09-29
**Context:** Resume goal is to reduce hallucinated citations; prompting 
alone doesn't guarantee the model never invents a source.
**Decision:** Retrieved chunks are numbered `[S1]`..`[Sn]` in the prompt; the 
model is instructed to tag every factual sentence with the source(s) it 
came from. `app/generation/citations.py` then parses the answer, and any 
`[Sn]` tag citing a number outside the actual retrieved set is stripped 
before the answer is returned, with the invalid/valid counts tracked as a 
hallucination rate.
**Rationale:** This is a deterministic backstop, not another LLM call — it 
can't itself hallucinate, and it gives a concrete metric (hallucination 
rate) to report from `eval/run_ragas.py` rather than relying solely on 
RAGAS's faithfulness score.
**Alternatives considered:** A second LLM call to fact-check the first 
answer's citations — more thorough, but doubles generation latency and 
cost, and is itself imperfect at self-grading. May revisit if the 
regex-based check proves too coarse (e.g. it can't tell *which specific 
claim* in a sentence is unsupported, only that a cited source number is 
invalid).

## Decision: Word-count-based chunk windows, not a model-specific tokenizer
**Date:** 2026-09-29
**Context:** Considered `tiktoken` for token-exact chunk sizing.
**Decision:** Measure chunk size/overlap in whitespace-delimited words 
instead.
**Rationale:** `tiktoken` downloads its BPE vocab file from a Microsoft blob 
endpoint on first use; that's a network dependency with no benefit here, 
since ingestion doesn't need token-exact precision — just a consistent, 
roughly proportional chunk size across PDF prose and source code. Word 
count is offline, dependency-free, and good enough for that purpose.
**Alternatives considered:** `tiktoken` (token-exact, matches what the 
embedding/chat model actually sees, but adds a network dependency and a 
vocab-file cache for marginal accuracy gain at this scope).