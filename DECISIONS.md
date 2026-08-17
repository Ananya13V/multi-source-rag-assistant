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