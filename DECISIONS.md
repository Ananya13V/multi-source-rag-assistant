## Decision: LangChain over LlamaIndex
**Date:** 2026-08-17
**Context:** Choosing orchestration framework for multi-source RAG assistant.
**Decision:** LangChain
**Rationale:** Broader multi-source document loader ecosystem; better fit 
for heterogeneous ingestion (docs, code, web) vs. LlamaIndex's strength 
in pure retrieval-optimization.
**Alternatives considered:** LlamaIndex (stronger indexing/retrieval 
primitives, but narrower loader support out of the box)

