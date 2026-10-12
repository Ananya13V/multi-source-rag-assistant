-- Multi-Source Technical Docs RAG Assistant
-- Run this once against your Supabase Postgres instance (SQL editor or psql).
--
-- Metadata design follows DECISIONS.md: a flat schema with source-type
-- prefixed columns (pdf_*, code_*) rather than nested JSONB, so retrieval
-- filters stay plain SQL with no null-path traversal.

create extension if not exists vector;

create table if not exists documents (
    chunk_id         text primary key,            -- sha256(source_path + chunk_index), stable across re-ingestion
    content_hash     text not null,                -- sha256(chunk_text), used to skip re-embedding unchanged chunks
    source_path      text not null,
    source_type      text not null check (source_type in ('pdf', 'code')),
    chunk_index      integer not null,
    content          text not null,

    -- pdf_* fields (null for non-pdf sources)
    pdf_page_number  integer,
    pdf_title        text,

    -- code_* fields (null for non-code sources)
    code_language      text,
    code_function_name text,
    code_start_line    integer,
    code_end_line      integer,

    embedding        vector(768) not null,
    content_tsv      tsvector generated always as (to_tsvector('english', content)) stored,

    created_at       timestamptz not null default now(),
    updated_at       timestamptz not null default now()
);

-- Vector similarity search (approximate nearest neighbor).
create index if not exists documents_embedding_idx
    on documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Full-text search leg of hybrid retrieval.
create index if not exists documents_content_tsv_idx
    on documents using gin (content_tsv);

create index if not exists documents_source_path_idx on documents (source_path);
create index if not exists documents_source_type_idx on documents (source_type);

-- Keep updated_at current on upsert.
create or replace function set_updated_at() returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists documents_set_updated_at on documents;
create trigger documents_set_updated_at
    before update on documents
    for each row execute function set_updated_at();

-- ---------------------------------------------------------------------
-- Two narrow RPCs, one per retrieval leg. Fusion (Reciprocal Rank Fusion)
-- happens in Python (app/retrieval/hybrid_search.py) so it's unit-testable
-- without a live database.
-- ---------------------------------------------------------------------

create or replace function match_documents_vector(
    query_embedding vector(768),
    match_count integer default 20
)
returns table (
    chunk_id text,
    content text,
    source_path text,
    source_type text,
    pdf_page_number integer,
    code_function_name text,
    similarity float
)
language sql stable
as $$
    select
        d.chunk_id,
        d.content,
        d.source_path,
        d.source_type,
        d.pdf_page_number,
        d.code_function_name,
        1 - (d.embedding <=> query_embedding) as similarity
    from documents d
    order by d.embedding <=> query_embedding
    limit match_count;
$$;

create or replace function match_documents_text(
    query_text text,
    match_count integer default 20
)
returns table (
    chunk_id text,
    content text,
    source_path text,
    source_type text,
    pdf_page_number integer,
    code_function_name text,
    rank float
)
language sql stable
as $$
    select
        d.chunk_id,
        d.content,
        d.source_path,
        d.source_type,
        d.pdf_page_number,
        d.code_function_name,
        ts_rank(d.content_tsv, websearch_to_tsquery('english', query_text)) as rank
    from documents d
    where d.content_tsv @@ websearch_to_tsquery('english', query_text)
    order by rank desc
    limit match_count;
$$;
