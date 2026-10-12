"""Ingestion orchestration: load -> chunk -> embed (only what changed) -> upsert.

Re-running this over an unchanged directory is a no-op against the
embeddings API: chunk_ids are deterministic (see identity.py) and rows whose
content_hash already matches what's in Supabase are skipped before
embedding, per DECISIONS.md.
"""
import logging
from dataclasses import asdict
from pathlib import Path

from app.config import get_settings
from app.db import get_supabase_client
from app.embeddings import embed_texts
from app.ingestion.chunking import chunk_file
from app.ingestion.identity import make_chunk_id, make_content_hash

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".py", ".js", ".ts", ".java", ".go", ".md"}
_BATCH_SIZE = 100


def discover_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _existing_content_hashes(client, table: str, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    existing: dict[str, str] = {}
    for i in range(0, len(chunk_ids), _BATCH_SIZE):
        batch = chunk_ids[i : i + _BATCH_SIZE]
        resp = (
            client.table(table)
            .select("chunk_id, content_hash")
            .in_("chunk_id", batch)
            .execute()
        )
        for row in resp.data or []:
            existing[row["chunk_id"]] = row["content_hash"]
    return existing


def ingest_path(root: Path) -> dict:
    settings = get_settings()
    client = get_supabase_client()

    files = discover_files(root)
    rows = []
    for path in files:
        for record in chunk_file(path, settings.chunk_size_words, settings.chunk_overlap_words):
            row = asdict(record)
            row["chunk_id"] = make_chunk_id(record.source_path, record.chunk_index)
            row["content_hash"] = make_content_hash(record.content)
            rows.append(row)

    existing_hashes = _existing_content_hashes(
        client, settings.documents_table, [r["chunk_id"] for r in rows]
    )

    to_embed = [r for r in rows if existing_hashes.get(r["chunk_id"]) != r["content_hash"]]
    unchanged_count = len(rows) - len(to_embed)

    embeddings: list[list[float]] = []
    for i in range(0, len(to_embed), _BATCH_SIZE):
        batch = to_embed[i : i + _BATCH_SIZE]
        embeddings.extend(embed_texts([r["content"] for r in batch]))

    for row, embedding in zip(to_embed, embeddings):
        row["embedding"] = embedding

    if to_embed:
        for i in range(0, len(to_embed), _BATCH_SIZE):
            batch = to_embed[i : i + _BATCH_SIZE]
            client.table(settings.documents_table).upsert(batch, on_conflict="chunk_id").execute()

    logger.info(
        "Ingested %d file(s): %d chunk(s) total, %d embedded/updated, %d unchanged (skipped).",
        len(files),
        len(rows),
        len(to_embed),
        unchanged_count,
    )
    return {
        "files_processed": len(files),
        "chunks_total": len(rows),
        "chunks_embedded": len(to_embed),
        "chunks_unchanged": unchanged_count,
    }
