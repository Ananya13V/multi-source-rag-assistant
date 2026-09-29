"""Deterministic chunk identity.

See DECISIONS.md: chunk_id = hash(source_path + chunk_index), not a random
UUID, so re-running ingestion on an unchanged file reproduces the same ids
and upserts cleanly instead of wiping and reloading. content_hash lets the
ingestion pipeline detect which specific chunks actually changed and skip
re-embedding the rest.
"""
import hashlib


def make_chunk_id(source_path: str, chunk_index: int) -> str:
    key = f"{source_path}:{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
