"""CLI entry point for ingestion.

Usage:
    python scripts/ingest_cli.py data/sample_docs
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.pipeline import ingest_path  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest technical docs (PDF + code) into Supabase/pgvector.")
    parser.add_argument("path", type=Path, help="Directory to ingest recursively.")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Path not found: {args.path}")

    summary = ingest_path(args.path)
    print(summary)


if __name__ == "__main__":
    main()
