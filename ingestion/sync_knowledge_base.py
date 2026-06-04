"""Upload local documents to S3 and trigger a Bedrock Knowledge Base sync.

Usage:
    python -m ingestion.sync_knowledge_base ./docs/corpus/

Configuration is read from the environment (see app/config.py / .env):
    DOCS_BUCKET        - S3 bucket holding the source documents
    KNOWLEDGE_BASE_ID  - the Bedrock Knowledge Base id
    DATA_SOURCE_ID     - the Knowledge Base's S3 data source id
    AWS_REGION

Writing and committing this file costs nothing. It only incurs AWS charges
when run against a real, deployed Knowledge Base (embeddings on ingest, plus
the vector store behind the KB).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import boto3

from app.config import settings

# File types the Bedrock Knowledge Base S3 data source can ingest.
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".csv",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
}
TERMINAL_STATES = {"COMPLETE", "FAILED"}
POLL_SECONDS = 10


def _s3() -> Any:
    return boto3.client("s3", region_name=settings.aws_region)


def _agent() -> Any:
    return boto3.client("bedrock-agent", region_name=settings.aws_region)


def upload_documents(corpus_dir: Path) -> int:
    """Upload every supported document under corpus_dir to the docs bucket."""
    files = [
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if not files:
        raise SystemExit(f"No ingestible documents found in {corpus_dir}")

    s3 = _s3()
    for path in files:
        key = path.relative_to(corpus_dir).as_posix()
        print(f"  uploading {key}")
        s3.upload_file(str(path), settings.docs_bucket, key)
    print(f"Uploaded {len(files)} file(s) to s3://{settings.docs_bucket}")
    return len(files)


def start_sync() -> str:
    """Kick off an ingestion job and return its id."""
    response = _agent().start_ingestion_job(
        knowledgeBaseId=settings.knowledge_base_id,
        dataSourceId=settings.data_source_id,
    )
    job_id: str = response["ingestionJob"]["ingestionJobId"]
    print(f"Started ingestion job {job_id}")
    return job_id


def wait_for_job(job_id: str) -> str:
    """Poll the ingestion job until it reaches a terminal state."""
    agent = _agent()
    while True:
        job = agent.get_ingestion_job(
            knowledgeBaseId=settings.knowledge_base_id,
            dataSourceId=settings.data_source_id,
            ingestionJobId=job_id,
        )["ingestionJob"]
        status = job["status"]
        print(f"  job {job_id}: {status}")
        if status in TERMINAL_STATES:
            stats = job.get("statistics")
            if stats:
                print(f"  statistics: {stats}")
            return status
        time.sleep(POLL_SECONDS)


def sync(corpus_dir: Path) -> None:
    """Upload documents, start an ingestion job, and wait for it to finish."""
    upload_documents(corpus_dir)
    job_id = start_sync()
    status = wait_for_job(job_id)
    if status == "FAILED":
        raise SystemExit("Ingestion job FAILED - see the Bedrock console for details.")
    print("Knowledge base sync complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload docs to S3 and sync the Bedrock Knowledge Base."
    )
    parser.add_argument(
        "corpus_dir",
        nargs="?",
        default="docs/corpus",
        help="Directory of documents to ingest (default: docs/corpus).",
    )
    args = parser.parse_args()
    sync(Path(args.corpus_dir))


if __name__ == "__main__":
    main()
