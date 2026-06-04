"""Upload local documents to S3 and trigger a Bedrock Knowledge Base sync.

Usage:
    python -m ingestion.sync_knowledge_base ./docs/corpus/
"""

from __future__ import annotations

import sys


def main(corpus_dir: str) -> None:
    """TODO:
    1. Upload files under `corpus_dir` to the docs S3 bucket.
    2. Call bedrock-agent `start_ingestion_job` for the data source.
    3. Poll until the ingestion job completes.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./docs/corpus/")
