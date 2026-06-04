"""Unit tests for the ingestion pipeline.

No real AWS calls and no real waiting: the boto3 clients are mocked and the
poll interval is patched to zero.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import ingestion.sync_knowledge_base as sk


def _fake_settings():
    return SimpleNamespace(
        aws_region="us-east-1",
        docs_bucket="test-bucket",
        knowledge_base_id="KB123",
        data_source_id="DS123",
    )


def test_upload_skips_unsupported_files(monkeypatch, tmp_path):
    (tmp_path / "guide.md").write_text("hello")
    (tmp_path / "notes.txt").write_text("world")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")  # unsupported -> skipped

    s3 = MagicMock()
    monkeypatch.setattr(sk, "settings", _fake_settings())
    monkeypatch.setattr(sk, "_s3", lambda: s3)

    count = sk.upload_documents(tmp_path)

    assert count == 2
    assert s3.upload_file.call_count == 2


def test_sync_happy_path(monkeypatch, tmp_path):
    (tmp_path / "doc.md").write_text("content")

    s3 = MagicMock()
    agent = MagicMock()
    agent.start_ingestion_job.return_value = {
        "ingestionJob": {"ingestionJobId": "job-1"}
    }
    agent.get_ingestion_job.side_effect = [
        {"ingestionJob": {"status": "IN_PROGRESS"}},
        {"ingestionJob": {"status": "COMPLETE", "statistics": {"scanned": 1}}},
    ]
    monkeypatch.setattr(sk, "settings", _fake_settings())
    monkeypatch.setattr(sk, "_s3", lambda: s3)
    monkeypatch.setattr(sk, "_agent", lambda: agent)
    monkeypatch.setattr(sk, "POLL_SECONDS", 0)

    sk.sync(tmp_path)

    agent.start_ingestion_job.assert_called_once_with(
        knowledgeBaseId="KB123", dataSourceId="DS123"
    )
    assert agent.get_ingestion_job.call_count == 2


def test_sync_raises_on_failed_job(monkeypatch, tmp_path):
    (tmp_path / "doc.md").write_text("content")

    agent = MagicMock()
    agent.start_ingestion_job.return_value = {
        "ingestionJob": {"ingestionJobId": "job-x"}
    }
    agent.get_ingestion_job.return_value = {"ingestionJob": {"status": "FAILED"}}
    monkeypatch.setattr(sk, "settings", _fake_settings())
    monkeypatch.setattr(sk, "_s3", lambda: MagicMock())
    monkeypatch.setattr(sk, "_agent", lambda: agent)
    monkeypatch.setattr(sk, "POLL_SECONDS", 0)

    try:
        sk.sync(tmp_path)
        raise AssertionError("expected SystemExit on FAILED job")
    except SystemExit:
        pass
