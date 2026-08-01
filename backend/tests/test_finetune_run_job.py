"""
Tests for backend/finetune/run_job.py — the OpenAI fine-tuning job wrapper.

All offline: the OpenAI client is mocked. Covers the exact bug that shipped
initially — client.files.create(purpose="fine-tuning") is REJECTED by the
real API (valid values: assistants|batch|fine-tune|vision|user_data|evals),
so purpose must be "fine-tune" (no "-ing").

Run from repo root:
    pytest backend/tests/test_finetune_run_job.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

from backend.finetune.run_job import create_job, poll_until_done, upload_file


class TestUploadFile:

    def test_purpose_is_fine_tune_not_fine_tuning(self):
        # Regression test: the OpenAI Files API's `purpose` enum is
        # ['assistants', 'batch', 'fine-tune', 'vision', 'user_data', 'evals']
        # — "fine-tuning" (with -ing) is rejected with a 400.
        client = MagicMock()
        client.files.create.return_value = MagicMock(id="file-abc")
        with patch("builtins.open", mock_open(read_data=b"{}")):
            file_id = upload_file(client, "train.jsonl")
        assert file_id == "file-abc"
        assert client.files.create.call_args.kwargs["purpose"] == "fine-tune"


class TestCreateJob:

    def test_minimal_call(self):
        client = MagicMock()
        client.fine_tuning.jobs.create.return_value = MagicMock(id="ftjob-1")
        job_id = create_job(client, "file-abc", "gpt-4o-mini-2024-07-18")
        assert job_id == "ftjob-1"
        kwargs = client.fine_tuning.jobs.create.call_args.kwargs
        assert kwargs["training_file"] == "file-abc"
        assert kwargs["model"] == "gpt-4o-mini-2024-07-18"
        assert "validation_file" not in kwargs
        assert "suffix" not in kwargs

    def test_validation_file_and_suffix_forwarded(self):
        client = MagicMock()
        client.fine_tuning.jobs.create.return_value = MagicMock(id="ftjob-2")
        create_job(client, "file-abc", "gpt-4o-mini-2024-07-18",
                   validation_file_id="file-val", suffix="bird-simple")
        kwargs = client.fine_tuning.jobs.create.call_args.kwargs
        assert kwargs["validation_file"] == "file-val"
        assert kwargs["suffix"] == "bird-simple"


class TestPollUntilDone:

    def test_returns_on_terminal_state(self):
        client = MagicMock()
        client.fine_tuning.jobs.retrieve.return_value = MagicMock(status="succeeded")
        client.fine_tuning.jobs.list_events.return_value = MagicMock(data=[])
        with patch("time.sleep"):
            job = poll_until_done(client, "ftjob-1", poll_seconds=0)
        assert job.status == "succeeded"

    def test_polls_until_terminal(self):
        client = MagicMock()
        running = MagicMock(status="running")
        done = MagicMock(status="succeeded")
        client.fine_tuning.jobs.retrieve.side_effect = [running, running, done]
        client.fine_tuning.jobs.list_events.return_value = MagicMock(data=[])
        with patch("time.sleep"):
            job = poll_until_done(client, "ftjob-1", poll_seconds=0)
        assert job.status == "succeeded"
        assert client.fine_tuning.jobs.retrieve.call_count == 3
