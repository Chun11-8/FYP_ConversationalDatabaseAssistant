"""
backend/finetune/run_job.py — upload a fine-tuning JSONL and run the OpenAI
supervised fine-tuning job, polling to completion.

Requires OPENAI_API_KEY in the environment or backend/config/secrets.env
(reuses load_config() exactly like every other script in this repo).

Usage
-----
    python -m backend.finetune.run_job \\
        --training-file backend/finetune/train.jsonl \\
        --validation-file backend/finetune/train.eval.jsonl \\
        --base-model gpt-4o-mini-2024-07-18

On success, prints the fine_tuned_model id — pass it straight to any eval
CLI's --model flag, or add it to config.yaml's llm.selectable_models to
offer it in the frontend's model picker (Issue #52).

See backend/finetune/README.md for cost expectations and how to evaluate
the result once it's ready.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Terminal job states (per the OpenAI fine-tuning API).
_DONE_STATES = {"succeeded", "failed", "cancelled"}


def upload_file(client, path: str) -> str:
    """Upload a JSONL file for fine-tuning and return its file id."""
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tuning")
    logger.info("Uploaded %s -> file id %s", path, uploaded.id)
    return uploaded.id


def create_job(
    client,
    training_file_id: str,
    base_model: str,
    validation_file_id: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Create the fine-tuning job and return its job id."""
    kwargs = {"training_file": training_file_id, "model": base_model}
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
    if suffix:
        kwargs["suffix"] = suffix
    job = client.fine_tuning.jobs.create(**kwargs)
    logger.info("Created fine-tuning job %s (base model: %s)", job.id, base_model)
    return job.id


def poll_until_done(client, job_id: str, poll_seconds: float = 20.0):
    """Block, printing new events, until the job reaches a terminal state."""
    seen_event_ids = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=20)
        for event in reversed(events.data):
            if event.id not in seen_event_ids:
                seen_event_ids.add(event.id)
                logger.info("[%s] %s", event.level, event.message)
        if job.status in _DONE_STATES:
            return job
        time.sleep(poll_seconds)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Upload training data and run an OpenAI fine-tuning job"
    )
    parser.add_argument("--training-file", required=True, help="JSONL from prepare_data.py")
    parser.add_argument("--validation-file", default=None,
                        help="Optional held-out JSONL (prepare_data.py's --split-eval output)")
    parser.add_argument("--base-model", default="gpt-4o-mini-2024-07-18",
                        help="Base model to fine-tune (default: gpt-4o-mini-2024-07-18)")
    parser.add_argument("--suffix", default=None,
                        help="Optional suffix appended to the fine-tuned model's name")
    parser.add_argument("--poll-seconds", type=float, default=20.0,
                        help="Seconds between status polls while the job runs")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit the job and exit immediately instead of polling")
    args = parser.parse_args(argv)

    if not Path(args.training_file).exists():
        parser.error(f"Training file not found: {args.training_file}")

    from backend.config.config_loader import load_config
    from openai import OpenAI

    config = load_config()
    client = OpenAI(api_key=config["llm"]["api_key"])

    training_file_id = upload_file(client, args.training_file)
    validation_file_id = (
        upload_file(client, args.validation_file) if args.validation_file else None
    )
    job_id = create_job(
        client, training_file_id, args.base_model,
        validation_file_id=validation_file_id, suffix=args.suffix,
    )

    if args.no_wait:
        print(f"Job submitted: {job_id} (not waiting — check status with "
              f"`client.fine_tuning.jobs.retrieve('{job_id}')`)")
        return 0

    job = poll_until_done(client, job_id, poll_seconds=args.poll_seconds)

    if job.status != "succeeded":
        logger.error("Fine-tuning job ended with status: %s", job.status)
        return 1

    print("\n" + "─" * 70)
    print(f"  Fine-tuning SUCCEEDED: {job.fine_tuned_model}")
    print("─" * 70)
    print("Use it immediately:")
    print(f"  python -m backend.eval.bird ... --model {job.fine_tuned_model}")
    print("Or add it to backend/config/config.yaml's llm.selectable_models "
          "to offer it in the frontend's model picker.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
