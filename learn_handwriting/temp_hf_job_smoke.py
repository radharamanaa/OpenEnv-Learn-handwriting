#!/usr/bin/env python3
"""
One-shot Hugging Face Job smoke test (hello world on cpu-basic).

Usage (from repo root, with hub logged in or HF_TOKEN in env):
  uv run python temp_hf_job_smoke.py

  # or
  export HF_TOKEN=hf_...
  python temp_hf_job_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

from huggingface_hub import fetch_job_logs, get_token, inspect_job, login, run_job


def main() -> int:
    if not os.environ.get("HF_TOKEN") and not get_token():
        login()
    job = run_job(
        image="python:3.12",
        command=["python", "-c", "print('Hello from Hugging Face Jobs')"],
        flavor="cpu-basic",
        timeout="5m",
    )
    print("Job:", job)
    if getattr(job, "url", None):
        print("URL:", job.url)
    jid = getattr(job, "id", None)
    if not jid:
        print("No job id; open URL above for logs if present.", file=sys.stderr)
        return 0
    for _ in range(30):
        st = inspect_job(job_id=jid).status
        if getattr(st, "stage", str(st)) in ("COMPLETED", "ERROR", "CANCELLED"):
            break
        time.sleep(2)
    print("--- logs ---")
    for line in fetch_job_logs(job_id=jid):
        print(line, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
