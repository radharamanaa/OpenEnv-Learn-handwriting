#!/usr/bin/env python3
"""
Smoke test: one OpenEnv episode with a hard-coded stroke (no TRL required).

From the parent of the ``learn_handwriting`` package (e.g. ``.../scaler_8_april``)::

  export PYTHONPATH=/path/to/that/parent
  export OPENENV_BASE_URL=http://127.0.0.1:8000
  python learn_handwriting/datagen_sft/grpo_smoke_episode.py

Use this to verify connectivity and action/reward shape before GRPO.

Set ``LEARN_HANDWRITING_ENV_LOG=1`` for detailed WebSocket logs (see ``learn_handwriting.client``).
"""

from __future__ import annotations

import os
import sys

# Parent of the package dir so ``import learn_handwriting`` resolves (like pip -e .).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from learn_handwriting import (
    LearnHandwritingAction,
    LearnHandwritingEnv,
    configure_openenv_ws_logging,
)


def main() -> None:
    configure_openenv_ws_logging()

    base = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000")
    task = os.environ.get("GRPO_SMOKE_TASK", "easy")
    char = os.environ.get("GRPO_SMOKE_CHAR", "L")
    print(f"OPENENV_BASE_URL={base} task={task} character={char}", flush=True)

    raw = LearnHandwritingEnv(base_url=base)
    with raw.sync() as env:
        r0 = env.reset(task=task, character=char)
        if r0 is None or r0.observation is None:
            raise SystemExit("reset failed: no observation")
        print("After reset: match% =", r0.observation.match_percentage, flush=True)

        # One diagonal line (adjust if you need a visible stroke on the glyph)
        act = LearnHandwritingAction(
            action_type="line",
            x1=10, y1=10, x2=50, y2=50, x3=None, y3=None, radius=None, rx=None, ry=None
        )
        r1 = env.step(act)
        if r1 and r1.observation:
            print(
                "After one step: match% =",
                r1.observation.match_percentage,
                "reward =",
                r1.reward,
                "done =",
                r1.done,
                flush=True,
            )
        else:
            raise SystemExit("step failed")


if __name__ == "__main__":
    main()
