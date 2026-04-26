"""
GRPO reward functions for Learn Handwriting + OpenEnv.

TRL calls these with ``prompts``, ``completions``, and any extra columns from
``train_dataset`` as keyword args. Return one float per row.

See ``GRPO_PLAN.md`` for design notes (trajectory JSON in one completion).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# Episode cap aligned with server / inference
_MAX_STROKE_STEPS = 15


def _default_openenv_url() -> str:
    return os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000")


def _coerce_stroke_list(raw: str) -> list[dict[str, Any]] | None:
    """
    Parse model completion into a list of stroke dicts.

    Accepts:
    - A JSON object with a ``"strokes"`` key (list of objects).
    - A top-level JSON array of stroke objects.
    - A ```json ... ``` fence (best-effort).
    """
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "strokes" in data and isinstance(data["strokes"], list):
        return data["strokes"]
    return None


def _dict_to_action(d: dict[str, Any]) -> Any:
    from learn_handwriting import LearnHandwritingAction

    raw_type = d.get("action_type", "line")
    at = raw_type.strip().lower() if isinstance(raw_type, str) else "line"
    if at not in ("line", "curve", "circle", "ellipse"):
        at = "line"

    return LearnHandwritingAction(
        action_type=at,
        x1=int(d.get("x1", 0)),
        y1=int(d.get("y1", 0)),
        x2=(int(d["x2"]) if d.get("x2") is not None else None),
        y2=(int(d["y2"]) if d.get("y2") is not None else None),
        x3=(int(d["x3"]) if d.get("x3") is not None else None),
        y3=(int(d["y3"]) if d.get("y3") is not None else None),
        radius=(int(d["radius"]) if d.get("radius") is not None else None),
        rx=(int(d["rx"]) if d.get("rx") is not None else None),
        ry=(int(d["ry"]) if d.get("ry") is not None else None),
    )


def run_episode_from_stroke_list(
    base_url: str,
    task: str,
    target_character: str,
    stroke_dicts: list[dict[str, Any]] | None,
) -> float:
    """
    Run a single OpenEnv episode; return a scalar reward (final match percentage).

    If ``stroke_dicts`` is None or empty, returns 0.0.
    """
    if not stroke_dicts:
        return 0.0
    from learn_handwriting import LearnHandwritingEnv

    raw = LearnHandwritingEnv(base_url=base_url)
    with raw.sync() as env:
        result = env.reset(task=task, character=target_character)
        if result is None or result.observation is None:
            return 0.0
        obs = result.observation
        for stroke in stroke_dicts[:_MAX_STROKE_STEPS]:
            action = _dict_to_action(stroke)
            step_res = env.step(action)
            if step_res is None or step_res.observation is None:
                break
            obs = step_res.observation
            r = step_res.reward
            if step_res.done or getattr(obs, "done", False):
                if r is not None and r != 0.0:
                    return float(r)
                return float(getattr(obs, "match_percentage", 0.0) or 0.0)
        return float(getattr(obs, "match_percentage", 0.0) or 0.0)


def openenv_stroke_list_reward(
    prompts: list,
    completions: list[str | None] | list,
    **kwargs: Any,
) -> list[float | None]:
    """
    TRL ``reward_funcs`` entry: one OpenEnv roll-out per (prompt, completion) pair.

    TRL 1.2+ also passes (ignore unless you need them): ``completion_ids``, ``trainer_state``,
    ``log_extra``, ``log_metric``, and sometimes ``environments`` — use ``**kwargs`` and only read
    dataset columns you care about.

    Extra **dataset** columns (passed through by TRL) should include, per row:

    - ``task``: str — difficulty (e.g. ``"easy"``).
    - ``target_character``: str — target letter (e.g. ``"L"``).

    You can add these to ``train_dataset`` in ``grpo_train.py``.

    The completion should contain JSON with a ``strokes`` list (see ``GRPO_PLAN.md``).
    """
    task_col = kwargs.get("task")
    char_col = kwargs.get("target_character")
    n = len(prompts)
    base_url = _default_openenv_url()
    out: list[float] = []
    for i in range(n):
        c = completions[i] if i < len(completions) else None
        if not c or not isinstance(c, str):
            out.append(0.0)
            continue
        task = "easy"
        if task_col is not None:
            if isinstance(task_col, (list, tuple)) and i < len(task_col):
                task = str(task_col[i])
            elif not isinstance(task_col, (list, tuple)):
                task = str(task_col)
        ch = "L"
        if char_col is not None:
            if isinstance(char_col, (list, tuple)) and i < len(char_col):
                ch = str(char_col[i])
            elif not isinstance(char_col, (list, tuple)):
                ch = str(char_col)
        strokes = _coerce_stroke_list(c)
        r = run_episode_from_stroke_list(base_url, task, ch, strokes)
        out.append(r)
    return out
