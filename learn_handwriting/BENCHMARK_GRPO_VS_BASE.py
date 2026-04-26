#!/usr/bin/env python3
"""
Compare base Qwen2.5-7B-Instruct vs GRPO LoRA.

Runs six fixed episodes: two letters per difficulty (easy / medium / hard), then prints
per-tier and overall metrics plus JSON.

**API mode (default):** OpenAI-compatible HF router — needs HF_TOKEN and a runnable model id
for the GRPO checkpoint (merged or router-supported).

**Local mode (`--local`):** Loads base + LoRA with Transformers/PEFT on GPU/CPU — use this on
Hugging Face GPU Jobs so adapter-only Hub repos work without the Inference API.

Run from repo root:
  export HF_TOKEN=...
  PYTHONPATH=. python BENCHMARK_GRPO_VS_BASE.py --verbose
  PYTHONPATH=. python BENCHMARK_GRPO_VS_BASE.py --local --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List

from openai import OpenAI

# Repo root on sys.path (same pattern as inference.py / datagen scripts)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from inference import (  # noqa: E402
    MAX_STEPS,
    SCORE_MAX,
    SCORE_MIN,
    SUCCESS_SCORE_THRESHOLD,
    get_stroke,
)
from learn_handwriting import LearnHandwritingAction  # noqa: E402
from server.learn_handwriting_environment import (  # noqa: E402
    LearnHandwritingEnvironment,
    TASK_CHARACTERS,
)

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_GRPO = "abhijeetmishra101/Qwen2.5-7B-Handwriting-GRPO"


def eval_grid_two_per_level() -> list[tuple[str, str]]:
    """Two characters per difficulty, deterministic (first two in each pool)."""
    out: list[tuple[str, str]] = []
    for task in ("easy", "medium", "hard"):
        chars = TASK_CHARACTERS[task][:2]
        for c in chars:
            out.append((task, c))
    return out


def run_episode(
    client: Any,
    model: str,
    task: str,
    character: str,
    verbose: bool,
) -> dict[str, Any]:
    env = LearnHandwritingEnvironment(task=task)
    obs = env.reset(task=task, character=character)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    match_percentage = 0.0
    last_pixels_matched = 0
    last_pixels_wasted = 0
    ink_remaining = obs.ink_remaining
    last_reward = 0.0
    last_integrity_violated = False

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break
        strokes_remaining = MAX_STEPS - step + 1
        bbox = (obs.char_bbox_x1, obs.char_bbox_y1, obs.char_bbox_x2, obs.char_bbox_y2)
        stroke = get_stroke(
            client,
            step,
            strokes_remaining,
            character,
            obs.match_percentage,
            last_pixels_matched,
            last_pixels_wasted,
            ink_remaining,
            last_reward,
            history,
            model=model,
            last_integrity_violated=last_integrity_violated,
            char_bbox=bbox,
        )
        obs = env.step(
            LearnHandwritingAction(
                action_type=stroke.action_type,
                x1=stroke.x1,
                y1=stroke.y1,
                x2=stroke.x2,
                y2=stroke.y2,
                x3=stroke.x3,
                y3=stroke.y3,
                radius=stroke.radius,
                rx=stroke.rx,
                ry=stroke.ry,
            )
        )
        reward = obs.reward or 0.0
        rewards.append(reward)
        steps_taken = step
        match_percentage = obs.match_percentage
        last_pixels_matched = obs.pixels_matched_this_stroke
        last_pixels_wasted = getattr(obs, "pixels_wasted_this_stroke", 0)
        ink_remaining = obs.ink_remaining
        last_integrity_violated = obs.integrity_violated
        last_reward = reward
        if verbose:
            print(f"  step {step} reward={reward:.4f} coverage={match_percentage:.1%}", flush=True)
        integrity_tag = " [INTEGRITY]" if last_integrity_violated else ""
        history.append(
            f"Step {step}: {stroke.action_type} → matched={last_pixels_matched}px "
            f"reward={reward:.4f} coverage={match_percentage:.1%}{integrity_tag}"
        )
        if obs.done:
            break

    score = min(max(match_percentage, SCORE_MIN), SCORE_MAX)
    success = score >= SUCCESS_SCORE_THRESHOLD
    return {
        "task": task,
        "character": character,
        "model": model,
        "steps": steps_taken,
        "final_coverage": match_percentage,
        "score": score,
        "success": success,
        "integrity_violated": last_integrity_violated,
        "sum_reward": sum(rewards),
    }


def summarize_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for r in rows:
        by_task[r["task"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for task, task_rows in by_task.items():
        if not task_rows:
            continue
        n = len(task_rows)
        out[task] = {
            "episodes": n,
            "mean_coverage": sum(x["final_coverage"] for x in task_rows) / n,
            "successes": sum(1 for x in task_rows if x["success"]),
            "integrity_violations": sum(1 for x in task_rows if x["integrity_violated"]),
            "mean_steps": sum(x["steps"] for x in task_rows) / n,
        }
    return out


def summarize_overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "episodes": n,
        "mean_coverage": sum(x["final_coverage"] for x in rows) / n,
        "successes": sum(1 for x in rows if x["success"]),
        "integrity_violations": sum(1 for x in rows if x["integrity_violated"]),
        "mean_steps": sum(x["steps"] for x in rows) / n,
    }


def print_comparison(
    base_label: str,
    grpo_label: str,
    base_rows: list[dict[str, Any]],
    grpo_rows: list[dict[str, Any]],
) -> None:
    b_task = summarize_by_task(base_rows)
    g_task = summarize_by_task(grpo_rows)
    b_all = summarize_overall(base_rows)
    g_all = summarize_overall(grpo_rows)

    print("\n" + "=" * 72)
    print("Per difficulty (2 episodes each)")
    print("=" * 72)
    print(f"{'Tier':<8} | {'Model':<12} | {'Mean cov':>10} | {'Success':>8} | {'Integrity':>10}")
    print("-" * 72)
    for tier in ("easy", "medium", "hard"):
        bt = b_task.get(tier, {})
        gt = g_task.get(tier, {})
        print(
            f"{tier:<8} | {base_label:<12} | {bt.get('mean_coverage', 0):>10.1%} | "
            f"{bt.get('successes', 0)}/{bt.get('episodes', 0):<4} | {bt.get('integrity_violations', 0):>10}"
        )
        print(
            f"{'':8} | {grpo_label:<12} | {gt.get('mean_coverage', 0):>10.1%} | "
            f"{gt.get('successes', 0)}/{gt.get('episodes', 0):<4} | {gt.get('integrity_violations', 0):>10}"
        )
        delta = gt.get("mean_coverage", 0) - bt.get("mean_coverage", 0)
        print(f"{'':8} | {'Δ GRPO-base':<12} | {delta:>+10.1%} |")
        print("-" * 72)

    print("\nOverall")
    print("-" * 72)
    print(
        f"{base_label:<12}  mean_cov={b_all['mean_coverage']:.1%}  "
        f"success={b_all['successes']}/{b_all['episodes']}  "
        f"integrity={b_all['integrity_violations']}"
    )
    print(
        f"{grpo_label:<12}  mean_cov={g_all['mean_coverage']:.1%}  "
        f"success={g_all['successes']}/{g_all['episodes']}  "
        f"integrity={g_all['integrity_violations']}"
    )
    cov_delta = g_all["mean_coverage"] - b_all["mean_coverage"]
    print(f"{'Δ GRPO-base':<12}  mean_cov={cov_delta:+.1%}")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Base vs GRPO LoRA: 2 chars × easy/medium/hard (API or local PEFT).",
    )
    parser.add_argument("--base_model", default=os.environ.get("BASE_MODEL_NAME", DEFAULT_BASE))
    parser.add_argument("--grpo_model", default=os.environ.get("GRPO_MODEL_NAME", DEFAULT_GRPO))
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run on local GPU/CPU with Transformers+PEFT (Hub LoRA id); ignores --api_base_url.",
    )
    parser.add_argument(
        "--api_base_url",
        default=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--json_out",
        default="grpo_hf_benchmark_results.json",
        help="Write detailed results here (set empty to skip).",
    )
    args = parser.parse_args()

    grid = eval_grid_two_per_level()
    base_rows: list[dict[str, Any]] = []
    grpo_rows: list[dict[str, Any]] = []

    print("Grid (task, letter):", grid)

    if args.local:
        from datagen_sft.compare_base_vs_finetuned import LocalModelClient

        print("Backend: local Transformers + PEFT\n")
        client = LocalModelClient(args.base_model)
        for task, char in grid:
            print(f"BASE  {task}/{char} …", flush=True)
            base_rows.append(run_episode(client, args.base_model, task, char, args.verbose))
            print(f"  → coverage={base_rows[-1]['final_coverage']:.1%} success={base_rows[-1]['success']}")
        client.load_lora(args.grpo_model, "grpo")
        for task, char in grid:
            print(f"GRPO  {task}/{char} …", flush=True)
            grpo_rows.append(run_episode(client, args.grpo_model, task, char, args.verbose))
            print(f"  → coverage={grpo_rows[-1]['final_coverage']:.1%} success={grpo_rows[-1]['success']}")
    else:
        client = OpenAI(base_url=args.api_base_url, api_key=args.api_key or "dummy")
        print(f"Backend: HF Inference API ({args.api_base_url})\n")
        for task, char in grid:
            print(f"BASE  {task}/{char} …", flush=True)
            base_rows.append(run_episode(client, args.base_model, task, char, args.verbose))
            print(f"  → coverage={base_rows[-1]['final_coverage']:.1%} success={base_rows[-1]['success']}")

        for task, char in grid:
            print(f"GRPO  {task}/{char} …", flush=True)
            grpo_rows.append(run_episode(client, args.grpo_model, task, char, args.verbose))
            print(f"  → coverage={grpo_rows[-1]['final_coverage']:.1%} success={grpo_rows[-1]['success']}")

    print_comparison("base", "grpo", base_rows, grpo_rows)

    if args.json_out:
        payload = {
            "backend": "local" if args.local else "api",
            "base_model": args.base_model,
            "grpo_model": args.grpo_model,
            "api_base_url": None if args.local else args.api_base_url,
            "grid": [{"task": t, "character": c} for t, c in grid],
            "summary": {
                "base": {
                    "overall": summarize_overall(base_rows),
                    "by_task": summarize_by_task(base_rows),
                },
                "grpo": {
                    "overall": summarize_overall(grpo_rows),
                    "by_task": summarize_by_task(grpo_rows),
                },
            },
            "episodes": {"base": base_rows, "grpo": grpo_rows},
        }
        out_path = (
            os.path.join(_ROOT, args.json_out) if not os.path.isabs(args.json_out) else args.json_out
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
