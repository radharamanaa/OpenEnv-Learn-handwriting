#!/usr/bin/env python3
"""
Example GRPO training entry (TRL + OpenEnv reward).

Prerequisites:
  - OpenEnv server up (set OPENENV_BASE_URL).
  - ``pip install 'trl>=1.0,<2'`` and project deps (``uv sync --extra train`` or equivalent; GRPOConfig changed across TRL 0.x/1.x).
  - GPU: unquantized GRPO uses bf16/fp16 weights (no 4/8-bit); see GRPO_PLAN.md for
    VRAM (e.g. ~48G class for 7B+LoRA, small models for 24G smoke).

  OPENENV_BASE_URL=https://your-space.hf.space PYTHONPATH=. python datagen_sft/grpo_train.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct --hub_model_id your-name/Qwen2.5-Handwriting-GRPO

  Verbose OpenEnv WebSocket logs (reset/step payloads, validation errors): set
  LEARN_HANDWRITING_ENV_LOG=1; ``grpo_train`` calls ``configure_openenv_ws_logging()``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DSF = os.path.join(_ROOT, "datagen_sft")
if _DSF not in sys.path:
    sys.path.insert(0, _DSF)

from grpo_rewards import openenv_stroke_list_reward

from learn_handwriting import configure_openenv_ws_logging


def _smoke_dataset():
    from datasets import Dataset

    # Placeholder: model must output JSON with a "strokes" list (see GRPO_PLAN.md).
    prompt = (
        "You output only valid JSON. The object must have a key 'strokes' which is a list "
        "of stroke objects with fields action_type, x1, y1, and for lines x2, y2. "
        "Draw a capital L on a 100x100 canvas for task easy, letter L. "
        "Example: {\"strokes\":[{\"action_type\":\"line\",\"x1\":10,\"y1\":10,\"x2\":10,\"y2\":80}]}"
    )
    n = 4
    return Dataset.from_dict(
        {
            "prompt": [prompt] * n,
            "task": ["easy"] * n,
            "target_character": ["L"] * n,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training with OpenEnv reward (smoke / template).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="outputs/grpo-handwriting")
    parser.add_argument(
        "--openenv_base_url",
        default=os.environ.get("OPENENV_BASE_URL") or "http://127.0.0.1:8000",
        help="OpenEnv HTTP base URL (no /health). Overrides env OPENENV_BASE_URL for this process.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=2,
        help="Group size (TRL 1.2+ requires >= 2). Must divide the effective batch.",
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=1024, help="Must allow full JSON for stroke list."
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--hub_model_id",
        default=(os.environ.get("HUB_MODEL_ID") or "").strip() or None,
        help="Hugging Face repo id for push_to_hub (adapter). Can also set env HUB_MODEL_ID.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Create/use a private Hub repo when pushing the adapter.",
    )
    args = parser.parse_args()

    configure_openenv_ws_logging()

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    openenv_url = args.openenv_base_url.rstrip("/")
    os.environ["OPENENV_BASE_URL"] = openenv_url

    peft = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    # Do not set hub_strategy to "end": in current Transformers, that skips uploads inside
    # _push_from_checkpoint, so nothing is ever uploaded despite push_to_hub=True. Default
    # "every_save" pushes after each checkpoint (including the mandatory save on the last step).
    extra_hub: dict[str, object] = {}
    if args.hub_model_id:
        extra_hub = {
            "push_to_hub": True,
            "hub_model_id": args.hub_model_id,
            "hub_private_repo": args.hub_private,
            "hub_token": os.environ.get("HF_TOKEN"),
        }

    print(f"Model: {args.model}", flush=True)
    print(f"OpenEnv: {openenv_url}", flush=True)
    if args.hub_model_id:
        print(f"Hub (adapter): {args.hub_model_id}  private={args.hub_private}", flush=True)
    elif os.environ.get("HF_TOKEN"):
        print("Note: HF_TOKEN is set but --hub_model_id / HUB_MODEL_ID is empty; skipping Hub push.", flush=True)

    per_device_train_batch_size = 1
    gradient_accumulation_steps = 1
    # TRL: generation_batch_size (defaults to per_device * world * steps) must be divisible by num_generations.
    # With per_device=1 and one process that default is 1, which fails for num_generations=2+.
    world = max(1, int(os.environ.get("WORLD_SIZE", "1") or 1))
    base = per_device_train_batch_size * world * gradient_accumulation_steps
    gen_batch = (base * args.num_generations) // math.gcd(base, args.num_generations)

    # Newer TRL dropped GRPOConfig.max_prompt_length; truncate/filter prompts in the dataset if needed.
    tcfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        generation_batch_size=gen_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none",
        **extra_hub,
    )
    if torch.cuda.is_available():
        tcfg.bf16 = bool(torch.cuda.is_bf16_supported())
        tcfg.fp16 = not tcfg.bf16

    train_ds = _smoke_dataset()
    trainer = GRPOTrainer(
        model=args.model,
        args=tcfg,
        train_dataset=train_ds,
        reward_funcs=openenv_stroke_list_reward,
        peft_config=peft,
    )
    trainer.train()
    print(f"Done. Checkpoints in {args.output_dir}", flush=True)
    if args.hub_model_id and os.environ.get("HF_TOKEN"):
        # One blocking upload guarantees the Hub has weights (async checkpoint push can race or be skipped).
        trainer.push_to_hub(commit_message="GRPO adapter", blocking=True)
        print(f"Hub: https://huggingface.co/{args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
