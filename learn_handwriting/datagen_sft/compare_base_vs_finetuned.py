#!/usr/bin/env python3
"""
Compare stroke policy quality: base Qwen vs multiple fine-tuned experiments.

This script supports:
1. API-based inference (OpenAI compatible).
2. Local in-process inference (Transformers + PEFT).
3. High-efficiency adapter swapping (benchmarking multiple LoRAs without reloading the base model).

Example (local base + LoRA folders; quote the glob so the shell does not expand it first):
  PYTHONPATH=. python datagen_sft/compare_base_vs_finetuned.py \
    --model_dirs 'outputs/rev-r*' \
    --suite quick

Example (Hugging Face Inference / OpenAI-style API: hub model ids, no local GPU):
  export HF_TOKEN=...
  PYTHONPATH=. python datagen_sft/compare_base_vs_finetuned.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --finetuned_model <hub-id> \
    --suite quick
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _bootstrap_learn_handwriting_package() -> None:
    """Same import graph as ``pip install -e .`` so ``import learn_handwriting`` works with PYTHONPATH=."""
    if "learn_handwriting" in sys.modules:
        return
    import importlib.util

    init_path = os.path.join(_ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "learn_handwriting",
        init_path,
        submodule_search_locations=[_ROOT],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load learn_handwriting from {init_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learn_handwriting"] = mod
    spec.loader.exec_module(mod)


_bootstrap_learn_handwriting_package()

from openai import OpenAI

from inference import (  # noqa: E402
    BENCHMARK,
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


# ── Local Model Client (Mimics OpenAI for Transformers) ──────────────────────


def _is_local_model_path(path: str) -> bool:
    """True if ``path`` is an on-disk file or directory (not a Hub id like ``org/model``)."""
    if not (path and path.strip()):
        return False
    expanded = os.path.normpath(os.path.expanduser(path))
    return os.path.isdir(expanded) or os.path.isfile(expanded)


class LocalModelClient:
    def __init__(self, model_id_or_path: str, revision: str = "main"):
        # --- Device Discovery ---
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if self.device == "cuda":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device == "mps":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        print(f"📦 Loading base model: {model_id_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            revision=revision
        )
        self.model.eval()
        # Match OpenAI client: get_stroke() calls client.chat.completions.create(...)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._openai_create_chat_completion)
        )

    def _param_device(self):
        return next(self.model.parameters()).device

    def load_lora(self, lora_path: str, adapter_name: str):
        """Loads or switches to a specific LoRA adapter."""
        if adapter_name in getattr(self.model, "peft_config", {}):
            self.model.set_adapter(adapter_name)
        else:
            if not isinstance(self.model, PeftModel):
                self.model = PeftModel.from_pretrained(self.model, lora_path, adapter_name=adapter_name)
            else:
                self.model.load_adapter(lora_path, adapter_name=adapter_name)
                self.model.set_adapter(adapter_name)
        self.model.eval()

    def unload_adapter(self):
        """Unloads current adapter and reverts to base model."""
        if isinstance(self.model, PeftModel):
            self.model.unload() # Note: This might be destructive depending on PEFT version
            # A safer way to 'disable' is model.base_model.disable_adapter_layers() 
            # but for benchmarking, we usually just set to a different one or use 'default'

    def _openai_create_chat_completion(
        self,
        *,
        model: str,
        messages: list,
        temperature: float = 0.7,
        response_format: dict | None = None,
        **kwargs: Any,
    ):
        """Mimic ``openai.OpenAI().chat.completions.create`` (keyword args only, like the real client)."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self._param_device())

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0
            )

        content = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        class MockChoice:
            def __init__(self, c): self.message = type('obj', (object,), {'content': c})

        class MockResponse:
            def __init__(self, c): self.choices = [MockChoice(c)]

        return MockResponse(content)


def _eval_grid(suite: str) -> list[tuple[str, str]]:
    """(task, character) pairs for deterministic episodes."""
    if suite == "full":
        out: list[tuple[str, str]] = []
        for task, chars in TASK_CHARACTERS.items():
            for c in chars:
                out.append((task, c))
        return out
    if suite == "quick":
        return [
            ("easy", "L"),
            ("easy", "X"),
            ("medium", "A"),
            ("medium", "E"),
            ("hard", "O"),
            ("hard", "B"),
        ]
    raise ValueError(f"Unknown suite: {suite}")


def run_episode_local(
    client: OpenAI | LocalModelClient,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple stroke models on fixed episodes.")
    parser.add_argument(
        "--base_model",
        default=os.environ.get("BASE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument(
        "--finetuned_model",
        default=os.environ.get("FINETUNED_MODEL_NAME", ""),
        help="HF model id or local merged checkpoint path served by your API",
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        help="One or more local directory paths (supports globs like outputs/rev-*)",
    )
    parser.add_argument(
        "--api_base_url",
        default=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "",
    )
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--json_out",
        default="benchmark_results.json",
        help="Optional path to write detailed per-episode results as JSON",
    )
    args = parser.parse_args()

    # Local mode: --model_dirs (LoRA sweeps) or a base path on disk. Hub ids (e.g. Qwen/...) are not local
    # even though they contain "/" — a previous `"/" in base_model` check wrongly forced local inference.
    is_local = bool(args.model_dirs) or _is_local_model_path(args.base_model)

    if is_local:
        client = LocalModelClient(args.base_model)
    else:
        client = OpenAI(base_url=args.api_base_url, api_key=args.api_key or "dummy")

    grid = _eval_grid(args.suite)
    
    # Collect all models to test
    models_to_test = []
    if not args.model_dirs:
        models_to_test.append(("base", args.base_model))
        if args.finetuned_model:
            models_to_test.append(("fine-tuned", args.finetuned_model))
    else:
        models_to_test.append(("base", args.base_model))
        for pattern in args.model_dirs:
            for d in glob.glob(pattern):
                if os.path.isdir(d):
                    models_to_test.append((os.path.basename(d), d))

    print(f"\n🚀 Starting Benchmark | Suite: {args.suite} | Mode: {'LOCAL' if is_local else 'API'}")
    print(f"Models to test: {[m[0] for m in models_to_test]}\n")

    results = {}

    for name, path in models_to_test:
        print(f"🧪 Testing: {name} ({path})")
        
        if is_local:
            if name == "base":
                # Ensure no adapters are active
                if isinstance(client.model, PeftModel):
                    client.model = client.model.unload() # Or disable
            else:
                client.load_lora(path, name)
            model_id = name
        else:
            model_id = path

        rows = []
        for task, char in grid:
            row = run_episode_local(client, model_id, task, char, args.verbose)
            rows.append(row)
            if args.verbose:
                print(f"  {task}/{char} -> coverage={row['final_coverage']:.1%}")
        
        results[name] = rows
        
        # Intermediate summary
        succ = sum(1 for r in rows if r["success"])
        cov = sum(r["final_coverage"] for r in rows) / len(rows)
        print(f"  🏁 Result: Success Rate={succ/len(rows):.1%} | Mean Coverage={cov:.1%}\n")

        if is_local and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final Summary Table
    print("=" * 60)
    print(f"{'MODEL':<20} | {'SUCCESS RATE':<15} | {'MEAN COVERAGE':<15}")
    print("-" * 60)
    
    summary_data = {}
    for name, rows in results.items():
        succ_rate = sum(1 for r in rows if r["success"]) / len(rows)
        mean_cov = sum(r["final_coverage"] for r in rows) / len(rows)
        print(f"{name:<20} | {succ_rate:<15.1%} | {mean_cov:<15.1%}")
        summary_data[name] = {
            "success_rate": succ_rate,
            "mean_coverage": mean_cov,
            "episodes": len(rows)
        }
    print("=" * 60)

    if args.json_out:
        out = {
            "summary": summary_data,
            "details": results
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nDetailed results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
