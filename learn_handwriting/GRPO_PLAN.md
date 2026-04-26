# GRPO-only training (OpenEnv + TRL)

This project uses **Group Relative Policy Optimization (GRPO)** in Hugging Face **TRL** with rewards computed from the **real** Learn Handwriting **OpenEnv** server. There is **no** supervised SFT or Hugging Face Jobs path in this document.

## Scope

- **In scope:** `GRPOTrainer` + `GRPOConfig` + custom `reward_funcs` that connect to `LearnHandwritingEnv` and [`LearnHandwritingAction`](./models.py).
- **Out of scope:** `train_qwen_lora.py`, JSONL SFT, Hub Jobs unless you add them back yourself.

## How TRL GRPO fits this repo (important)

`GRPOTrainer` (see [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer)) takes:

- a **causal LM** (Hub id or path),
- a **`train_dataset` with a `prompt` column** (plus any extra columns your reward needs),
- one or more **`reward_funcs`**.

Each reward function receives **batched** keyword args: at least `prompts`, `completions`, and **any extra column** from the dataset. It must return a **list of floats** (one per sample in the batch), using `**kwargs` for forward compatibility.

**There is no `env_factory` in current TRL** for generic OpenEnv. The **environment runs inside the reward function**: for each generated `completion` string, you parse it, run `reset` / `step` on [`LearnHandwritingEnv`](./client.py), and map the final observation to a scalar (e.g. `match_percentage` or last-step reward). The old sketch in `temp.md` (OpenEnvBridge + `env_factory`) is **not** the current TRL API; use `reward_funcs` + `LearnHandwritingEnv` as in [`datagen_sft/grpo_rewards.py`](datagen_sft/grpo_rewards.py).

## Multi-step handwriting vs one GRPO completion

The live policy in [`inference.py`](inference.py) issues **up to 15** separate LLM calls per episode. **Standard GRPO** usually scores **one completion per prompt** per group. Practical options:

1. **Trajectory JSON in one completion (implemented default):** the model outputs a **single** JSON value whose text includes a `strokes` array; the reward function rolls the env forward stroke-by-stroke. See `parse_stroke_list_completion` in `grpo_rewards.py`.
2. **Shorter curriculum:** one completion = one stroke; dataset prompts include conversation history (heavier dataset design).
3. **Custom outer loop** outside TRL: not covered here.

## Operational checklist

1. **Start the OpenEnv HTTP server** (e.g. `uv run server` or your Docker image) and confirm `OPENENV_BASE_URL` (e.g. `http://127.0.0.1:8000`).
2. **Concurrency:** `num_generations` in `GRPOConfig` creates multiple generations per prompt; the reward function may run **many** env episodes per optimizer step. Ensure Uvicorn/workers and machine resources can handle it, or **lower** `num_generations` and `per_device_train_batch_size` for smoke tests.
3. **Memory:** 7B GRPO is **heavy**; use **LoRA** (`peft_config` in `grpo_train.py`) and, if available, `use_vllm=True` in `GRPOConfig` per TRL docs.
4. **Verify connectivity** before a long run: `python datagen_sft/grpo_smoke_episode.py` (see that file for usage).

## GPU and VRAM (unquantized base, LoRA adapters)

This GRPO path keeps the **base model in bf16/fp16** (no 4/8-bit weight quantization). **LoRA** still trains a small set of parameters, but the **full backbone** stays in half precision, and **GRPO** adds **sampling** (`num_generations`), long **`max_completion_length`**, and often a **reference / KL** path—so it is **more VRAM-hungry** than SFT. OpenEnv (HTTP + CPU) is small compared to the LLM.

**Unquantized 7B + GRPO + LoRA (recommended targets)**

| VRAM (typical) | Examples (cloud / workstation) | Use case |
|----------------|----------------------------------|----------|
| **~40–48 GB** (primary) | L40S, A100-40, RTX 6000 Ada | **Sweet spot** for 7B: room to raise `num_generations`, use TRL vLLM, longer completions without constant OOM. |
| **80+ GB** | A100-80, H100 80 | Maximum headroom for 7B or larger policies later. |
| **~32 GB** (tight) | RTX 5090, etc. | 7B **possible** only with **small** `num_generations`, **short** completions, LoRA, and vLLM / careful `GRPOConfig`; expect to tune. |
| **~20–24 GB** | RTX 4000 Ada, RTX 4090 | **Poor fit for 7B** unquant + GRPO; use for **sub-3B** smoke (e.g. default `--model Qwen/Qwen2.5-0.5B-Instruct` in `grpo_train.py`) or development only. |

**Rule of thumb:** plan **~48 GB** for **7B** unquantized GRPO; **0.5B** smoke on **24 GB** is fine.

## Files added for this track

| File | Role |
|------|------|
| [`datagen_sft/grpo_rewards.py`](datagen_sft/grpo_rewards.py) | Reward that parses a stroke-list JSON completion and calls OpenEnv. |
| [`datagen_sft/grpo_train.py`](datagen_sft/grpo_train.py) | Example `GRPOTrainer` + tiny dataset; extend with your prompts and PEFT. |
| [`datagen_sft/grpo_smoke_episode.py`](datagen_sft/grpo_smoke_episode.py) | One-episode smoke test to validate server + action schema. |
| `GRPO_PLAN.md` | This plan (GRPO-only). |

## Dependencies

Use the `train` optional extra from `pyproject.toml` (includes `trl`, `peft`, etc.) and run with `PYTHONPATH=.` from the repo root. Optional: `vllm` for faster generation, per TRL `GRPOConfig`.

## References

- [TRL `GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)
- [`server/learn_handwriting_environment.py`](server/learn_handwriting_environment.py) — episode rules, `TASK_CHARACTERS`, `reset(task=..., character=...)`
