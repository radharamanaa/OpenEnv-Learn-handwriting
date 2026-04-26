---
title: Learn Handwriting Environment Server
emoji: 📻
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Learn Handwriting (OpenEnv)

A **Learn Handwriting** [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-style environment: an LLM draws **26 capital letters (A–Z)** on a **100×100** canvas using **vector** actions (lines, curves, circles, ellipses). The **HTTP server** renders targets from **Roboto Bold** on the fly (no image dataset). The project includes:

- a **reinforcement-style training path** using **Hugging Face TRL GRPO** and **real** environment rewards (HTTP client to the same server you run on a Space or locally), and
- a **supervised (SFT) path** with **LoRA/QLoRA** on **multi-turn JSONL** (one stroke per turn), **benchmarks**, and **notebooks** to watch the policy.

**Python package name:** `openenv-learn_handwriting` (see `pyproject.toml`). The import package is `learn_handwriting` with the server under `learn_handwriting.server`.

---

## Requirements

- **Python 3.10+**
- **Core install** (server + client + OpenEnv): from this directory → `pip install -e .` or `uv sync`
- **Training (PyTorch, TRL, PEFT, etc.):** `pip install -e ".[train]"` (or `uv sync --extra train`)

`[train]` pins **`trl` in the 1.x range** (GRPO / `GRPOConfig` changed across TRL 0.x/1.x; see `datagen_sft/grpo_train.py`).

---

## Run the environment server

The OpenEnv HTTP API is what **GRPO rewards** and **inference** talk to. Examples:

```bash
# from repo root (this folder)
uv run server
# or
python -m learn_handwriting.server.app
```

For deployment, this repo’s Space config lives in the YAML header above; the app serves on the configured port (e.g. **8000**). Point clients at the base URL with **no** `/health` path suffix in env vars like `OPENENV_BASE_URL` (e.g. `https://username-myspace.hf.space`).

---

## Training: two main tracks

| Track | What | Entry points |
|--------|------|----------------|
| **SFT + LoRA** | Multi-turn **ChatML** JSONL; one JSON **stroke** per assistant turn. | `datagen_sft/simulate_and_format.py` → `qwen25_finetune_data.jsonl` → `datagen_sft/train_qwen_lora.py`; batch runner `run_experiments.sh` |
| **GRPO + TRL** | **Group** sampling; **scalar reward** from the **OpenEnv** server by parsing a **full stroke list** in one model completion. | `datagen_sft/grpo_train.py` + `datagen_sft/grpo_rewards.py`; see **`GRPO_PLAN.md` |

These are complementary: SFT is multi-step chat data; the current GRPO template scores **one completion** containing a `strokes` list (trajectory in JSON), as described in `GRPO_PLAN.md`.

**Connectivity check before a long GRPO run:**

```bash
PYTHONPATH=. python datagen_sft/grpo_smoke_episode.py
```

---

## GRPO on Hugging Face GPU Jobs (optional)

The notebook **`run_hf_training_job.ipynb`** (repo root) logs into Hugging Face, runs a **GPU Job** that **clones a public Git** repo, installs **`[train]`-style** deps, and runs `datagen_sft/grpo_train.py` with `OPENENV_BASE_URL` and Hub adapter push (via `HF_TOKEN` in Job secrets and env like `HUB_MODEL_ID`).

You typically set:

1. **Public git URL** + **branch** (single-branch shallow clone) so the Job sees `learn_handwriting/` at the clone root (for example `REPO_ROOT=/tmp/lh/learn_handwriting` when cloning into `/tmp/lh`).
2. **`OPENENV_BASE_URL`** — the Space (or other host) where the **Learn Handwriting** server runs (rewards; not used for `git clone`).
3. **HF login** — Jobs and optional **`push_to_hub`** for the adapter (LoRA) repo.

Default flavor in the notebook is aimed at a **~48GB-class** GPU (e.g. L40S). GRPO is **memory-heavy**; see **`GRPO_PLAN.md`** for VRAM notes and `num_generations` / `generation_batch_size` behavior.

---

## Inference and data

- **`inference.py`** — live **multi-step** policy (up to 15 steps): builds prompts from `LearnHandwritingObservation`, calls the model, parses JSON actions.
- **`models.py`** — Pydantic (or shared) **action / observation** types aligned with the server.
- **`client.py`** — **`LearnHandwritingEnv`**: `reset` / `step` over HTTP (OpenEnv client pattern).

Data format for SFT: **`docs/FINETUNE_DATA_FORMAT.md`**. Generation/benchmarks: `datagen_sft/compare_base_vs_finetuned.py`, `datagen_sft/verify_readable.py`, `visualizations/`.

---

## Project layout (overview)

```text
learn_handwriting/          # package root; also HF Space / Job working directory name
├── server/                 # FastAPI app, renderer, LearnHandwritingEnvironment rules
├── datagen_sft/            # SFT data, LoRA & GRPO scripts, oracles, benchmarks
├── visualizations/         # Notebooks / helpers to watch drawing runs
├── docs/                   # e.g. FINETUNE_DATA_FORMAT.md
├── Roboto/                 # Font assets for on-the-fly target rasterization
├── run_hf_training_job.ipynb
├── GRPO_PLAN.md
├── inference.py
├── client.py
├── models.py
├── pyproject.toml
└── run_experiments.sh      # optional multi–LoRA SFT sweeps
```

---

## Tests

```bash
pytest
```

---

## License and credits

Code under the project **LICENSE** (see repository root). Built for the **Scaler / Google DeepMind Hackathon** (April 2026).

For GRPO design details, TRL reward contract, and VRAM: **`GRPO_PLAN.md`**.
