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

# ✍️ Learn Handwriting Environment

A Reinforcement Learning (RL) environment and Supervised Fine-Tuning (SFT) pipeline where an AI agent learns to draw **all 26 capital letters (A–Z)** using geometric vector tools (lines, curves, circles, ellipses) on a 100×100 canvas.

This project bridges the gap between high-level reasoning and low-level motor control by forcing an LLM to "think" in coordinates and geometry rather than pixel grids.

---

## 🏗️ Architecture & Core Mechanics

### 1. The Environment (`server/`)
The environment is built using **OpenCV** for high-performance vector-to-raster rendering.
- **On-the-fly Rendering**: Targets are rasterized from **Roboto Bold** at runtime. No image datasets are needed.
- **Vector Toolset**: The agent uses 4 primitive actions: `line`, `curve` (3-point Bezier), `circle`, and `ellipse`.
- **Integrity Enforcement**: To prevent "cheating" (e.g., filling a solid block instead of an 'O'), we use **disqualification masks**. If an agent fills "protected" internal regions (like the hole in a 'D'), it fails immediately.
- **Ink Efficiency**: A strict `MAX_DRAWN_MULTIPLIER` (default 1.7x) prevents "scribbling." If the agent uses too much ink, the episode terminates.

### 2. The Agent Policy (`inference.py`)
The agent is an LLM (typically **Qwen 2.5 Instruct**) that receives structured observations and returns structured JSON actions.
- **Spatial Grounding**: On every step, the agent receives the tight **Bounding Box** (`char_bbox`) of the target character, grounding its coordinates in 2D space.
- **Reasoning Loop**: The agent is encouraged to provide its "thought process" before each stroke.

---

## 🚀 The Training Workflow (RunPod ➡️ Colab)

The project is optimized for a hybrid cloud workflow:

### Step 1: Generate Synthetic Data (`datagen_sft/`)
We use an **Oracle** that knows the perfect geometric paths for every letter to generate thousands of "perfect" episodes.
```bash
python datagen_sft/simulate_and_format.py
```
This produces `qwen25_finetune_data.jsonl`, a multi-turn ChatML dataset.

### Step 2: Mass Training on RunPod (`run_experiments.sh`)
Run multiple LoRA experiments with varying rank and alpha settings. This script pushes each version to the Hugging Face Hub under a unique revision (e.g., `rev-r16-a32`).
```bash
bash run_experiments.sh
```

### Step 3: Fast Benchmarking (`compare_base_vs_finetuned.py`)
Find the "winner" among your experiments without reloading the base model. This script uses **Adapter Swapping** for ultra-fast evaluation.
```bash
# Benchmark all revisions in your output folder
PYTHONPATH=. python datagen_sft/compare_base_vs_finetuned.py --model_dirs outputs/rev-* --suite quick
```

### Step 4: Visual Validation on Colab (`visualizations/`)
Load the winning revision in a Colab notebook and "watch" the agent draw in real-time.
- Set `MODEL_REVISION` to your winning experiment.
- Use the **`LocalModelClient`** to load weights directly into the Colab GPU.

---

## 🛠️ Key Tools & Scripts

### `datagen_sft/compare_base_vs_finetuned.py`
A high-throughput benchmarking suite. 
- **Multi-Model Support**: Compare any number of local LoRA adapters or remote API models.
- **Adapter Swapping**: Uses `peft` to swap LoRA weights in milliseconds on a single base model instance.
- **Leaderboard**: Generates a comparative summary of Success Rate and Mean Coverage.

### `visualizations/watch_runner.py`
The primary tool for observing agent behavior live. 
- Automatically detects hardware (CUDA, Apple Silicon MPS, or CPU).
- Mimics the OpenAI API for seamless switching between local weights and remote endpoints.

---

## 📦 Project Structure

```text
learn_handwriting/
├── server/                # OpenCV Environment & Renderer
├── datagen_sft/           # Data generation, LoRA training, & Benchmarking
├── visualizations/        # Watch notebooks & Local In-Memory Client
├── Roboto/                # Vendored font assets
├── models.py              # Pydantic schemas for Actions/Observations
├── inference.py           # Core agent logic and prompt building
├── run_experiments.sh     # Master experiment runner
└── pyproject.toml         # Dependency management (use [train] for DL)
```

---
*Built for the Scaler/Google Deepmind Hackathon - April 2026*
