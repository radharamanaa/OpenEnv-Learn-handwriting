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

# Learn Handwriting Environment

A Reinforcement Learning environment where an agent learns to draw **all 26 capital letters (A–Z)**
by issuing geometric actions (lines, curves, circles, ellipses) on a 100×100 canvas.
Letters are grouped into **easy**, **medium**, and **hard** pools by geometric complexity.
The agent is rewarded for each action that overlaps with the target character's white pixels.
To prevent "scribbling", there is a strict **ink penalty**: the agent fails immediately if it draws more than 1.7x the target's total pixels (configurable via `MAX_DRAWN_MULTIPLIER` env var).
The episode succeeds when 90% of the target is covered within 15 actions without running out of ink or violating shape integrity.

## How It Works

### Overview

On every `reset()` a random capital letter is chosen from the difficulty pool. The environment
**renders the target character on-the-fly** using Roboto Bold (vendored in `Roboto/static/`) onto
a 100×100 binary canvas — no pre-processed image files are needed.

The agent then issues actions one at a time. The agent can choose to draw a straight line, a 3-point curve, a circle, or an ellipse. The environment draws the shape with a fixed width of 8 pixels, computes how many pixels of that shape land on white pixels of the target character, calculates wasted ink, and returns a reward.

### Action Workflow (per step)

```
1. temp_matrix = zeros(100×100)
2. Draw selected shape (line/curve/circle/ellipse) on temp_matrix

3. pixels_matched_this_stroke = intersection(temp_matrix, target_matrix)
   reward = pixels_matched_this_stroke / total_target_pixels

4. pixels_wasted_this_stroke = sum(temp_matrix) - pixels_matched_this_stroke
   total_drawn_pixels = sum(canvas) + sum(temp_matrix)

5. canvas = max(canvas, temp_matrix)   ← merge stroke into cumulative canvas

6. [Integrity check] if canvas covers > 60% of any protected zone → done=True, reward=0

7. total_matched_pixels = intersection(canvas, target_matrix)
   match_percentage = total_matched_pixels / total_target_pixels

8. strokes_used += 1

9. done = (match_percentage >= 0.90) OR (strokes_used >= 15)
        OR (total_drawn_pixels > 1.7 * total_target_pixels)
        OR integrity_violated
```

Key design decisions:
- **Font-based rendering** — targets are rasterized from Roboto Bold (`Roboto/static/Roboto-Bold.ttf`) at runtime. No per-letter image assets are required for training or the server. For human inspection, run `python export_character_images.py` to write upscaled PNGs of every pool letter into `characters/`.
- **Rich Action Space** — line, curve (3-point Bezier), circle, and ellipse tools.
- **Fixed Width** — all strokes are 8 pixels wide to reduce LLM cognitive load.
- **Bounding Box Awareness** — The agent receives the tight `[x1, y1, x2, y2]` bounding box of the character on every step, removing the need to "guess" where to draw.
- **Strict Ink Penalty** — limits total drawn pixels to prevent brute-force coverage.
- **Shape Integrity** — protected regions (holes, gaps, open arcs) enforce correct letter topology.
- **Reward is per-stroke only** — previous strokes do not inflate the current reward.
- **Canvas merges cumulatively** — pixels already drawn are preserved across strokes.

### Character Pools

Characters are grouped by geometric complexity:

| Pool | Characters | Strokes needed | Notes |
|---|---|---|---|
| `easy` | `L`, `T`, `V`, `X` | 2 | Pure straight-line strokes only |
| `medium` | `A`, `N`, `Z`, `E`, `F`, `H`, `I`, `K`, `M`, `W`, `Y` | 3–4 | Straight lines, 3+ strokes; `A` has integrity constraint |
| `hard` | `B`, `C`, `D`, `G`, `J`, `O`, `P`, `Q`, `R`, `S`, `U` | 1–4+ | Curves and/or enclosed counters; integrity where noted below |

All **26** capital letters `A`–`Z` are included across the three pools.

### Shape Integrity Constraints

Some characters have **protected regions** the agent must not fill in:

| Character | Protected Region | Disqualifier |
|---|---|---|
| `A` | Inner triangle counter | Flood-fill interior |
| `B` | Upper and lower lobe counters | Flood-fill interior (2 components) |
| `D` | Bowl interior (semicircle) | Flood-fill interior |
| `O` | Circle interior | Flood-fill interior |
| `P` | Bowl interior (loop) | Flood-fill interior |
| `R` | Bowl interior (above the leg) | Flood-fill interior |
| `C` | Right-side opening | `render(O) − render(C)` |
| `S` | Two bridge gaps | `render(8) − render(S)` |
| `G` | Right-side opening | `render(O) − render(G)` |
| `Q` | Circle interior (ring must stay open) | Flood-fill interior |

If the agent's canvas covers **> 60%** of any protected zone, the episode ends immediately with `integrity_violated=True` and `reward=0.0`.

Letters **without** a row in the table above have no disqualification masks (for example `L`, `T`, `I`, `J`, `U`, and other straight or open-outline glyphs).

### `reset(task=..., character=...)`

`LearnHandwritingEnvironment.reset()` accepts:

- **`task`** — `"easy"`, `"medium"`, or `"hard"`. Updates the active character pool for this and later episodes.
- **`character`** — optional. If set, that exact capital letter becomes the target (no random sample). Used by `datagen_sft/simulate_and_format.py` so oracle strokes always match the rendered glyph after the task pool switches. The HTTP/WebSocket API continues to use ordinary random sampling unless you add a similar parameter server-side.

## Quick Start

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

try:
    env = LearnHandwritingEnv.from_docker_image("learn_handwriting-env:latest")

    result = env.reset()
    print(f"Draw character: {result.observation.target_character}")

    # Action 1: vertical bar (e.g. left side of L)
    result = env.step(LearnHandwritingAction(action_type="line", x1=25, y1=10, x2=25, y2=85))
    print(f"Matched this action : {result.observation.pixels_matched_this_stroke}")
    print(f"Ink remaining       : {result.observation.ink_remaining}")
    print(f"Cumulative coverage : {result.observation.match_percentage:.1%}")
    print(f"Reward              : {result.reward:.4f}")

    # Action 2: horizontal bar (e.g. bottom of L)
    result = env.step(LearnHandwritingAction(action_type="line", x1=25, y1=85, x2=75, y2=85))
    print(f"Matched this stroke : {result.observation.pixels_matched_this_stroke}")
    print(f"Cumulative coverage : {result.observation.match_percentage:.1%}")
    print(f"Done                : {result.done}")

finally:
    env.close()
```

## Dataset Generation (SFT)

To fine-tune models like **Qwen2.5-7B-Instruct** to act as a robust stroke-based policy agent, we use a modular synthetic data pipeline located in the `datagen_sft/` directory.

### The Pipeline
1. **`oracle.py` (Geometric Definitions)**: Stores the mathematically ideal stroke paths for every character in normalized coordinates `(0, 1)`.
2. **`generator.py` (Augmentation & Combinatorics)**:
   - Permutes all possible **stroke orders** and **drawing directions**.
   - Applies **stochastic jitter** (random pixel offsets) to ensure spatial robustness.
   - Generates **parallel offset strokes** to simulate brush width and guarantee the 90% coverage threshold is physically reachable.
3. **`simulate_and_format.py` (Validation & Grounding)**:
   - Executes the generated trajectories in the actual `LearnHandwritingEnvironment`, calling `reset(task=..., character=...)` so the target letter always matches the oracle.
   - Strictly filters out any trajectories that fail the 90% coverage or violate shape integrity.
   - Formats successful episodes into ChatML-style JSONL for fine-tuning.
   - **Augmentation depth** scales with oracle length: more `num_augments` for long stroke lists (e.g. `W`) and for 2–3 stroke letters so short oracles still get enough samples.
4. **`verify_readable.py` (Visualization)**:
   - Groups episodes by `target_character` and writes up to **three** sample episodes per character to `datagen_sft/readable_samples/<CHAR>_sample.json` for inspection.
   - For display only, **assistant** turns are pretty-printed as JSON objects; the **canonical SFT file** is still `qwen25_finetune_data.jsonl`, where each assistant `content` is a **JSON string** (as written by `simulate_and_format.py`).

5. **`test_l.py` (optional)**: Ad-hoc grid search over stroke placement for the letter `L` in the local environment; not part of the main JSONL export.

### Dataset status and reproducing
The checked-in file [`datagen_sft/qwen25_finetune_data.jsonl`](datagen_sft/qwen25_finetune_data.jsonl) is the SFT source of truth. Regenerating rewrites the file; line count is typically **a few thousand** rows (exact number depends on oracle hit rates and augmentation settings). Episodes are drawn from all three pools covering **every capital A–Z**, but **per-letter counts are not uniform**: some hard glyphs (e.g. sparse oracle coverage) appear less often than easy/medium letters. Row metadata includes `target_character`, `task`, `final_match_percentage`, and `num_strokes` (length of the saved stroke list after augmentation — each oracle primitive may become two offset traces for brush width).

To **regenerate** the JSONL and the readable sample JSON files from the project root (the datagen scripts extend `sys.path` to the repo root; tests use `PYTHONPATH=.`):

```bash
python datagen_sft/simulate_and_format.py
python datagen_sft/verify_readable.py
```

### Fine-tuning (Qwen2.5-7B-Instruct) — workflow
- **Training data** should be the **`messages` field** from each line of `qwen25_finetune_data.jsonl` (Chat-style turns: system, then alternating user/assistant for each stroke in an episode).
- **Prompt alignment**: `simulate_and_format.py` and `inference.py` both define the same **system** prompt text; the **user** template includes bounding box and per-step feedback so that a fine-tuned policy matches the hackathon `inference.py` loop. For interactive debugging, `visualizations/watch_runner.py` uses a shorter alternative prompt; use the long-form prompts for SFT and for fair before/after comparisons.
- A **Colab-oriented fine-tune script** (e.g. TRL + PEFT/LoRA, optional push to the Hugging Face Hub) and a **local eval script** (base model vs. fine-tuned on the in-process environment) are part of the intended workflow but may live **outside** this tree or be added under `datagen_sft/` as the project evolves. Install training stacks (PyTorch, `transformers`, `trl`, `peft`, etc.) in that environment; they are not required for the base `pyproject.toml` server client.

## Building the Docker Image

```bash
# From project root
docker build -t learn_handwriting-env:latest -f Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action — `LearnHandwritingAction`

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | str | `'line'`, `'curve'`, `'circle'`, or `'ellipse'` |
| `x1`, `y1` | int | Start x/y (for line/curve) or Center x/y (for circle/ellipse) |
| `x2`, `y2` | int | End x/y (ignored for circle/ellipse) |
| `x3`, `y3` | int | Pass-through midpoint x/y (for curve only) |
| `radius` | int | Radius (for circle only) |
| `rx`, `ry` | int | Horizontal/Vertical radius (for ellipse only) |

### Observation — `LearnHandwritingObservation`

| Field | Type | Description |
|-------|------|-------------|
| `target_character`          | str   | Capital letter to draw (e.g. `"L"`) |
| `strokes_used`              | int   | Actions consumed so far |
| `pixels_matched_this_stroke`| int   | Raw pixel overlap for this action only |
| `total_matched_pixels`      | int   | Cumulative pixels covered across all actions |
| `match_percentage`          | float | `total_matched / total_target` — progress toward 90% |
| `pixels_wasted_this_stroke` | int   | Pixels drawn this step that completely missed the target |
| `total_drawn_pixels`        | int   | Total ink on canvas |
| `max_allowed_pixels`        | int   | Maximum allowed ink (1.7x target) |
| `ink_remaining`             | int   | Remaining ink before failure |
| `integrity_violated`        | bool  | True if last stroke violated a shape integrity zone |
| `char_bbox_x1`              | int   | Left boundary (x) of the character's foreground |
| `char_bbox_y1`              | int   | Top boundary (y) of the character's foreground |
| `char_bbox_x2`              | int   | Right boundary (x) of the character's foreground |
| `char_bbox_y2`              | int   | Bottom boundary (y) of the character's foreground |
| `reward`                    | float | `pixels_matched_this_stroke / total_target_pixels` (0.0–1.0) |
| `done`                      | bool  | True when success, action limit, ink limit, or integrity violated |

> **Note:** The 100×100 canvas is intentionally **not** included in the observation.
> It is available via the `/state` endpoint (see `LearnHandwritingState` below).

### State — `LearnHandwritingState`

Returned by the `/state` endpoint. Contains the full episode state.

| Field              | Type             | Description |
|--------------------|------------------|-------------|
| `episode_id`       | str              | Unique ID for this episode |
| `step_count`       | int              | Total steps taken |
| `target_character` | str              | Character being drawn |
| `strokes_used`     | int              | Strokes used so far |
| `canvas`           | List[List[int]]  | 100×100 matrix of accumulated strokes (0 or 1) |

### Reward

```
reward = pixels_matched_this_stroke / total_target_pixels
```

- Range: **0.0 – 1.0** (compliant with hackathon grader requirements)
- Based on the **current stroke only** — previously drawn pixels do not contribute
- Returns **0.0** on integrity violation (never negative)

### Done Conditions

| Condition | Outcome |
|-----------|---------|
| `match_percentage >= 0.90` | ✅ Success — agent covered 90% of the character |
| `strokes_used >= 15`       | ❌ Failure — action budget exhausted |
| `total_drawn_pixels > max_allowed` | ❌ Failure — too much ink wasted |
| `integrity_violated`       | ❌ Failure — protected zone overfilled (reward=0) |

## Advanced Usage

### Connecting to an Existing Server

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

env = LearnHandwritingEnv(base_url="<ENV_HTTP_URL_HERE>")
result = env.reset()
result = env.step(LearnHandwritingAction(action_type="line", x1=10, y1=10, x2=80, y2=80))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

with LearnHandwritingEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print(f"Draw: {obs.observation.target_character}")
    result = env.step(LearnHandwritingAction(action_type="line", x1=25, y1=10, x2=25, y2=85))
    print(f"Coverage: {result.observation.match_percentage:.1%}")
```

### Concurrent WebSocket Sessions

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    LearnHandwritingEnvironment,  # Pass class, not instance
    LearnHandwritingAction,
    LearnHandwritingObservation,
    max_concurrent_envs=4,
)
```

### Inspecting the Canvas via `/state`

```python
state = env.get_state()
print(state.target_character)   # e.g. "A"
print(state.strokes_used)       # e.g. 3
canvas = state.canvas           # List[List[int]], 100×100, values 0 or 1
```

## Development & Testing

### Quick Sanity Check (no server needed)

With the repo root on `PYTHONPATH` (e.g. `PYTHONPATH=. python`):

```python
from models import LearnHandwritingAction
from server.learn_handwriting_environment import LearnHandwritingEnvironment

env = LearnHandwritingEnvironment()
obs = env.reset()
print(f"character={obs.target_character}")

obs = env.step(LearnHandwritingAction(action_type="line", x1=25, y1=10, x2=25, y2=85))
print(f"matched={obs.pixels_matched_this_stroke}, coverage={obs.match_percentage:.2%}, reward={obs.reward:.4f}")
```

If the package is **installed** in editable mode, the same imports work as `learn_handwriting.models` / `learn_handwriting.server.learn_handwriting_environment`.

### Running Tests

From the repository root, point Python at the flat package layout (`server/`, `models.py` at top level):

```bash
PYTHONPATH=. pytest tests/ -v
```

`tests/test_renderer.py` covers all **26** letters (pixel ranges, integrity mask presence, mask/foreground separation). `tests/test_integrity.py` exercises flood-fill and diff-based disqualifiers.

### Running Locally

```bash
uvicorn server.app:app --reload
```

## Local Monitoring & Visualization (Jupyter)

The project includes **local watch notebooks** designed to help you visually debug the agent's behavior step-by-step. These are separated by difficulty level: `watch_easy.ipynb`, `watch_medium.ipynb`, and `watch_hard.ipynb`. Notebook titles list the current character sets for each task (easy: L T V X; medium: includes I; hard: includes D J P R U among others).

To render **all** task characters and integrity overlays to `images/` (large PNGs for debugging masks), run:

```bash
python preview_characters.py
```

### Features
1. **Live Stroke Animation**: Watch the agent draw live! The canvas updates instantly after every stroke inside the notebook.
2. **LLM Prompt Auditing**: By default, the exact prompt string being sent to the LLM is printed to standard output before the API call to help you monitor what the model is reasoning about during long waits (`PRINT_LLM_PROMPT=true` in `.env`).
3. **Graphing & Analytics**: After the episode ends, the notebook automatically plots:
    - **Coverage Progression**: A line chart showing % coverage over steps, overlaying any shape integrity violations.
    - **Ink Efficiency**: A grouped bar chart comparing `Matched` vs `Wasted` pixels per stroke.
    - **Reward Function**: A bar chart mapping step reward.

To run them, simply launch your environment server in one terminal:
```bash
uv run python -m server.app
```
And execute the cells in any of the `watch_*.ipynb` files in Jupyter or your IDE!

## Project Structure

```
learn_handwriting/
├── __init__.py            # Exports: Action, Observation, State, Client
├── README.md              # This file
├── Dockerfile             # Container image (OpenEnv / HF Spaces)
├── client.py              # LearnHandwritingEnv WebSocket client
├── models.py              # LearnHandwritingAction / Observation / State
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies (package-dir maps learn_handwriting → .)
├── inference.py           # LLM inference loop (hackathon validator)
├── export_character_images.py  # Writes characters/*.png (reference renders from TASK_CHARACTERS)
├── preview_characters.py  # Renders targets + masks → images/ for inspection
├── characters/            # PNG exports (from export script; git may ignore or track)
├── images/                # Output of preview_characters.py
├── docs/                  # Extra notes (e.g. finetune format)
├── datagen_sft/           # SFT data generation pipeline
│   ├── oracle.py          # Stroke definitions (normalized bbox coords) for A–Z
│   ├── generator.py       # Combinatorial variation & jitter engine
│   ├── simulate_and_format.py  # Grounded trajectories → JSONL (uses reset(character=...))
│   ├── verify_readable.py # JSONL → grouped pretty-printed JSON per character
│   ├── test_l.py          # Optional L-only grid search (not the main export)
│   ├── qwen25_finetune_data.jsonl  # Generated ChatML SFT dataset (regenerate to refresh)
│   └── readable_samples/  # `*_sample.json` (human review; not the training source of truth)
├── visualizations/        # Watch notebooks and local debugging tools
│   ├── watch_runner.py
│   ├── watch_easy.ipynb
│   ├── watch_medium.ipynb
│   ├── watch_hard.ipynb
│   └── inference_local.ipynb
├── Roboto/
│   └── static/
│       └── Roboto-Bold.ttf   # Vendored font (Apache 2.0 / OFL)
├── tests/
│   ├── test_renderer.py   # All letters: shape, pixel ranges, masks
│   └── test_integrity.py  # Integrity triggers and “simple letter” checks
└── server/
    ├── __init__.py        # Server module exports
    ├── app.py             # FastAPI app (HTTP + WebSocket + Gradio UI)
    ├── renderer.py        # Font rendering + disqualification masks
    └── learn_handwriting_environment.py  # Core RL environment (TASK_CHARACTERS, step/reset)
```
