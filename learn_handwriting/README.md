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

A Reinforcement Learning environment where an agent learns to draw capital letters
by issuing geometric actions (lines, curves, circles, ellipses) on a 100×100 canvas. 
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
- **Font-based rendering** — characters are rendered from Roboto Bold at runtime; no static image files.
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
| `medium` | `A`, `N`, `Z`, `E` | 3–4 | Straight lines, 3+ strokes; `A` has integrity constraint |
| `hard` | `B`, `C`, `S`, `O`, `G`, `Q` | 1–3+ | Curves or enclosed counters; all have integrity constraints |

### Shape Integrity Constraints

Some characters have **protected regions** the agent must not fill in:

| Character | Protected Region | Disqualifier |
|---|---|---|
| `A` | Inner triangle counter | Flood-fill interior |
| `B` | Upper and lower lobe counters | Flood-fill interior (2 components) |
| `O` | Circle interior | Flood-fill interior |
| `C` | Right-side opening | `render(O) − render(C)` |
| `S` | Two bridge gaps | `render(8) − render(S)` |
| `G` | Right-side opening | `render(O) − render(G)` |
| `Q` | Circle interior (ring must stay open) | Flood-fill interior |

If the agent's canvas covers **> 60%** of any protected zone, the episode ends immediately with `integrity_violated=True` and `reward=0.0`.

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

```python
from learn_handwriting.models import LearnHandwritingAction
from learn_handwriting.server.learn_handwriting_environment import LearnHandwritingEnvironment

env = LearnHandwritingEnvironment()
obs = env.reset()
print(f"character={obs.target_character}")

obs = env.step(LearnHandwritingAction(action_type="line", x1=25, y1=10, x2=25, y2=85))
print(f"matched={obs.pixels_matched_this_stroke}, coverage={obs.match_percentage:.2%}, reward={obs.reward:.4f}")
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Locally

```bash
uvicorn server.app:app --reload
```

## Local Monitoring & Visualization (Jupyter)

The project includes **local watch notebooks** designed to help you visually debug the agent's behavior step-by-step. These are separated by difficulty level: `watch_easy.ipynb`, `watch_medium.ipynb`, and `watch_hard.ipynb`.

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
├── client.py              # LearnHandwritingEnv WebSocket client
├── models.py              # LearnHandwritingAction / Observation / State
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── inference.py           # LLM inference loop (hackathon validator)
├── visualizations/        # Watch notebooks and local debugging tools
│   ├── watch_runner.py        
│   ├── watch_easy.ipynb       
│   ├── watch_medium.ipynb     
│   └── watch_hard.ipynb       
├── Roboto/
│   └── static/
│       └── Roboto-Bold.ttf   # Vendored font (Apache 2.0 / OFL)
├── tests/
│   ├── test_renderer.py   # Renderer shape/pixel/mask tests
│   └── test_integrity.py  # Integrity disqualification tests
└── server/
    ├── __init__.py        # Server module exports
    ├── app.py             # FastAPI app (HTTP + WebSocket + Gradio UI)
    ├── renderer.py        # Font rendering + disqualification masks
    ├── learn_handwriting_environment.py  # Core RL environment logic
    └── Dockerfile         # Container image definition
```
