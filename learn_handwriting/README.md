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
by issuing strokes on a 100×100 canvas. The agent is rewarded for each stroke that
overlaps with the target character's white pixels, and the episode succeeds when
90% of the target is covered within 15 strokes.

## How It Works

### Overview

On every `reset()` a random capital letter (A, B, C, L, O, V, Z) is chosen as the
target. The environment loads a pre-processed 100×100 binary image of that letter
where every pixel is either **0** (background) or **240** (stroke), treated as **1**
internally for all calculations.

The agent then issues strokes one at a time. Each stroke is a straight line from
`(x1, y1)` to `(x2, y2)` on the canvas coordinate system (0–99). The environment
draws the line using **Bresenham's line algorithm**, computes how many pixels of
that stroke land on white pixels of the target character, and returns a reward.

### Stroke Workflow (per step)

```
1. temp_matrix = zeros(100×100)
2. Draw Bresenham line on temp_matrix  →  stroke pixels set to 1

3. pixels_matched_this_stroke = intersection(temp_matrix, target_matrix)
   reward = pixels_matched_this_stroke / total_target_pixels   ← range 0.0–1.0

4. canvas = max(canvas, temp_matrix)   ← merge stroke into cumulative canvas

5. total_matched_pixels = intersection(canvas, target_matrix)
   match_percentage = total_matched_pixels / total_target_pixels

6. strokes_used += 1

7. done = (match_percentage >= 0.90) OR (strokes_used >= 15)
```

Key design decisions:
- **Reward is per-stroke only** — previous strokes do not inflate the current reward
- **Canvas merges cumulatively** — pixels already drawn are preserved across strokes
- **Done on success OR exhaustion** — episode ends at 90% coverage or 15 strokes used

### Target Character Images

All target images live in `characters/` and are pre-processed:
- Resized to exactly **100×100 pixels**, grayscale
- Binarised: pixels **> 50 → 240**, else **0**
- Rendered with a large bold font so strokes are **8–15 pixels thick**,
  giving ~1500–3000 white pixels per character for meaningful reward signals

## Quick Start

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

try:
    env = LearnHandwritingEnv.from_docker_image("learn_handwriting-env:latest")

    result = env.reset()
    print(f"Draw character: {result.observation.target_character}")

    # Stroke 1: vertical bar (e.g. left side of L)
    result = env.step(LearnHandwritingAction(x1=25, y1=10, x2=25, y2=85))
    print(f"Matched this stroke : {result.observation.pixels_matched_this_stroke}")
    print(f"Cumulative coverage : {result.observation.match_percentage:.1%}")
    print(f"Reward              : {result.reward:.4f}")

    # Stroke 2: horizontal bar (e.g. bottom of L)
    result = env.step(LearnHandwritingAction(x1=25, y1=85, x2=75, y2=85))
    print(f"Matched this stroke : {result.observation.pixels_matched_this_stroke}")
    print(f"Cumulative coverage : {result.observation.match_percentage:.1%}")
    print(f"Done                : {result.done}")

finally:
    env.close()
```

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t learn_handwriting-env:latest -f server/Dockerfile .
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

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action — `LearnHandwritingAction`

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `x1`  | int  | 0–99  | Stroke start x coordinate |
| `y1`  | int  | 0–99  | Stroke start y coordinate |
| `x2`  | int  | 0–99  | Stroke end x coordinate   |
| `y2`  | int  | 0–99  | Stroke end y coordinate   |

### Observation — `LearnHandwritingObservation`

| Field | Type | Description |
|-------|------|-------------|
| `target_character`          | str   | Capital letter to draw (e.g. `"L"`) |
| `strokes_used`              | int   | Strokes consumed so far |
| `pixels_matched_this_stroke`| int   | Raw pixel overlap for this stroke only |
| `total_matched_pixels`      | int   | Cumulative pixels covered across all strokes |
| `match_percentage`          | float | `total_matched / total_target` — progress toward 90% |
| `reward`                    | float | `pixels_matched_this_stroke / total_target_pixels` (0.0–1.0) |
| `done`                      | bool  | True when success or stroke limit reached |

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
- A perfectly aimed stroke through a thick character stroke (~80px long) on a
  character with ~2000 white pixels yields a reward of `80 / 2000 = 0.04`

### Done Conditions

| Condition | Outcome |
|-----------|---------|
| `match_percentage >= 0.90` | ✅ Success — agent covered 90% of the character |
| `strokes_used >= 15`       | ❌ Failure — stroke budget exhausted |

## Advanced Usage

### Connecting to an Existing Server

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

env = LearnHandwritingEnv(base_url="<ENV_HTTP_URL_HERE>")
result = env.reset()
result = env.step(LearnHandwritingAction(x1=10, y1=10, x2=80, y2=80))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

```python
from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

with LearnHandwritingEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print(f"Draw: {obs.observation.target_character}")
    result = env.step(LearnHandwritingAction(x1=25, y1=10, x2=25, y2=85))
    print(f"Coverage: {result.observation.match_percentage:.1%}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains canvas and stroke state
- **Efficient for episodes**: Better for many sequential steps

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

obs = env.step(LearnHandwritingAction(x1=25, y1=10, x2=25, y2=85))
print(f"matched={obs.pixels_matched_this_stroke}, coverage={obs.match_percentage:.2%}, reward={obs.reward:.4f}")
```

### Running Locally

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
learn_handwriting/
├── __init__.py            # Exports: Action, Observation, State, Client
├── README.md              # This file
├── client.py              # LearnHandwritingEnv WebSocket client
├── models.py              # LearnHandwritingAction / Observation / State
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── characters/            # Pre-processed 100×100 binary character images
│   ├── A.jpg
│   ├── B.jpg
│   ├── C.jpg
│   ├── L.jpg
│   ├── O.jpg
│   ├── V.jpg
│   └── Z.jpg
└── server/
    ├── __init__.py        # Server module exports
    ├── app.py             # FastAPI app (HTTP + WebSocket endpoints)
    ├── learn_handwriting_environment.py  # Core RL environment logic
    └── Dockerfile         # Container image definition
```
