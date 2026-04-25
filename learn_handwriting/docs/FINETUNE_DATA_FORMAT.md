# Fine-tuning data format: Learn Handwriting environment

This document specifies the **dataset shape** needed to fine-tune a model (e.g. Qwen2.5 Instruct with LoRA) so it can act as the stroke policy for the Learn Handwriting RL environment. It is aligned with `models.py`, `server/learn_handwriting_environment.py`, and `inference.py`.

---

## 1. What is being learned

- The model does **not** receive a raster image of the canvas in the observation. It only sees **text**: target letter, bounding box, step counts, coverage, ink budget, and feedback from the **last** stroke.
- Each model turn outputs **one stroke** as a **single JSON object** (same contract as the live `inference.py` client).
- Training data should be **multi-turn**: one user message per step (after the previous assistant stroke), one assistant message = one stroke, until the episode ends or success.

---

## 2. Episode structure (conversation)

Recommended container: **one training example = one full episode** as a `messages` array (ChatML / OpenAI style).

| Role         | When | Content |
|--------------|------|-----------|
| `system`     | Once at start | Same rules as `SYSTEM_PROMPT` in `inference.py` (canvas 100×100, coordinates, action types, 15-stroke cap, 90% coverage goal, ink limit, shape-integrity protected regions). |
| `user`       | After `reset` and after each `env.step` | The **current** observation encoded as text: target character, bbox, step index, strokes remaining, match %, last stroke’s matched/wasted pixels, ink remaining, optional integrity warning, short action history. |
| `assistant`  | Once per step | A **JSON object** describing the **next** stroke (see §3). No markdown fences unless your trainer strips them. |

- **First user turn** corresponds to **step 1** (before any stroke in that episode), using whatever scalars you use for “no previous stroke” (e.g. 0 matched/wasted, initial ink from `reset`).
- **Subsequent user turns** use the **observation returned by** `env.step` **after** the previous assistant action.

Your fine-tuning framework may expect a wrapper field (e.g. `{"messages": [...]}`) or a specific template (Llama, Qwen, etc.); the **semantic** contract below stays the same.

---

## 3. Assistant message: JSON stroke schema

The assistant must emit valid JSON with these keys. Types and ranges must match the environment.

### Required

- `reasoning` (string): Short rationale (chain-of-thought for SFT; can be a few words).
- `action_type` (string, enum): One of `"line"`, `"curve"`, `"circle"`, `"ellipse"`.
- `x1` (int): 0–99.
- `y1` (int): 0–99.

### By `action_type`

| `action_type` | Required extra fields | Notes |
|---------------|----------------------|--------|
| `line`        | `x2`, `y2` (int, 0–99) | Segment from (x1,y1) to (x2,y2). |
| `curve`       | `x2`, `y2`, `x3`, `y3` (int, 0–99) | Quadratic Bezier: endpoints (x1,y1)–(x2,y2), curve passes through (x3,y3). |
| `circle`      | `radius` (int, 1–100) | Center (x1,y1). |
| `ellipse`     | `rx`, `ry` (int, 1–100) | Center (x1,y1), axis-aligned. |

- Omit or set to `null` any field not used by the chosen `action_type` (e.g. no `x2` for `circle`).

**Coordinate system:** Image coordinates: `x` left→right, `y` top→bottom, origin top-left; integers only.

**Rendering:** The server draws with fixed brush width (`BRUSH_WIDTH` in `server/renderer.py`); the model does not control thickness.

---

## 4. User message: which fields to encode

Encode the same information the agent is allowed to see in production (`LearnHandwritingObservation`):

- `target_character` (e.g. `"A"`).
- `char_bbox_x1`, `char_bbox_y1`, `char_bbox_x2`, `char_bbox_y2` (tight bbox of the rendered target on the 100×100 grid).
- Step index and limit (e.g. step 1 of 15).
- Strokes used or “actions remaining” (consistent with your prompt template).
- `match_percentage` (0.0–1.0) — current cumulative coverage of target pixels.
- `pixels_matched_this_stroke` / `pixels_wasted_this_stroke` — feedback for the **last** action (0 for the first user turn if you define it that way).
- `ink_remaining` and/or `max_allowed_pixels` (from observation).
- `integrity_violated` — if true, the episode is done; such turns are usually not used for “next stroke” training, or used as negatives.

You may copy the **exact** wording of `build_user_prompt()` in `inference.py` so that fine-tuned and inference prompts match.

**Explicitly not in the observation (do not require the model to predict from a hidden image):** the full `canvas` matrix. If you add canvas to training only, the live env will not provide it — avoid that mismatch unless you change the environment.

---

## 5. Per-example metadata (recommended)

Store alongside `messages` for filtering and analysis:

- `target_character`, `task` (`easy` / `medium` / `hard` if you follow the env pools),
- `final_match_percentage` (or success flag ≥ 0.90),
- `num_strokes`,
- `integrity_violated` at end (false for “good” demos),
- optional `episode_id`, `source` (e.g. expert / augmented).

---

## 6. Quality requirements for “positive” examples

- End state: **`match_percentage` ≥ 0.90** (env success threshold) unless you intentionally include failure examples for contrast.
- **No integrity violation** on the winning trajectory (or filter failing episodes for the main SFT set).
- **Ink budget:** total drawn pixels on canvas must not exceed the env’s cap (default `MAX_DRAWN_MULTIPLIER` × target pixel count, see environment).
- Strokes: **≤ 15** per episode.

---

## 7. File format for tooling

- **JSONL:** one JSON object per line, each with at least `messages` (and optional metadata fields).
- Ensure **UTF-8** and **valid JSON** (escape newlines inside strings; no raw line breaks in assistant content unless your template allows it).
- If the trainer requires a single `text` field, concatenate system/user/assistant with the chat template the **same** way you will at inference (Qwen, Llama, etc.).

---

## 8. Augmentation (optional)

- **Coordinate jitter** in valid range (e.g. ±2–3 pixels) re-simulated in the environment preserves schema but increases diversity; drop examples that fail coverage or integrity.
- **Per-character balance** — match the character pools in `TASK_CHARACTERS` if you want uniform difficulty.

---

## 9. Reference implementation

- **Action / observation types:** `models.py`
- **Prompts and JSON parsing:** `inference.py` (`StrokeOutput`, `build_user_prompt`, `SYSTEM_PROMPT`)
- **Step semantics and limits:** `server/learn_handwriting_environment.py`
- **Integrity masks and rendering:** `server/renderer.py`

Keeping dataset prompts and JSON keys aligned with these files reduces train–serve skew when you deploy the same model in `inference.py`.
