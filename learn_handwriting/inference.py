"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import json
import os
import textwrap
from typing import List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# All task difficulties to run — produces 3 graded tasks required by the validator
ALL_TASKS = ["easy", "medium", "hard"]
BENCHMARK = "learn_handwriting"
MAX_STEPS = 15
MAX_DRAWN_MULTIPLIER = os.getenv("MAX_DRAWN_MULTIPLIER", "1.7")
TEMPERATURE = 0.7
SUCCESS_SCORE_THRESHOLD = 0.90  # 90% pixel coverage required
SCORE_MIN = 0.001  # grader rejects score == 0.0  (must be strictly > 0)
SCORE_MAX = 0.999  # grader rejects score == 1.0  (must be strictly < 1)


# ── Structured output schema ────────────────────────────────────────────────

class StrokeOutput(BaseModel):
    """Structured stroke action output from the LLM."""
    reasoning: str = Field(description="Brief explanation of why this action and parameters were chosen")
    action_type: Literal["line", "curve", "circle", "ellipse"] = Field(description="Type of action: 'line', 'curve', 'circle', or 'ellipse'")
    x1: int = Field(description="For 'line' and 'curve': Start x. For 'circle' and 'ellipse': Center x.")
    y1: int = Field(description="For 'line' and 'curve': Start y. For 'circle' and 'ellipse': Center y.")
    x2: Optional[int] = Field(default=None, description="For 'line' and 'curve': End x. Ignored for 'circle' and 'ellipse'.")
    y2: Optional[int] = Field(default=None, description="For 'line' and 'curve': End y. Ignored for 'circle' and 'ellipse'.")
    x3: Optional[int] = Field(default=None, description="For 'curve' only: Pass-through midpoint x. Ignored for others.")
    y3: Optional[int] = Field(default=None, description="For 'curve' only: Pass-through midpoint y. Ignored for others.")
    radius: Optional[int] = Field(default=None, description="For 'circle' only: Radius. Ignored for 'ellipse' and others.")
    rx: Optional[int] = Field(default=None, description="For 'ellipse' only: Horizontal radius. Ignored for 'circle' and others.")
    ry: Optional[int] = Field(default=None, description="For 'ellipse' only: Vertical radius. Ignored for 'circle' and others.")


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(f"""
    You are drawing capital letters on a 100×100 pixel canvas.

    Canvas coordinate system:
    - x: 0 (left) → 99 (right)
    - y: 0 (top)  → 99 (bottom)
    - Origin (0,0) is the TOP-LEFT corner

    You can draw lines, curves, circles, and ellipses using these action types:
    - "line": Straight line from (x1,y1) to (x2,y2).
    - "curve": A curve starting at (x1,y1), ending at (x2,y2), and passing through a midpoint at (x3,y3).
    - "circle": A circle centered at (x1,y1) with given `radius`.
    - "ellipse": An upright oval centered at (x1,y1) with horizontal radius `rx` and vertical radius `ry`.

    Rules:
    - You have at most 15 actions per episode.
    - Goal: cover 90% of the target character's white pixels.
    - INK PENALTY: You will fail the episode instantly if you draw more than {MAX_DRAWN_MULTIPLIER}x the target's total pixels. Do not waste ink!
    - Coordinates are integers 0-99.

    SHAPE INTEGRITY — protected regions you must NOT fill in:
    - A: inner triangle hole (the counter between the two legs and crossbar)
    - B: two enclosed lobe holes (upper and lower bumps)
    - D: interior of the D bowl (semicircle counter — outline only, like O)
    - O: circle interior (do not fill the hole)
    - P: bowl interior (do not fill the hole in the loop)
    - R: bowl interior (do not fill the hole above the diagonal leg)
    - C: right-side opening (do not close it — that would make O)
    - S: two bridge gaps (do not connect the loops — that would make 8)
    - G: right-side opening (do not close it — that would make O)
    - Q: circle interior (same as O — do not fill the hole; draw the tail as a separate stroke)
    If you cover more than 60% of a protected region the episode ends immediately
    with reward=0 and integrity_violated=True. Plan your strokes to follow the
    character outline only, never filling in holes or closing open gaps.

    You MUST respond with a valid JSON object and nothing else. No markdown, no explanation outside the JSON.

    Sample Invocations:
    - To draw a vertical line on the left side:
      {{"reasoning": "Drawing the vertical spine of the letter D", "action_type": "line", "x1": 20, "y1": 10, "x2": 20, "y2": 90}}

    - To draw a curved right side of a D (starts top, ends bottom, bows out to x=80):
      {{"reasoning": "Drawing the curved belly of the letter D", "action_type": "curve", "x1": 20, "y1": 10, "x2": 20, "y2": 90, "x3": 80, "y3": 50}}

    - To draw a perfect circle for an O:
      {{"reasoning": "Drawing the letter O using a circle centered in the canvas", "action_type": "circle", "x1": 50, "y1": 50, "radius": 40}}

    - To draw a tall, narrow oval:
      {{"reasoning": "Drawing a tall vertical ellipse", "action_type": "ellipse", "x1": 50, "y1": 50, "rx": 20, "ry": 40}}
""").strip()


def build_user_prompt(
    target_character: str,
    step: int,
    strokes_remaining: int,
    match_percentage: float,
    last_pixels_matched: int,
    last_pixels_wasted: int,
    ink_remaining: int,
    last_reward: float,
    history: List[str],
    last_integrity_violated: bool = False,
    char_bbox: Optional[tuple[int, int, int, int]] = None,
) -> str:
    history_block = "\n".join(history[-5:]) if history else "None yet"
    integrity_warning = (
        "\n⚠️  LAST ACTION VIOLATED SHAPE INTEGRITY — episode ended with reward=0."
        if last_integrity_violated else ""
    )
    bbox_info = (
        f"Bounding Box: x=[{char_bbox[0]}→{char_bbox[2]}], y=[{char_bbox[1]}→{char_bbox[3]}]\n"
        if char_bbox else ""
    )
    return textwrap.dedent(f"""
        Draw the capital letter: {target_character}
        {bbox_info}
        Step: {step} / {MAX_STEPS}
        Actions remaining: {strokes_remaining}
        Current coverage: {match_percentage:.1%} (goal: 90%)

        FEEDBACK ON LAST ACTION:
        - Pixels matched: {last_pixels_matched}
        - Pixels wasted (missed target): {last_pixels_wasted}
        - Ink remaining before failure: {ink_remaining}{integrity_warning}

        Action history:
        {history_block}

        Plan your next action to maximise coverage of '{target_character}' without wasting ink or violating shape integrity.
    """).strip()


# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM call with structured output ─────────────────────────────────────────

def get_stroke(
    client: OpenAI,
    step: int,
    strokes_remaining: int,
    target_character: str,
    match_percentage: float,
    last_pixels_matched: int,
    last_pixels_wasted: int,
    ink_remaining: int,
    last_reward: float,
    history: List[str],
) -> StrokeOutput:
    user_prompt = build_user_prompt(
        target_character, step, strokes_remaining,
        match_percentage, last_pixels_matched, last_pixels_wasted, ink_remaining,
        last_reward, history,
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        stroke = StrokeOutput(
            reasoning=data.get("reasoning", ""),
            action_type=data.get("action_type", "line"),
            x1=int(data.get("x1", 10)),
            y1=int(data.get("y1", 10)),
            x2=int(data.get("x2")) if data.get("x2") is not None else None,
            y2=int(data.get("y2")) if data.get("y2") is not None else None,
            x3=int(data.get("x3")) if data.get("x3") is not None else None,
            y3=int(data.get("y3")) if data.get("y3") is not None else None,
            radius=int(data.get("radius")) if data.get("radius") is not None else None,
            rx=int(data.get("rx")) if data.get("rx") is not None else None,
            ry=int(data.get("ry")) if data.get("ry") is not None else None,
        )
        # Clamp to valid ranges defensively
        stroke.x1 = max(0, min(99, stroke.x1))
        stroke.y1 = max(0, min(99, stroke.y1))
        if stroke.x2 is not None: stroke.x2 = max(0, min(99, stroke.x2))
        if stroke.y2 is not None: stroke.y2 = max(0, min(99, stroke.y2))
        if stroke.x3 is not None: stroke.x3 = max(0, min(99, stroke.x3))
        if stroke.y3 is not None: stroke.y3 = max(0, min(99, stroke.y3))
        if stroke.radius is not None: stroke.radius = max(1, min(100, stroke.radius))
        if stroke.rx is not None: stroke.rx = max(1, min(100, stroke.rx))
        if stroke.ry is not None: stroke.ry = max(1, min(100, stroke.ry))
        return stroke
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Safe fallback: diagonal line across the canvas
        return StrokeOutput(reasoning="fallback", action_type="line", x1=10, y1=10, x2=90, y2=90)


# ── Single-task episode loop ─────────────────────────────────────────────────

async def run_task(task_name: str, env: LearnHandwritingEnv, client: OpenAI) -> float:
    """Run one full episode for the given task difficulty and return the final score.

    Emits exactly one [START] line, one [STEP] line per stroke, and one [END] line.

    Args:
        task_name: Difficulty level — "easy", "medium", or "hard".
        env:       Shared environment client (connection is reused across tasks).
        client:    OpenAI client for LLM stroke generation.

    Returns:
        Final score (match_percentage clamped to [0, 1]).
    """
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = SCORE_MIN  # default to minimum valid score; updated after each episode
    success = False
    match_percentage = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        target_character = obs.target_character
        last_pixels_matched = 0
        last_pixels_wasted = 0
        ink_remaining = getattr(obs, "ink_remaining", 1000)
        last_reward = 0.0
        last_integrity_violated = False

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            strokes_remaining = MAX_STEPS - step + 1
            bbox = (obs.char_bbox_x1, obs.char_bbox_y1, obs.char_bbox_x2, obs.char_bbox_y2)
            stroke = get_stroke(
                client, step, strokes_remaining, target_character,
                obs.match_percentage, last_pixels_matched, last_pixels_wasted,
                ink_remaining, last_reward, history,
                last_integrity_violated, bbox,
            )

            if stroke.action_type == "circle":
                action_str = f"circle({stroke.x1},{stroke.y1},r={stroke.radius})"
            elif stroke.action_type == "ellipse":
                action_str = f"ellipse({stroke.x1},{stroke.y1},rx={stroke.rx},ry={stroke.ry})"
            elif stroke.action_type == "curve":
                action_str = f"curve({stroke.x1},{stroke.y1},{stroke.x2},{stroke.y2},{stroke.x3},{stroke.y3})"
            else:
                action_str = f"line({stroke.x1},{stroke.y1},{stroke.x2},{stroke.y2})"

            result = await env.step(LearnHandwritingAction(
                action_type=stroke.action_type,
                x1=stroke.x1, y1=stroke.y1,
                x2=stroke.x2, y2=stroke.y2,
                x3=stroke.x3, y3=stroke.y3,
                radius=stroke.radius,
                rx=stroke.rx, ry=stroke.ry,
            ))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            match_percentage = obs.match_percentage
            last_pixels_matched = obs.pixels_matched_this_stroke
            last_pixels_wasted = getattr(obs, "pixels_wasted_this_stroke", 0)
            ink_remaining = getattr(obs, "ink_remaining", 0)
            last_integrity_violated = getattr(obs, "integrity_violated", False)
            last_reward = reward

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            integrity_tag = " [INTEGRITY VIOLATED]" if last_integrity_violated else ""
            history.append(
                f"Step {step}: {action_str} → matched={last_pixels_matched}px "
                f"reward={reward:.4f} coverage={match_percentage:.1%}{integrity_tag}"
            )

            if done:
                break

        score = min(max(match_percentage, SCORE_MIN), SCORE_MAX)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] run_task({task_name}) error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        print(f"\n{'='*50}", flush=True)
        print(f"  Task        : {task_name}", flush=True)
        print(f"  Model       : {MODEL_NAME}", flush=True)
        print(f"  Total Steps : {steps_taken} / {MAX_STEPS}", flush=True)
        print(f"  Total Reward: {sum(rewards):.4f}", flush=True)
        print(f"  Final Score : {score:.2%}", flush=True)
        print(f"  Success     : {success}", flush=True)
        print(f"{'='*50}\n", flush=True)

    return score


# ── Main: run all 3 tasks to satisfy the ≥3 graders requirement ──────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name in ALL_TASKS:
        # Create a fresh WebSocket connection per task.
        # A single shared connection drops after the first episode ends,
        # causing "no close frame received or sent" for subsequent tasks.
        env = (
            await LearnHandwritingEnv.from_docker_image(IMAGE_NAME)
            if IMAGE_NAME
            else LearnHandwritingEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))
        )
        try:
            await run_task(task_name, env, client)
        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())