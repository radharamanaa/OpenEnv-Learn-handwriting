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
from typing import List, Optional

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
TEMPERATURE = 0.7
SUCCESS_SCORE_THRESHOLD = 0.90  # 90% pixel coverage required


# ── Structured output schema ────────────────────────────────────────────────

class StrokeOutput(BaseModel):
    """Structured stroke action output from the LLM."""
    reasoning: str = Field(description="Brief explanation of why this stroke was chosen")
    x1: int = Field(description="Stroke start x coordinate (0–99, left→right)")
    y1: int = Field(description="Stroke start y coordinate (0–99, top→bottom)")
    x2: int = Field(description="Stroke end x coordinate (0–99, left→right)")
    y2: int = Field(description="Stroke end y coordinate (0–99, top→bottom)")
    width: int = Field(description="Brush width in pixels (1–10)")


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are drawing capital letters on a 100×100 pixel canvas using straight strokes.

    Canvas coordinate system:
    - x: 0 (left) → 99 (right)
    - y: 0 (top)  → 99 (bottom)
    - Origin (0,0) is the TOP-LEFT corner

    Each action is one straight stroke: a line from (x1,y1) to (x2,y2) with a brush of given width (1–10px).
    The stroke pixels are compared against the target character image to compute reward.

    Rules:
    - You have at most 15 strokes per episode.
    - Goal: cover 90% of the target character's white pixels.
    - Use wider strokes (width 6–8) for thick bars, narrower (width 3–4) for thin parts.
    - Think about the skeleton of the letter and plan strokes along its main lines.
    - All coordinates must be integers between 0 and 99.
    - Width must be an integer between 1 and 10.

    You MUST respond with a valid JSON object and nothing else. No markdown, no explanation outside the JSON.
    Format:
    {"reasoning": "<brief explanation>", "x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "width": <int>}
""").strip()


def build_user_prompt(
    target_character: str,
    step: int,
    strokes_remaining: int,
    match_percentage: float,
    last_pixels_matched: int,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-5:]) if history else "None yet"
    return textwrap.dedent(f"""
        Draw the capital letter: {target_character}

        Step: {step} / {MAX_STEPS}
        Strokes remaining: {strokes_remaining}
        Current coverage: {match_percentage:.1%} (goal: 90%)
        Last stroke matched: {last_pixels_matched} pixels (reward: {last_reward:.4f})

        Stroke history (most recent last):
        {history_block}

        Plan your next stroke to maximise coverage of letter '{target_character}'.
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
    last_reward: float,
    history: List[str],
) -> StrokeOutput:
    user_prompt = build_user_prompt(
        target_character, step, strokes_remaining,
        match_percentage, last_pixels_matched, last_reward, history,
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
            x1=int(data.get("x1", 10)),
            y1=int(data.get("y1", 10)),
            x2=int(data.get("x2", 90)),
            y2=int(data.get("y2", 90)),
            width=int(data.get("width", 5)),
        )
        # Clamp to valid ranges defensively
        stroke.x1 = max(0, min(99, stroke.x1))
        stroke.y1 = max(0, min(99, stroke.y1))
        stroke.x2 = max(0, min(99, stroke.x2))
        stroke.y2 = max(0, min(99, stroke.y2))
        stroke.width = max(1, min(10, stroke.width))
        return stroke
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Safe fallback: diagonal stroke across the canvas
        return StrokeOutput(reasoning="fallback", x1=10, y1=10, x2=90, y2=90, width=5)


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
    score = 0.0
    success = False
    match_percentage = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        target_character = obs.target_character
        last_pixels_matched = 0
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            strokes_remaining = MAX_STEPS - step + 1
            stroke = get_stroke(
                client, step, strokes_remaining, target_character,
                obs.match_percentage, last_pixels_matched, last_reward, history,
            )

            action_str = f"stroke({stroke.x1},{stroke.y1},{stroke.x2},{stroke.y2},w={stroke.width})"
            result = await env.step(LearnHandwritingAction(
                x1=stroke.x1, y1=stroke.y1,
                x2=stroke.x2, y2=stroke.y2,
                width=stroke.width,
            ))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            match_percentage = obs.match_percentage
            last_pixels_matched = obs.pixels_matched_this_stroke
            last_reward = reward

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(
                f"Step {step}: {action_str} → matched={last_pixels_matched}px "
                f"reward={reward:.4f} coverage={match_percentage:.1%}"
            )

            if done:
                break

        score = min(max(match_percentage, 0.0), 1.0)
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