import json
import os
import sys
import textwrap
from typing import List, Dict, Any, Optional

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.learn_handwriting_environment import LearnHandwritingEnvironment, TASK_CHARACTERS, MAX_STROKES
from models import LearnHandwritingAction
from datagen_sft.generator import get_augmented_episodes
from datagen_sft.oracle import ORACLE_GEOMETRIES

# System Prompt MUST exactly match inference.py
MAX_DRAWN_MULTIPLIER = float(os.getenv("MAX_DRAWN_MULTIPLIER", "1.7"))
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
        Step: {step} / {MAX_STROKES}
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

def format_action_to_json(stroke: Dict[str, Any]) -> Dict[str, Any]:
    """Format oracle stroke to exact Assistant JSON string expected by the schema."""
    action_type = stroke["type"]
    points = stroke["points"]
    desc = stroke.get("desc", f"Drawing {action_type}")
    
    payload = {
        "reasoning": f"Drawing the {desc}",
        "action_type": action_type
    }
    
    if action_type == "line":
        payload["x1"] = points[0][0]
        payload["y1"] = points[0][1]
        payload["x2"] = points[1][0]
        payload["y2"] = points[1][1]
    elif action_type == "curve":
        payload["x1"] = points[0][0]
        payload["y1"] = points[0][1]
        payload["x2"] = points[1][0]
        payload["y2"] = points[1][1]
        payload["x3"] = points[2][0]
        payload["y3"] = points[2][1]
    elif action_type == "ellipse":
        payload["x1"] = points[0][0]
        payload["y1"] = points[0][1]
        payload["rx"] = points[1][0]
        payload["ry"] = points[1][1]
    elif action_type == "circle":
        payload["x1"] = points[0][0]
        payload["y1"] = points[0][1]
        payload["radius"] = points[1][0]
        
    return payload

def main():
    output_file = os.path.join(os.path.dirname(__file__), "qwen25_finetune_data.jsonl")
    f_out = open(output_file, "w", encoding="utf-8")
    
    env = LearnHandwritingEnvironment(task="easy")
    
    total_episodes_saved = 0
    total_episodes_attempted = 0
    
    for task_name, char_list in TASK_CHARACTERS.items():
        print(f"Processing Task Pool: {task_name}...")
        for char in char_list:
            # Re-seed the env so reset(task) can pick the char? Actually, LearnHandwritingEnvironment
            # picks random char from pool. We can force it by temporarily overriding the pool.
            # Pin ``character=char`` so reset(task=...) does not replace the pool then sample
            # a different letter (would misalign oracle strokes vs target).
            obs = env.reset(task=task_name, character=char)
            bbox = (obs.char_bbox_x1, obs.char_bbox_y1, obs.char_bbox_x2, obs.char_bbox_y2)
            
            # scale_and_jitter doubles each logical stroke; long oracles need more samples to hit 90%.
            n_logical = len(ORACLE_GEOMETRIES.get(char, []))
            if n_logical >= 7:
                num_augments = 20
            elif n_logical >= 6:
                num_augments = 12
            elif n_logical >= 4:
                num_augments = 12
            elif n_logical >= 2:
                num_augments = 15
            else:
                num_augments = 8
            episodes = get_augmented_episodes(char, bbox, num_augments=num_augments)
            print(f" - Character '{char}': Found {len(episodes)} augmented trajectories.")
            
            for ep_idx, generated_strokes in enumerate(episodes):
                total_episodes_attempted += 1
                
                obs = env.reset(task=task_name, character=char)
                history = []
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                success = False
                failed = False
                
                for step, stroke_def in enumerate(generated_strokes, start=1):
                    strokes_remaining = MAX_STROKES - step + 1
                    
                    user_prompt = build_user_prompt(
                        target_character=char,
                        step=step,
                        strokes_remaining=strokes_remaining,
                        match_percentage=obs.match_percentage,
                        last_pixels_matched=obs.pixels_matched_this_stroke,
                        last_pixels_wasted=obs.pixels_wasted_this_stroke,
                        ink_remaining=obs.ink_remaining,
                        last_reward=obs.reward,
                        history=history,
                        last_integrity_violated=obs.integrity_violated,
                        char_bbox=bbox
                    )
                    
                    messages.append({"role": "user", "content": user_prompt})
                    
                    action_payload = format_action_to_json(stroke_def)
                    # Assistant responds with exact JSON string representation
                    messages.append({"role": "assistant", "content": json.dumps(action_payload)})
                    
                    # Apply action to environment
                    env_payload = action_payload.copy()
                    env_payload.pop("reasoning", None)
                    env_action = LearnHandwritingAction(**env_payload)
                    obs = env.step(env_action)
                    
                    # Update history for next prompt
                    action_str = f"{action_payload['action_type']}(...)" # Simplified for history trace
                    history.append(f"Step {step}: {action_str} → matched={obs.pixels_matched_this_stroke}px reward={obs.reward:.4f} coverage={obs.match_percentage:.1%}")
                    
                    if obs.integrity_violated or obs.ink_remaining <= 0:
                        failed = True
                        break
                        
                    if obs.match_percentage >= 0.90:
                        success = True
                        break

                if success and not failed:
                    metadata = {
                        "target_character": char,
                        "task": task_name,
                        "final_match_percentage": obs.match_percentage,
                        "num_strokes": len(generated_strokes)
                    }
                    data_row = {
                        "messages": messages,
                        "metadata": metadata
                    }
                    f_out.write(json.dumps(data_row) + "\n")
                    total_episodes_saved += 1
                    
            print(f"   -> Saved successful episodes: {total_episodes_saved} / {total_episodes_attempted}")

    f_out.close()
    print(f"\\nDone! Saved {total_episodes_saved} highly-optimised episodes to {output_file}.")

if __name__ == "__main__":
    main()
