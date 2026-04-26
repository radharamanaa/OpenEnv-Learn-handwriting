"""
Shared watch runner for the Learn Handwriting environment.

Runs a single episode for the given task difficulty, displaying the target
character and the agent's canvas side-by-side in a matplotlib window.
The canvas is updated after every stroke so you can watch the model draw.

Usage (via the task-specific entry-point scripts):
    python watch_easy.py
    python watch_medium.py
    python watch_hard.py
"""

import asyncio
import json
import os
import textwrap
from typing import List, Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from learn_handwriting import LearnHandwritingAction, LearnHandwritingEnv
from server.renderer import BRUSH_WIDTH, render_target_character

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS     = 15
TEMPERATURE   = 0.7
STROKE_DELAY  = 1.5   # seconds to pause after each stroke update
MAX_DRAWN_MULTIPLIER = float(os.getenv("MAX_DRAWN_MULTIPLIER", "1.7"))
PRINT_LLM_PROMPT = os.getenv("PRINT_LLM_PROMPT", "true").lower() in ("true", "1", "yes")


def get_watch_client():
    """Returns either a LocalModelClient or an OpenAI API client based on MODEL_NAME."""
    is_local = "/" in (MODEL_NAME or "") or os.path.exists(MODEL_NAME or "")
    if is_local:
        return LocalModelClient(MODEL_NAME, revision=MODEL_REVISION)
    else:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── LLM schema ────────────────────────────────────────────────────────────────

class StrokeOutput(BaseModel):
    """Structured stroke action returned by the LLM."""
    reasoning:   str = Field(default="")
    action_type: Literal["line", "curve", "circle", "ellipse"] = "line"
    x1: int = 10;  y1: int = 10
    x2: Optional[int] = None;  y2: Optional[int] = None
    x3: Optional[int] = None;  y3: Optional[int] = None
    radius: Optional[int] = None
    rx: Optional[int] = None;  ry: Optional[int] = None


SYSTEM_PROMPT = textwrap.dedent(f"""
    You are drawing capital letters on a 100x100 pixel canvas.
    Coordinate system: x=0 left → x=99 right, y=0 top → y=99 bottom.

    Actions:
    - "line"   : straight line from (x1,y1) to (x2,y2)
    - "curve"  : bezier curve (x1,y1) -> (x2,y2) passing through (x3,y3)
    - "circle" : circle centred at (x1,y1) with `radius`
    - "ellipse": oval centred at (x1,y1) with horizontal `rx`, vertical `ry`

    Rules:
    - Max {MAX_STEPS} actions. Goal: cover 90% of the target character's pixels.
    - INK PENALTY: fail instantly if you draw more than {MAX_DRAWN_MULTIPLIER}x the target pixels.
    - CRITICAL for circle/ellipse: centre ± radius must stay within [0,99].
      Safe example: ellipse centre=(50,50) rx=30 ry=35

    SHAPE INTEGRITY — do NOT fill these regions:
    A=inner triangle, B=two lobe holes, O/Q=circle interior,
    C/G=right-side opening, S=two bridge gaps.
    Covering >60% of a protected region ends the episode with reward=0.

    Respond ONLY with a valid JSON object, no markdown.
""").strip()


# ── LLM call ──────────────────────────────────────────────────────────────────

def _action_str(s: StrokeOutput) -> str:
    if s.action_type == "circle":
        return f"circle({s.x1},{s.y1},r={s.radius})"
    if s.action_type == "ellipse":
        return f"ellipse({s.x1},{s.y1},rx={s.rx},ry={s.ry})"
    if s.action_type == "curve":
        return f"curve({s.x1},{s.y1}->{s.x2},{s.y2})"
    return f"line({s.x1},{s.y1}->{s.x2},{s.y2})"


# ── Local Model Client (Mimics OpenAI for Transformers) ──────────────────────

class LocalModelClient:
    def __init__(self, model_id_or_path: str, revision: str = "main"):
        # --- Device Discovery ---
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

        print(f"\n" + "="*40)
        print(f"DEVICE DISCOVERY (Local Inference)")
        print(f"Detected device: {device_type.upper()}")
        if device_type == "cuda":
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device_type == "mps":
            print(f"  Apple Silicon GPU detected (MPS)")
            torch_dtype = torch.float16
        else:
            print(f"  WARNING: No GPU detected. Local inference will be slow.")
            torch_dtype = torch.float32
        print("="*40 + "\n")
        # ------------------------

        print(f"📦 Loading local/HF model: {model_id_or_path} (revision: {revision})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True, revision=revision)
        
        # Check if it's a PEFT model or base model
        try:
            # Try loading as a PEFT model first
            base_model_id = "Qwen/Qwen2.5-7B-Instruct" # Default base
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, 
                torch_dtype=torch_dtype, 
                device_map="auto", 
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, model_id_or_path, revision=revision)
            print("✅ Loaded as LoRA adapter.")
        except Exception:
            # Fallback to standard loading
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path, 
                torch_dtype=torch_dtype, 
                device_map="auto", 
                trust_remote_code=True,
                revision=revision
            )
            print("✅ Loaded as standalone model.")
        
        self.model.eval()
        self.chat_completions = self # Mimic structure: client.chat.completions.create

    def create(self, model: str, messages: list, temperature: float, response_format: dict):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0
            )
        
        content = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        
        # Wrap in a mock response object
        class MockChoice:
            def __init__(self, c): self.message = type('obj', (object,), {'content': c})
        class MockResponse:
            def __init__(self, c): self.choices = [MockChoice(c)]
        
        return MockResponse(content)


def get_stroke(
    client: OpenAI | LocalModelClient,
    step: int,
    target_char: str,
    match_pct: float,
    last_matched: int,
    last_wasted: int,
    ink_remaining: int,
    last_reward: float,
    history: List[str],
    integrity_violated: bool = False,
    char_bbox: Optional[tuple[int, int, int, int]] = None,
) -> StrokeOutput:
    integrity_warn = "\n⚠️ LAST ACTION VIOLATED INTEGRITY!" if integrity_violated else ""
    history_block  = "\n".join(history[-5:]) if history else "None yet"
    bbox_info = (
        f"Bounding Box: x=[{char_bbox[0]}→{char_bbox[2]}], y=[{char_bbox[1]}→{char_bbox[3]}]\n        "
        if char_bbox else ""
    )
    user_msg = textwrap.dedent(f"""
        Draw capital letter: {target_char}
        {bbox_info}Step {step}/{MAX_STEPS}  |  Coverage: {match_pct:.1%}  (goal 90%)
        Last action: matched={last_matched}px  wasted={last_wasted}px  ink_left={ink_remaining}{integrity_warn}
        History (last 5):
        {history_block}
    """).strip()

    if PRINT_LLM_PROMPT:
        print(f"\n{'='*50}\n[SYSTEM PROMPT]\n{SYSTEM_PROMPT}")
        print(f"\n[USER PROMPT]\n{user_msg}")
        print(f"{'='*50}\n⏳ Waiting for LLM response...\n")

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        d = json.loads(resp.choices[0].message.content or "{}")
        s = StrokeOutput(
            reasoning   = d.get("reasoning", ""),
            action_type = d.get("action_type", "line"),
            x1=int(d.get("x1", 10)), y1=int(d.get("y1", 10)),
            x2=int(d["x2"]) if d.get("x2") is not None else None,
            y2=int(d["y2"]) if d.get("y2") is not None else None,
            x3=int(d["x3"]) if d.get("x3") is not None else None,
            y3=int(d["y3"]) if d.get("y3") is not None else None,
            radius=int(d["radius"]) if d.get("radius") is not None else None,
            rx=int(d["rx"]) if d.get("rx") is not None else None,
            ry=int(d["ry"]) if d.get("ry") is not None else None,
        )
        # Defensive clamp — individual fields
        s.x1 = max(0, min(99, s.x1));  s.y1 = max(0, min(99, s.y1))
        for attr in ("x2", "y2", "x3", "y3"):
            v = getattr(s, attr)
            if v is not None:
                setattr(s, attr, max(0, min(99, v)))
        if s.radius is not None: s.radius = max(1, min(100, s.radius))
        if s.rx     is not None: s.rx     = max(1, min(100, s.rx))
        if s.ry     is not None: s.ry     = max(1, min(100, s.ry))
        return s
    except Exception as exc:
        print(f"  [WARN] LLM error: {exc}")
        return StrokeOutput(reasoning="fallback", action_type="line",
                            x1=10, y1=10, x2=90, y2=90)


# ── Local canvas drawing (mirrors server-side _draw_action) ───────────────────

def _apply_stroke(canvas: np.ndarray, action: LearnHandwritingAction) -> np.ndarray:
    """Draw the action onto a fresh temp matrix and merge into canvas via max."""
    temp = np.zeros((100, 100), dtype=np.int32)
    w = BRUSH_WIDTH

    if action.action_type == "circle" and action.radius is not None:
        cv2.circle(temp, (action.x1, action.y1), action.radius, 1, w)

    elif action.action_type == "ellipse" and action.rx and action.ry:
        cv2.ellipse(temp, (action.x1, action.y1),
                    (action.rx, action.ry), 0, 0, 360, 1, w)

    elif (action.action_type == "curve"
          and action.x2 is not None and action.y2 is not None
          and action.x3 is not None and action.y3 is not None):
        cx = 2 * action.x3 - 0.5 * action.x1 - 0.5 * action.x2
        cy = 2 * action.y3 - 0.5 * action.y1 - 0.5 * action.y2
        t = np.linspace(0, 1, 50)
        x = (1 - t) ** 2 * action.x1 + 2 * (1 - t) * t * cx + t ** 2 * action.x2
        y = (1 - t) ** 2 * action.y1 + 2 * (1 - t) * t * cy + t ** 2 * action.y2
        pts = np.stack((x, y), axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(temp, [pts], isClosed=False, color=1, thickness=w)

    elif action.action_type == "line" and action.x2 is not None and action.y2 is not None:
        cv2.line(temp, (action.x1, action.y1), (action.x2, action.y2), 1, w)

    return np.maximum(canvas, temp)


# ── Matplotlib display ────────────────────────────────────────────────────────

def _init_display(task: str, char: str) -> tuple:
    """Create the figure layout. Returns (fig, ax_target, ax_canvas, ax_info)."""
    plt.ion()
    fig = plt.figure(figsize=(11, 6), facecolor="#111")
    fig.suptitle(
        f"Learn Handwriting  ·  Task: {task.upper()}  ·  Character: '{char}'",
        fontsize=14, fontweight="bold", color="white", y=0.97,
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], hspace=0.35, wspace=0.1)
    ax_target = fig.add_subplot(gs[0, 0])
    ax_canvas = fig.add_subplot(gs[0, 1])
    ax_info   = fig.add_subplot(gs[1, :])

    for ax in (ax_target, ax_canvas, ax_info):
        ax.set_facecolor("#111")

    ax_target.set_title("🎯 Target", color="white", fontsize=11)
    ax_canvas.set_title("🖌️  Canvas", color="white", fontsize=11)
    ax_info.axis("off")
    return fig, ax_target, ax_canvas, ax_info


def _update_display(
    fig, ax_target, ax_canvas, ax_info,
    target: np.ndarray,
    canvas: np.ndarray,
    step: int,
    astr: str,
    reasoning: str,
    coverage: float,
    reward: float,
    integrity: bool,
    done: bool,
) -> None:
    ax_target.clear();  ax_canvas.clear();  ax_info.clear()
    ax_target.set_facecolor("#111");  ax_canvas.set_facecolor("#111")
    ax_info.set_facecolor("#111");    ax_info.axis("off")

    # Blend target (dim blue) with canvas (bright white) for the canvas panel
    canvas_rgb = np.stack([canvas * 0.3, canvas * 0.3, canvas.astype(float)], axis=-1)
    target_dim = np.stack([np.zeros_like(target), target * 0.2, np.zeros_like(target)], axis=-1)
    overlay = np.clip(canvas_rgb + target_dim, 0, 1)

    ax_target.imshow(target, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_target.set_title("🎯 Target", color="white", fontsize=11)
    ax_target.axis("off")

    ax_canvas.imshow(overlay, vmin=0, vmax=1, interpolation="nearest")
    cov_color = "#4caf50" if coverage >= 90 else "#ff9800" if coverage >= 50 else "#ef5350"
    ax_canvas.set_title(
        f"🖌️  Canvas  ·  Step {step}/{MAX_STEPS}  ·  {coverage:.1f}%",
        color=cov_color, fontsize=11,
    )
    ax_canvas.axis("off")

    # Progress bar
    bar_len = 30
    filled = int(bar_len * coverage / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    status = "⚠️  INTEGRITY VIOLATED" if integrity else ("✅ Done!" if done else "")
    info = (
        f"Action : {astr}\n"
        f"Reward : {reward:.4f}   Coverage: [{bar}] {coverage:.1f}%   {status}\n"
        f"💭  {reasoning[:100]}"
    )
    ax_info.text(
        0.01, 0.95, info,
        transform=ax_info.transAxes,
        color="white", fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.pause(STROKE_DELAY)


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_watch(task_name: str) -> None:
    """Run one full episode and display every stroke live."""
    # Logic to decide between API and Local loading
    is_local = "/" in (MODEL_NAME or "") or os.path.exists(MODEL_NAME or "")
    
    if is_local:
        client = LocalModelClient(MODEL_NAME, revision=MODEL_REVISION)
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
    env = LearnHandwritingEnv(base_url=ENV_BASE_URL)

    try:
        result = await env.reset(task=task_name)
        obs    = result.observation
        char   = obs.target_character
        target = render_target_character(char).astype(float)
        canvas = np.zeros((100, 100), dtype=np.int32)

        fig, ax_target, ax_canvas, ax_info = _init_display(task_name, char)

        # Show blank canvas at start
        _update_display(
            fig, ax_target, ax_canvas, ax_info,
            target, canvas.astype(float), step=0,
            astr="(starting…)", reasoning="Episode started",
            coverage=0.0, reward=0.0, integrity=False, done=False,
        )

        history: List[str] = []
        last_matched = last_wasted = 0
        ink_remaining = getattr(obs, "ink_remaining", 9999)
        last_reward = 0.0
        last_integrity = False

        print(f"\n{'═'*60}")
        print(f"  Task: {task_name.upper():<8}  Character: '{char}'")
        print(f"{'═'*60}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            bbox = (obs.char_bbox_x1, obs.char_bbox_y1, obs.char_bbox_x2, obs.char_bbox_y2)
            stroke = get_stroke(
                client, step, char, obs.match_percentage,
                last_matched, last_wasted, ink_remaining,
                last_reward, history, last_integrity, bbox,
            )

            action = LearnHandwritingAction(
                action_type=stroke.action_type,
                x1=stroke.x1, y1=stroke.y1,
                x2=stroke.x2, y2=stroke.y2,
                x3=stroke.x3, y3=stroke.y3,
                radius=stroke.radius, rx=stroke.rx, ry=stroke.ry,
            )
            astr = _action_str(stroke)

            # Update canvas locally (same logic as server) — no extra HTTP round-trip
            canvas = _apply_stroke(canvas, action)

            result = await env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0

            last_matched  = obs.pixels_matched_this_stroke
            last_wasted   = getattr(obs, "pixels_wasted_this_stroke", 0)
            ink_remaining = getattr(obs, "ink_remaining", 0)
            last_integrity = getattr(obs, "integrity_violated", False)
            last_reward   = reward
            coverage      = obs.match_percentage * 100

            flag = "  ⚠️  INTEGRITY" if last_integrity else ""
            print(
                f"  Step {step:>2}: {astr:<38}  cov={coverage:>5.1f}%  "
                f"reward={reward:.4f}{flag}"
            )
            print(f"         💭 {stroke.reasoning[:90]}")

            _update_display(
                fig, ax_target, ax_canvas, ax_info,
                target, canvas.astype(float),
                step, astr, stroke.reasoning,
                coverage, reward, last_integrity, result.done,
            )

            history.append(
                f"Step {step}: {astr} → matched={last_matched}px "
                f"coverage={coverage:.1f}% reward={reward:.4f}"
            )

        # Final frame — hold until window is closed
        score   = obs.match_percentage * 100
        success = score >= 90
        print(f"\n  {'✅ SUCCESS' if success else '❌ FAILED'}  —  "
              f"Final coverage: {score:.1f}%  Steps: {len(history)}/{MAX_STEPS}")

        plt.ioff()
        plt.savefig(
            f"watch_{task_name}_{char}.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(),
        )
        print(f"  Saved: watch_{task_name}_{char}.png")
        print("  Close the window to exit.")
        plt.show()

    finally:
        try:
            await env.close()
        except Exception:
            pass
