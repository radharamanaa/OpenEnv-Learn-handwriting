# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Learn Handwriting Environment.

This module creates an HTTP server that exposes the LearnHandwritingEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False

try:
    from ..models import LearnHandwritingAction, LearnHandwritingObservation
    from .learn_handwriting_environment import LearnHandwritingEnvironment
    from .renderer import render_target_character_as_image
except ImportError:
    from models import LearnHandwritingAction, LearnHandwritingObservation
    from server.learn_handwriting_environment import LearnHandwritingEnvironment
    from server.renderer import render_target_character_as_image

import numpy as np
from PIL import Image


def _canvas_to_image(canvas: list, scale: int = 2) -> Image.Image:
    """Convert the 100×100 binary canvas matrix to an upscaled PIL Image."""
    arr = np.array(canvas, dtype=np.uint8) * 255
    img = Image.fromarray(arr, mode="L").resize(
        (100 * scale, 100 * scale), resample=Image.NEAREST
    )
    return img.convert("RGB")


def _build_handwriting_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Custom Gradio UI tab for the Learn Handwriting environment."""
    with gr.Blocks() as blocks:
        gr.Markdown("## ✏️ Learn Handwriting — Draw a Character")
        gr.Markdown(
            "Use **Reset** to get a new character, then draw strokes with the controls below. "
            "Goal: cover **90%** of the character's pixels within **15 strokes**."
        )

        # ── Task selector ────────────────────────────────────────────────────
        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task Difficulty",
                scale=1,
            )
            gr.HTML(
                "<p style='margin:auto 0; line-height:2.4em'>"
                "<b>easy</b>: L T V X &nbsp;|&nbsp; "
                "<b>medium</b>: A N Z E &nbsp;|&nbsp; "
                "<b>hard</b>: B C S O G Q</p>"
            )

        # ── State display ────────────────────────────────────────────────────
        with gr.Row():
            target_char   = gr.Textbox(label="🎯 Target Character", interactive=False, scale=1)
            strokes_info  = gr.Textbox(label="✏️ Actions Used", interactive=False, scale=1)
            coverage      = gr.Textbox(label="📊 Coverage (goal: 90%)", interactive=False, scale=1)

        with gr.Row():
            last_matched  = gr.Number(label="Pixels matched (last action)", interactive=False, scale=1)
            ink_remaining = gr.Number(label="💧 Ink Remaining", interactive=False, scale=1)
            last_reward   = gr.Number(label="⭐ Reward (this action)", interactive=False, scale=1, precision=4)
            done_box      = gr.Textbox(label="Done?", interactive=False, scale=1)

        with gr.Row():
            integrity_box = gr.Textbox(label="🚨 Integrity Violated?", interactive=False, scale=2)

        status_msg = gr.Markdown("_Press Reset to start a new episode._")

        # ── Canvas display ───────────────────────────────────────────────────
        blank_img = Image.new("RGB", (200, 200), color=(20, 20, 20))
        with gr.Row():
            target_img = gr.Image(label="🎯 Target Character", interactive=False, width=300, height=300, value=blank_img)
            canvas_img = gr.Image(label="🖌️ Current Canvas", interactive=False, width=300, height=300, value=blank_img)

        # ── Action controls ──────────────────────────────────────────────────
        gr.Markdown("### Draw Action")
        with gr.Row():
            action_type = gr.Dropdown(choices=["line", "curve", "circle", "ellipse"], value="line", label="Action Type")
        with gr.Row():
            x1 = gr.Slider(0, 99, value=50, step=1, label="x1 (Start / Center X)")
            y1 = gr.Slider(0, 99, value=50, step=1, label="y1 (Start / Center Y)")
            x2 = gr.Slider(0, 99, value=50, step=1, label="x2 (End X)")
            y2 = gr.Slider(0, 99, value=50, step=1, label="y2 (End Y)")
        with gr.Row():
            x3 = gr.Slider(0, 99, value=50, step=1, label="x3 (Pass-through X)")
            y3 = gr.Slider(0, 99, value=50, step=1, label="y3 (Pass-through Y)")
            radius = gr.Slider(1, 100, value=20, step=1, label="Radius (circle)")
            rx = gr.Slider(1, 100, value=20, step=1, label="rx (ellipse)")
            ry = gr.Slider(1, 100, value=40, step=1, label="ry (ellipse)")

        with gr.Row():
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
            step_btn  = gr.Button("▶️ Step (draw shape)", variant="primary")

        # ── Helpers ──────────────────────────────────────────────────────────
        def _parse(data: dict):
            obs       = data.get("observation", {})
            char      = obs.get("target_character", "?")
            used      = obs.get("strokes_used", 0)
            pct       = obs.get("match_percentage", 0.0)
            px        = obs.get("pixels_matched_this_stroke", 0)
            ink_rem   = obs.get("ink_remaining", 0)
            rew       = data.get("reward") or 0.0
            done      = data.get("done", False)
            integrity = obs.get("integrity_violated", False)
            return (
                char,
                f"{used} / 15",
                f"{pct:.1%}",
                px,
                ink_rem,
                round(float(rew), 4),
                "✅ Yes" if done else "❌ No",
                "🚨 YES — shape integrity violated, episode ended" if integrity else "✅ No violation",
            )

        async def _get_images(char: str):
            # Target is rendered directly from font — no server call needed.
            tgt_img = render_target_character_as_image(char, scale=2)
            # Canvas requires the server state (accumulated strokes).
            # get_state() is synchronous — do NOT await it.
            try:
                state = web_manager.get_state()
                canvas = state.get("canvas", [[0] * 100 for _ in range(100)])
            except Exception:
                canvas = [[0] * 100 for _ in range(100)]
            return tgt_img, _canvas_to_image(canvas, scale=2)

        async def on_reset(task: str):
            data = await web_manager.reset_environment({"task": task})
            char, strokes, cov, px, ink_rem, rew, done, integrity = _parse(data)
            tgt_img, cvs_img = await _get_images(char)
            return char, strokes, cov, px, ink_rem, rew, done, integrity, f"_Episode started ({task}). Draw your first shape!_", tgt_img, cvs_img

        async def on_step(act_type, x1v, y1v, x2v, y2v, x3v, y3v, rad_v, rx_v, ry_v):
            action = {"action_type": act_type, "x1": int(x1v), "y1": int(y1v)}
            if act_type in ["line", "curve"]:
                action.update({"x2": int(x2v), "y2": int(y2v)})
            if act_type == "curve":
                action.update({"x3": int(x3v), "y3": int(y3v)})
            if act_type == "circle":
                action.update({"radius": int(rad_v)})
            if act_type == "ellipse":
                action.update({"rx": int(rx_v), "ry": int(ry_v)})

            data = await web_manager.step_environment(action)
            char, strokes, cov, px, ink_rem, rew, done, integrity = _parse(data)
            if data.get("observation", {}).get("integrity_violated"):
                msg = "🚨 Integrity violated — episode ended!"
            elif data.get("done"):
                msg = "🎉 Episode complete!"
            else:
                msg = f"_Action sent. Coverage: {cov}_"
            tgt_img, cvs_img = await _get_images(char)
            return char, strokes, cov, px, ink_rem, rew, done, integrity, msg, tgt_img, cvs_img

        outputs = [
            target_char, strokes_info, coverage, last_matched, ink_remaining,
            last_reward, done_box, integrity_box, status_msg, target_img, canvas_img,
        ]

        reset_btn.click(fn=on_reset, inputs=[task_dropdown], outputs=outputs)
        step_btn.click(fn=on_step, inputs=[action_type, x1, y1, x2, y2, x3, y3, radius, rx, ry], outputs=outputs)

    return blocks


app = create_app(
    LearnHandwritingEnvironment,
    LearnHandwritingAction,
    LearnHandwritingObservation,
    env_name="learn_handwriting",
    max_concurrent_envs=1,
    gradio_builder=_build_handwriting_ui if _GRADIO_AVAILABLE else None,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn learn_handwriting.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Learn Handwriting Environment Server")
    parser.add_argument("--host", default=host, help="Host address to bind to")
    parser.add_argument("--port", type=int, default=port, help="Port number to listen on")
    args, _ = parser.parse_known_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
