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
    from .learn_handwriting_environment import LearnHandwritingEnvironment, CHARACTERS_DIR
except ImportError:
    from models import LearnHandwritingAction, LearnHandwritingObservation
    from server.learn_handwriting_environment import LearnHandwritingEnvironment, CHARACTERS_DIR

import numpy as np
from PIL import Image


def _canvas_to_image(canvas: list, scale: int = 4) -> Image.Image:
    """Convert the 100x100 binary canvas matrix to an upscaled PIL Image."""
    arr = np.array(canvas, dtype=np.uint8) * 255  # 0 or 255
    img = Image.fromarray(arr, mode="L").resize(
        (100 * scale, 100 * scale), resample=Image.NEAREST
    )
    return img.convert("RGB")


def _target_to_image(character: str, scale: int = 4) -> Image.Image:
    """Load the target character reference image and upscale it."""
    path = CHARACTERS_DIR / f"{character}.jpg"
    if not path.exists():
        # Return a blank image if character file not found
        return Image.new("RGB", (100 * scale, 100 * scale), color=(200, 200, 200))
    img = Image.open(path).convert("L").resize(
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

        # ── State display ────────────────────────────────────────────────────
        with gr.Row():
            target_char   = gr.Textbox(label="🎯 Target Character", interactive=False, scale=1)
            strokes_info  = gr.Textbox(label="✏️ Strokes Used", interactive=False, scale=1)
            coverage      = gr.Textbox(label="📊 Coverage (goal: 90%)", interactive=False, scale=1)

        with gr.Row():
            last_matched  = gr.Number(label="Pixels matched (this stroke)", interactive=False, scale=1)
            last_reward   = gr.Number(label="⭐ Reward (this stroke)", interactive=False, scale=2, precision=4)
            done_box      = gr.Textbox(label="Done?", interactive=False, scale=1)

        status_msg = gr.Markdown("_Press Reset to start a new episode._")

        # ── Canvas display ───────────────────────────────────────────────────
        with gr.Row():
            target_img = gr.Image(label="🎯 Target Character", interactive=False, width=300, height=300)
            canvas_img = gr.Image(label="🖌️ Current Canvas", interactive=False, width=300, height=300)

        # ── Action controls ──────────────────────────────────────────────────
        gr.Markdown("### Stroke Action")
        with gr.Row():
            x1 = gr.Slider(0, 99, value=10, step=1, label="x1 (start col)")
            y1 = gr.Slider(0, 99, value=10, step=1, label="y1 (start row)")
            x2 = gr.Slider(0, 99, value=90, step=1, label="x2 (end col)")
            y2 = gr.Slider(0, 99, value=90, step=1, label="y2 (end row)")
            width = gr.Slider(1, 10, value=5, step=1, label="Brush Width (px)")

        with gr.Row():
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
            step_btn  = gr.Button("▶️ Step (draw stroke)", variant="primary")

        # ── Helpers ──────────────────────────────────────────────────────────
        def _parse(data: dict):
            obs   = data.get("observation", {})
            char  = obs.get("target_character", "?")
            used  = obs.get("strokes_used", 0)
            pct   = obs.get("match_percentage", 0.0)
            px    = obs.get("pixels_matched_this_stroke", 0)
            rew   = data.get("reward") or 0.0
            done  = data.get("done", False)
            return (
                char,
                f"{used} / 15",
                f"{pct:.1%}",
                px,
                round(float(rew), 4),
                "✅ Yes" if done else "❌ No",
            )

        def _get_images(char: str):
            state = web_manager.get_state()
            canvas = state.get("canvas", [[0] * 100 for _ in range(100)])
            return _target_to_image(char), _canvas_to_image(canvas)

        async def on_reset():
            data = await web_manager.reset_environment()
            char, strokes, cov, px, rew, done = _parse(data)
            tgt_img, cvs_img = _get_images(char)
            return char, strokes, cov, px, rew, done, "_Episode started. Draw your first stroke!_", tgt_img, cvs_img

        async def on_step(x1v, y1v, x2v, y2v, wv):
            action = {"x1": int(x1v), "y1": int(y1v), "x2": int(x2v), "y2": int(y2v), "width": int(wv)}
            data = await web_manager.step_environment(action)
            char, strokes, cov, px, rew, done = _parse(data)
            msg = "🎉 Episode complete!" if data.get("done") else f"_Stroke sent. Coverage: {cov}_"
            tgt_img, cvs_img = _get_images(char)
            return char, strokes, cov, px, rew, done, msg, tgt_img, cvs_img

        outputs = [target_char, strokes_info, coverage, last_matched, last_reward, done_box, status_msg, target_img, canvas_img]

        reset_btn.click(fn=on_reset, inputs=[], outputs=outputs)
        step_btn.click(fn=on_step, inputs=[x1, y1, x2, y2, width], outputs=outputs)

    return blocks


# Create the app with web interface and README integration
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

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m learn_handwriting.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn learn_handwriting.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
