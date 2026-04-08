# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Learn Handwriting Environment Implementation.

The agent draws a target character on a 100x100 canvas by issuing strokes.
Each stroke draws a Bresenham line on a temporary matrix, computes its
intersection with the target character image for the reward, then merges
it into the cumulative canvas. The episode ends when 90% of the target
pixels are covered or 15 strokes have been used.
"""

import random
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState
except ImportError:
    from models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState  # noqa: E402

CHARACTERS_DIR = Path(__file__).parent.parent / "characters"
MAX_STROKES = 15
MATCH_THRESHOLD = 0.90

# Task difficulty pools — ordered by geometric complexity for straight-line stroke agents.
#
# Difficulty rationale:
#   easy   → L only.  L is two perpendicular straight lines; a perfect agent
#             needs exactly 2 strokes.  Unambiguously the simplest character.
#   medium → V, Z, A.  All composed of straight diagonal/horizontal lines that
#             a line-drawing agent can hit efficiently.  No curves.
#   hard   → C, B, O.  All involve arcs or bumps.  Straight strokes can only
#             approximate curves, so coverage per stroke is inherently lower.
#             Note: C was incorrectly placed in "easy" — it is geometrically
#             harder than A, V, or Z because it is a curved arc, not a polyline.
TASK_CHARACTERS: dict[str, list[str]] = {
    "easy":   ["L", "V"],
    "medium": ["B", "A"],
    "hard":   ["C", "S"],
}


def _draw_stroke(x1: int, y1: int, x2: int, y2: int, width: int) -> np.ndarray:
    """Return a 100x100 binary matrix with a thick line drawn using PIL."""
    img = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(img)
    draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
    arr = np.array(img, dtype=np.int32)
    return (arr > 0).astype(np.int32)


class LearnHandwritingEnvironment(Environment):
    """
    Handwriting RL environment.

    The agent receives a target character to draw and issues up to 15 strokes
    on a 100x100 canvas. Each stroke is evaluated against the target image.
    The episode succeeds when 90% of the target's white pixels are covered.

    Example:
        >>> env = LearnHandwritingEnvironment()
        >>> obs = env.reset()
        >>> print(obs.target_character)   # e.g. "A"
        >>> obs = env.step(LearnHandwritingAction(x1=10, y1=10, x2=50, y2=90))
        >>> print(obs.reward, obs.match_percentage)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        """
        Args:
            task: Difficulty level — "easy", "medium", or "hard".
                  Controls which characters can be selected on reset().
        """
        allowed = TASK_CHARACTERS.get(task, TASK_CHARACTERS["easy"])
        all_paths = sorted(CHARACTERS_DIR.glob("*.jpg"))
        self._char_paths: list[Path] = [p for p in all_paths if p.stem in allowed]
        if not self._char_paths:
            raise ValueError(f"No character images found for task={task!r}. "
                             f"Expected one of: {list(TASK_CHARACTERS.keys())}")
        self._task = task
        self._target_matrix: np.ndarray = np.zeros((100, 100), dtype=np.int32)
        self._total_target_pixels: int = 0
        self._canvas: np.ndarray = np.zeros((100, 100), dtype=np.int32)
        self._state = LearnHandwritingState(episode_id=str(uuid4()), step_count=0)

    def _load_target(self, path: Path) -> np.ndarray:
        """Load character image as a binary 100x100 matrix (pixel >= 240 → 1)."""
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.int32)
        return (arr >= 240).astype(np.int32)

    def reset(self, task: str | None = None) -> LearnHandwritingObservation:
        """Pick a random character, reset the canvas, and return initial observation.

        Args:
            task: Optional difficulty level — "easy", "medium", or "hard".
                  If provided, switches the character pool for this and future episodes.
        """
        if task is not None and task != self._task:
            allowed = TASK_CHARACTERS.get(task, TASK_CHARACTERS["easy"])
            all_paths = sorted(CHARACTERS_DIR.glob("*.jpg"))
            new_paths = [p for p in all_paths if p.stem in allowed]
            if new_paths:
                self._char_paths = new_paths
                self._task = task

        char_path = random.choice(self._char_paths)
        char_name = char_path.stem  # e.g. "A"

        self._target_matrix = self._load_target(char_path)
        self._total_target_pixels = int(np.sum(self._target_matrix))
        self._canvas = np.zeros((100, 100), dtype=np.int32)

        self._state = LearnHandwritingState(
            episode_id=str(uuid4()),
            step_count=0,
            target_character=char_name,
            strokes_used=0,
            canvas=self._canvas.tolist(),
        )

        return LearnHandwritingObservation(
            target_character=char_name,
            strokes_used=0,
            pixels_matched_this_stroke=0,
            total_matched_pixels=0,
            match_percentage=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: LearnHandwritingAction) -> LearnHandwritingObservation:  # type: ignore[override]
        """
        Execute one stroke action.

        Workflow:
          1. Draw stroke on a fresh temp_matrix using Bresenham.
          2. Intersect temp_matrix with target → pixels_matched_this_stroke → reward.
          3. Merge temp_matrix into canvas via element-wise max.
          4. Intersect updated canvas with target → total_matched_pixels → match_percentage.
          5. Increment strokes_used; check done conditions.
        """
        self._state.step_count += 1

        # Step 1 & 2: draw thick stroke onto temp matrix via PIL
        temp_matrix = _draw_stroke(action.x1, action.y1, action.x2, action.y2, action.width)

        # Step 3: reward — intersection of this stroke with target
        pixels_matched_this_stroke = int(np.sum(temp_matrix * self._target_matrix))
        reward = (
            pixels_matched_this_stroke / self._total_target_pixels
            if self._total_target_pixels > 0
            else 0.0
        )

        # Step 4: merge stroke into cumulative canvas
        self._canvas = np.maximum(self._canvas, temp_matrix)

        # Step 5: cumulative match stats
        total_matched_pixels = int(np.sum(self._canvas * self._target_matrix))
        match_percentage = (
            total_matched_pixels / self._total_target_pixels
            if self._total_target_pixels > 0
            else 0.0
        )

        # Step 6: update stroke count
        strokes_used = self._state.strokes_used + 1

        # Step 7: done conditions
        done = match_percentage >= MATCH_THRESHOLD or strokes_used >= MAX_STROKES

        # Persist into state
        self._state.strokes_used = strokes_used
        self._state.canvas = self._canvas.tolist()

        return LearnHandwritingObservation(
            target_character=self._state.target_character,
            strokes_used=strokes_used,
            pixels_matched_this_stroke=pixels_matched_this_stroke,
            total_matched_pixels=total_matched_pixels,
            match_percentage=match_percentage,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> LearnHandwritingState:
        """Return the current episode state including the accumulated canvas."""
        return self._state
