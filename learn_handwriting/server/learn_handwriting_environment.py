# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Learn Handwriting Environment Implementation.

The agent draws a target character on a 100x100 canvas by issuing strokes.
Each stroke draws a shape on a temporary matrix, computes its intersection
with the target character image for the reward, then merges it into the
cumulative canvas. The episode ends when 90% of the target pixels are
covered, 15 strokes have been used, or an ink / integrity limit is hit.
"""

import os
import random
from uuid import uuid4

import cv2
import numpy as np

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState
except ImportError:
    from models import LearnHandwritingAction, LearnHandwritingObservation, LearnHandwritingState  # noqa: E402

try:
    from .renderer import (
        BRUSH_WIDTH,
        DISQUALIFICATION_MASKS,
        INTEGRITY_THRESHOLD,
        render_target_character,
        compute_character_bbox,
    )
except ImportError:
    from server.renderer import (  # noqa: E402
        BRUSH_WIDTH,
        DISQUALIFICATION_MASKS,
        INTEGRITY_THRESHOLD,
        render_target_character,
        compute_character_bbox,
    )

MAX_STROKES = 15
MATCH_THRESHOLD = 0.90

# Task difficulty pools — ordered by geometric complexity.
#
# easy   → pure straight-line characters; a perfect agent needs ≤ 2 strokes.
# medium → straight lines but 3+ strokes, or mild complexity; no curves.
# hard   → curves, open gaps, enclosed counters, or complex topology.
#           Many of these letters use shape-integrity constraints (see renderer.py).
TASK_CHARACTERS: dict[str, list[str]] = {
    "easy":   ["L", "T", "V", "X"],
    "medium": ["A", "N", "Z", "E", "F", "H", "I", "K", "M", "W", "Y"],
    "hard":   ["B", "C", "D", "G", "J", "O", "P", "Q", "R", "S", "U"],
}


def _draw_action(action: LearnHandwritingAction) -> np.ndarray:
    """Return a 100×100 binary matrix with the drawn shape."""
    canvas = np.zeros((100, 100), dtype=np.int32)
    w = BRUSH_WIDTH

    if action.action_type == "circle" and action.radius is not None:
        cv2.circle(canvas, (int(action.x1), int(action.y1)), int(action.radius), 1, w)
    elif action.action_type == "ellipse" and action.rx is not None and action.ry is not None:
        cv2.ellipse(
            canvas,
            (int(action.x1), int(action.y1)),
            (int(action.rx), int(action.ry)),
            0, 0, 360, 1, w,
        )
    elif (
        action.action_type == "curve"
        and action.x2 is not None and action.y2 is not None
        and action.x3 is not None and action.y3 is not None
    ):
        cx = 2 * action.x3 - 0.5 * action.x1 - 0.5 * action.x2
        cy = 2 * action.y3 - 0.5 * action.y1 - 0.5 * action.y2
        t = np.linspace(0, 1, 50)
        x = (1 - t) ** 2 * action.x1 + 2 * (1 - t) * t * cx + t ** 2 * action.x2
        y = (1 - t) ** 2 * action.y1 + 2 * (1 - t) * t * cy + t ** 2 * action.y2
        pts = np.stack((x, y), axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=1, thickness=w)
    elif action.action_type == "line" and action.x2 is not None and action.y2 is not None:
        cv2.line(
            canvas,
            (int(action.x1), int(action.y1)),
            (int(action.x2), int(action.y2)),
            1, w,
        )

    return canvas


class LearnHandwritingEnvironment(Environment):
    """
    Handwriting RL environment.

    The agent receives a target character to draw and issues up to 15 strokes
    on a 100×100 canvas. Each stroke is evaluated against the font-rendered
    target image. The episode succeeds when 90% of the target's pixels are
    covered; it fails on stroke exhaustion, ink overflow, or integrity violation.

    Example:
        >>> env = LearnHandwritingEnvironment()
        >>> obs = env.reset()
        >>> print(obs.target_character)   # e.g. "A"
        >>> obs = env.step(LearnHandwritingAction(action_type="line", x1=10, y1=10, x2=50, y2=90))
        >>> print(obs.reward, obs.match_percentage)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        """
        Args:
            task: Difficulty level — "easy", "medium", or "hard".
                  Controls which characters can be selected on reset().
        """
        self._char_pool: list[str] = TASK_CHARACTERS.get(task, TASK_CHARACTERS["easy"])
        if not self._char_pool:
            raise ValueError(
                f"No characters found for task={task!r}. "
                f"Expected one of: {list(TASK_CHARACTERS.keys())}"
            )
        self._task = task
        self._target_matrix: np.ndarray = np.zeros((100, 100), dtype=np.int32)
        self._total_target_pixels: int = 0
        self._target_bbox: tuple[int, int, int, int] = (0, 0, 99, 99)
        self._canvas: np.ndarray = np.zeros((100, 100), dtype=np.int32)
        self._state = LearnHandwritingState(episode_id=str(uuid4()), step_count=0)

    def reset(self, task: str | None = None, character: str | None = None) -> LearnHandwritingObservation:
        """Pick a random character, reset the canvas, and return initial observation.

        Args:
            task: Optional difficulty level — "easy", "medium", or "hard".
                  If provided, switches the character pool for this and future episodes.
            character: If set, use this exact target letter instead of sampling from the pool
                  (used by datagen so strokes stay aligned after ``task`` switches pools).
        """
        if task is not None and task != self._task:
            new_pool = TASK_CHARACTERS.get(task, TASK_CHARACTERS["easy"])
            if new_pool:
                self._char_pool = new_pool
                self._task = task

        if character is not None:
            char = character
        else:
            char = random.choice(self._char_pool)
        self._target_matrix = render_target_character(char).astype(np.int32)
        self._total_target_pixels = int(np.sum(self._target_matrix))
        self._target_bbox = compute_character_bbox(char)
        self._canvas = np.zeros((100, 100), dtype=np.int32)

        max_drawn_multiplier = float(os.getenv("MAX_DRAWN_MULTIPLIER", "1.7"))
        max_allowed_pixels = int(max_drawn_multiplier * self._total_target_pixels)

        self._state = LearnHandwritingState(
            episode_id=str(uuid4()),
            step_count=0,
            target_character=char,
            strokes_used=0,
            canvas=self._canvas.tolist(),
        )

        return LearnHandwritingObservation(
            target_character=char,
            strokes_used=0,
            pixels_matched_this_stroke=0,
            total_matched_pixels=0,
            match_percentage=0.0,
            pixels_wasted_this_stroke=0,
            total_drawn_pixels=0,
            max_allowed_pixels=max_allowed_pixels,
            ink_remaining=max_allowed_pixels,
            integrity_violated=False,
            char_bbox_x1=self._target_bbox[0],
            char_bbox_y1=self._target_bbox[1],
            char_bbox_x2=self._target_bbox[2],
            char_bbox_y2=self._target_bbox[3],
            done=False,
            reward=0.0,
        )

    def _check_integrity_violation(self) -> bool:
        """
        Return True if the canvas covers > INTEGRITY_THRESHOLD of any
        disqualification zone for the current character.
        Called after every stroke merge.
        """
        masks = DISQUALIFICATION_MASKS.get(self._state.target_character, [])
        for mask in masks:
            zone_pixels = int(np.sum(mask))
            if zone_pixels == 0:
                continue
            covered = int(np.sum(self._canvas * mask))
            if covered / zone_pixels > INTEGRITY_THRESHOLD:
                return True
        return False

    def step(self, action: LearnHandwritingAction) -> LearnHandwritingObservation:  # type: ignore[override]
        """
        Execute one stroke action.

        Workflow:
          1. Draw stroke on a fresh temp_matrix.
          2. Record pre-merge cumulative match; merge stroke into canvas (element-wise max).
          3. Check shape integrity — episode ends if a protected zone is filled.
          4. Reward from **new** target coverage only (marginal pixels); retracing gives 0.
             Wasted ink = stroke pixels that never intersect the target.
          5. total_matched_pixels → match_percentage; increment strokes_used; done checks.
        """
        self._state.step_count += 1

        temp_matrix = _draw_action(action)

        # Stroke ∩ target (all pixels this stroke paints on the letter shape)
        stroke_on_target = int(np.sum(temp_matrix * self._target_matrix))
        prev_total_matched = int(np.sum(self._canvas * self._target_matrix))

        # Merge stroke into cumulative canvas
        self._canvas = np.maximum(self._canvas, temp_matrix)

        # Integrity check — must happen after merge so the canvas reflects this stroke
        integrity_violated = self._check_integrity_violation()

        # Cumulative match stats
        total_matched_pixels = int(np.sum(self._canvas * self._target_matrix))
        # Reward only **new** coverage: retracing already-filled target pixels gives 0 progress
        pixels_matched_this_stroke = total_matched_pixels - prev_total_matched
        reward = (
            pixels_matched_this_stroke / self._total_target_pixels
            if self._total_target_pixels > 0
            else 0.0
        )
        if integrity_violated:
            reward = 0.0  # no reward for violating shape integrity (grader requires ≥ 0)
        match_percentage = (
            total_matched_pixels / self._total_target_pixels
            if self._total_target_pixels > 0
            else 0.0
        )

        strokes_used = self._state.strokes_used + 1

        # Ink tracking — wasted = stroke pixels that never intersect the target (not "redundant" retracing)
        pixels_drawn_this_stroke = int(np.sum(temp_matrix))
        pixels_wasted_this_stroke = max(0, pixels_drawn_this_stroke - stroke_on_target)
        total_drawn_pixels = int(np.sum(self._canvas))
        max_drawn_multiplier = float(os.getenv("MAX_DRAWN_MULTIPLIER", "1.7"))
        max_allowed_pixels = int(max_drawn_multiplier * self._total_target_pixels)
        ink_remaining = max(0, max_allowed_pixels - total_drawn_pixels)

        done = (
            integrity_violated
            or match_percentage >= MATCH_THRESHOLD
            or strokes_used >= MAX_STROKES
            or total_drawn_pixels > max_allowed_pixels
        )

        self._state.strokes_used = strokes_used
        self._state.canvas = self._canvas.tolist()

        return LearnHandwritingObservation(
            target_character=self._state.target_character,
            strokes_used=strokes_used,
            pixels_matched_this_stroke=pixels_matched_this_stroke,
            total_matched_pixels=total_matched_pixels,
            match_percentage=match_percentage,
            pixels_wasted_this_stroke=pixels_wasted_this_stroke,
            total_drawn_pixels=total_drawn_pixels,
            max_allowed_pixels=max_allowed_pixels,
            ink_remaining=ink_remaining,
            integrity_violated=integrity_violated,
            char_bbox_x1=self._target_bbox[0],
            char_bbox_y1=self._target_bbox[1],
            char_bbox_x2=self._target_bbox[2],
            char_bbox_y2=self._target_bbox[3],
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> LearnHandwritingState:
        """Return the current episode state including the accumulated canvas."""
        return self._state
