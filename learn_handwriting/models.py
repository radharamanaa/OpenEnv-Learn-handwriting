# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Learn Handwriting Environment.

The agent draws a target character on a 100x100 canvas by issuing strokes.
Each stroke is a line from (x1, y1) to (x2, y2). The environment tracks
accumulated strokes in a canvas matrix and compares them against the target
character image to compute reward and match progress.
"""

from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class LearnHandwritingAction(Action):
    """A single stroke from (x1, y1) to (x2, y2) on the 100x100 canvas."""

    x1: int = Field(..., ge=0, le=99, description="Stroke start x coordinate (0–99)")
    y1: int = Field(..., ge=0, le=99, description="Stroke start y coordinate (0–99)")
    x2: int = Field(..., ge=0, le=99, description="Stroke end x coordinate (0–99)")
    y2: int = Field(..., ge=0, le=99, description="Stroke end y coordinate (0–99)")
    width: int = Field(default=3, ge=1, le=10, description="Stroke brush width in pixels (1–10)")


class LearnHandwritingObservation(Observation):
    """Observation returned to the agent after each stroke.

    The canvas itself is NOT included — it lives in LearnHandwritingState.
    The agent receives summary statistics only.
    """

    target_character: str = Field(default="", description="The character to draw (e.g. 'A')")
    strokes_used: int = Field(default=0, description="Number of strokes used so far")
    pixels_matched_this_stroke: int = Field(
        default=0,
        description="Pixels in this stroke that intersect the target (reward numerator)",
    )
    total_matched_pixels: int = Field(
        default=0,
        description="Cumulative pixels in canvas that intersect the target across all strokes",
    )
    match_percentage: float = Field(
        default=0.0,
        description="total_matched_pixels / total_target_pixels — progress toward 90% goal",
    )


class LearnHandwritingState(State):
    """Full episode state, including the accumulated 100x100 stroke canvas.

    Returned by the /state endpoint. The canvas is kept here and NOT sent
    to the agent in observations.
    """

    target_character: str = Field(default="", description="Character being drawn this episode")
    strokes_used: int = Field(default=0, description="Number of strokes used so far")
    canvas: List[List[int]] = Field(
        default_factory=lambda: [[0] * 100 for _ in range(100)],
        description="100x100 matrix of accumulated strokes (0 = empty, 1 = stroke pixel)",
    )
