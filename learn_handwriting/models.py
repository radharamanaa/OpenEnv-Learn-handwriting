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

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

_ALLOWED_ACTIONS = frozenset({"line", "curve", "circle", "ellipse"})


def _coerce_canvas_int(v: object) -> int:
    """Integer grid 0–99 from JSON (ints, floats, numeric strings)."""
    if isinstance(v, bool):
        raise ValueError("boolean is not a valid coordinate")
    if isinstance(v, str):
        v = v.strip()
        if not v:
            raise ValueError("empty coordinate")
        v = float(v)
    x = int(round(float(v)))
    return max(0, min(99, x))


def _coerce_radius_int(v: object) -> int:
    """Radii 1–100 for circle / ellipse."""
    if isinstance(v, bool):
        raise ValueError("boolean is not a valid radius")
    if isinstance(v, str):
        v = v.strip()
        if not v:
            raise ValueError("empty radius")
        v = float(v)
    x = int(round(float(v)))
    return max(1, min(100, x))


class LearnHandwritingAction(Action):
    """A single action (line, curve, circle, ellipse) on the 100x100 canvas."""

    action_type: Literal["line", "curve", "circle", "ellipse"] = Field(default="line", description="Type of action: 'line', 'curve', 'circle', or 'ellipse'")
    x1: int = Field(..., ge=0, le=99, description="For 'line' and 'curve': The starting x coordinate. For 'circle' and 'ellipse': The center x coordinate.")
    y1: int = Field(..., ge=0, le=99, description="For 'line' and 'curve': The starting y coordinate. For 'circle' and 'ellipse': The center y coordinate.")
    
    # Optional end coordinates for lines and curves
    x2: Optional[int] = Field(default=None, ge=0, le=99, description="For 'line' and 'curve': The ending x coordinate. Ignored for 'circle' and 'ellipse'.")
    y2: Optional[int] = Field(default=None, ge=0, le=99, description="For 'line' and 'curve': The ending y coordinate. Ignored for 'circle' and 'ellipse'.")
    
    # Optional pass-through point for curves
    x3: Optional[int] = Field(default=None, ge=0, le=99, description="For 'curve' only: The x coordinate the curve must pass through. Ignored for others.")
    y3: Optional[int] = Field(default=None, ge=0, le=99, description="For 'curve' only: The y coordinate the curve must pass through. Ignored for others.")
    
    # Optional radii for circle and ellipse
    radius: Optional[int] = Field(default=None, ge=1, le=100, description="For 'circle' only: The uniform radius of the circle. Ignored for 'ellipse' and others.")
    rx: Optional[int] = Field(default=None, ge=1, le=100, description="For 'ellipse' only: The horizontal radius. Ignored for 'circle' and others.")
    ry: Optional[int] = Field(default=None, ge=1, le=100, description="For 'ellipse' only: The vertical radius. Ignored for 'circle' and others.")

    @field_validator("action_type", mode="before")
    @classmethod
    def _normalize_action_type(cls, v: object) -> object:
        if v is None:
            return "line"
        if isinstance(v, str):
            key = v.strip().lower()
            if key in _ALLOWED_ACTIONS:
                return key
        return v

    @field_validator("x1", "y1", mode="before")
    @classmethod
    def _validate_xy_required(cls, v: object) -> int:
        return _coerce_canvas_int(v)

    @field_validator("x2", "y2", "x3", "y3", mode="before")
    @classmethod
    def _validate_xy_optional(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return _coerce_canvas_int(v)

    @field_validator("radius", "rx", "ry", mode="before")
    @classmethod
    def _validate_radii(cls, v: object) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return _coerce_radius_int(v)


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
    pixels_wasted_this_stroke: int = Field(
        default=0,
        description="Pixels drawn in the last action that missed the target completely",
    )
    total_drawn_pixels: int = Field(
        default=0,
        description="Total ink drawn on the canvas so far",
    )
    max_allowed_pixels: int = Field(
        default=0,
        description="Maximum ink allowed before episode fails",
    )
    ink_remaining: int = Field(
        default=0,
        description="Remaining pixels agent can draw before failing",
    )
    integrity_violated: bool = Field(
        default=False,
        description=(
            "True if the last stroke violated a shape integrity constraint. "
            "This means the agent filled more than 60% of a protected region "
            "(e.g. the hole in A, the inner loops of B or O, the gap in C, "
            "the bridges in S, or the opening in G). "
            "Episode is done when this is True."
        ),
    )
    char_bbox_x1: int = Field(default=0,  description="Left edge (x) of the target character's bounding box on the 100×100 canvas")
    char_bbox_y1: int = Field(default=0,  description="Top edge (y) of the target character's bounding box on the 100×100 canvas")
    char_bbox_x2: int = Field(default=99, description="Right edge (x) of the target character's bounding box on the 100×100 canvas")
    char_bbox_y2: int = Field(default=99, description="Bottom edge (y) of the target character's bounding box on the 100×100 canvas")



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
