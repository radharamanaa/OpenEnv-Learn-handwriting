"""
Tests for server/renderer.py.

Run from the project root:
    pytest tests/test_renderer.py -v
"""

import numpy as np
import pytest

from server.renderer import (
    DISQUALIFICATION_MASKS,
    render_target_character,
)

ALL_CHARS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]
INTEGRITY_CHARS = ["A", "B", "C", "D", "G", "O", "P", "Q", "R", "S"]

# Expected foreground pixel count ranges per character
PIXEL_RANGES: dict[str, tuple[int, int]] = {
    "A": (1200, 2500),
    "B": (1500, 3000),
    "C": (800, 1800),
    "D": (1350, 1750),
    "E": (1200, 2500),
    "F": (950, 1300),
    "G": (1000, 2200),
    "H": (1350, 1750),
    "I": (550, 850),
    "J": (800, 1100),
    "K": (1350, 1750),
    "L": (800, 1800),
    "M": (2000, 2500),
    "N": (1200, 2500),
    "O": (1200, 2500),
    "P": (1200, 1550),
    "Q": (1200, 2800),
    "R": (1450, 1800),
    "S": (1200, 2500),
    "T": (800, 1800),
    "U": (1200, 1650),
    "V": (800, 1800),
    "W": (1900, 2400),
    "X": (900, 2000),
    "Y": (950, 1250),
    "Z": (1000, 2200),
}

# Minimum meaningful size for each disqualification mask.
# Calibrated against Roboto-Bold FONT_SIZE=80 on 100×100 canvas.
MASK_MINIMUMS: dict[str, int] = {
    "A": 150,   # triangle interior (~161px actual)
    "B": 150,   # each lobe — checked per-component (~236 / 276px actual)
    "C": 100,   # right-arc diff vs O (~364px actual)
    "D": 400,   # D-bowl interior (~682px actual)
    "S": 80,    # each bridge diff vs 8 (~279px actual — single combined mask)
    "O": 750,   # circle interior (~825px actual; bold strokes reduce the hole)
    "G": 100,   # right-arc diff vs O (~184px actual)
    "P": 200,   # bowl interior (~342px actual)
    "Q": 600,   # circle interior (~825px actual; same as O)
    "R": 180,   # bowl interior (~303px actual)
}


def test_render_shape():
    for char in ALL_CHARS:
        arr = render_target_character(char)
        assert arr.shape == (100, 100), f"{char}: wrong shape"
        assert arr.dtype == np.uint8, f"{char}: wrong dtype"
        assert set(np.unique(arr)).issubset({0, 1}), f"{char}: not binary"


def test_foreground_pixel_range():
    for char, (lo, hi) in PIXEL_RANGES.items():
        count = int(np.sum(render_target_character(char)))
        assert lo <= count <= hi, (
            f"{char}: {count}px is outside expected range [{lo}, {hi}]. "
            "Adjust FONT_SIZE or BRUSH_WIDTH constants."
        )


def test_integrity_zones_present():
    for char in INTEGRITY_CHARS:
        masks = DISQUALIFICATION_MASKS.get(char, [])
        assert len(masks) > 0, f"{char}: no disqualification masks built"
        min_px = MASK_MINIMUMS.get(char, 80)
        for i, mask in enumerate(masks):
            px = int(np.sum(mask))
            assert px >= min_px, (
                f"{char} mask[{i}]: only {px}px, minimum is {min_px}. "
                "Consider decreasing FONT_SIZE to open up interior gaps."
            )


def test_disqualifier_does_not_overlap_foreground():
    """Disqualification pixels must not overlap the character's own foreground strokes."""
    for char in INTEGRITY_CHARS:
        target = render_target_character(char)
        for i, mask in enumerate(DISQUALIFICATION_MASKS.get(char, [])):
            overlap = int(np.sum(target * mask))
            assert overlap == 0, (
                f"{char} mask[{i}]: disqualifier overlaps {overlap} foreground pixels"
            )


def test_renderer_deterministic():
    for char in ALL_CHARS:
        a = render_target_character(char)
        b = render_target_character(char)
        assert np.array_equal(a, b), f"{char}: renderer is non-deterministic"


def test_no_integrity_masks_for_simple_chars():
    """Easy/medium chars with no holes should have no disqualification masks."""
    simple = ["L", "T", "V", "X", "N", "Z", "E", "F", "H", "I", "K", "M", "W", "Y", "J", "U"]
    for char in simple:
        masks = DISQUALIFICATION_MASKS.get(char, [])
        assert masks == [], f"{char}: unexpected disqualification mask"
