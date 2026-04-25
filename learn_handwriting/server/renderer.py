"""
Shared rendering module for the Learn Handwriting environment.

Single source of truth for:
- Character rendering (font-based, replaces static JPG pipeline)
- Disqualification mask computation

Both the environment and the web UI import exclusively from here.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).parent.parent / "Roboto" / "static" / "Roboto-Bold.ttf"
FONT_SIZE = 80
BRUSH_WIDTH = 8

# Fraction of a mask's pixels the agent may cover before the episode is ended.
INTEGRITY_THRESHOLD = 0.60


def render_target_character(character: str) -> np.ndarray:
    """
    Render a single capital letter onto a 100×100 binary numpy array.
    Returns uint8 array: foreground pixels = 1, background = 0.
    """
    img = Image.new("L", (100, 100), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)

    bbox = draw.textbbox((0, 0), character, font=font)
    x_off = (100 - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y_off = (100 - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x_off, y_off), character, font=font, fill=255)

    arr = np.array(img, dtype=np.uint8)
    return (arr >= 128).astype(np.uint8)


def render_target_character_as_image(character: str, scale: int = 2) -> Image.Image:
    """
    Render the target character and upscale for display.
    Returns an RGB PIL Image of size (100*scale, 100*scale).
    """
    arr = render_target_character(character)
    img = Image.fromarray(arr * 255, mode="L")
    return img.resize((100 * scale, 100 * scale), resample=Image.NEAREST).convert("RGB")


def compute_character_bbox(character: str) -> tuple[int, int, int, int]:
    """
    Return the tight bounding box of the rendered character's foreground pixels.

    Returns:
        (x_min, y_min, x_max, y_max) — all in canvas coordinates [0, 99].
        x_min/x_max are column indices (left/right edges).
        y_min/y_max are row indices (top/bottom edges).
    """
    arr = render_target_character(character)
    rows = np.any(arr, axis=1)  # rows that have at least one foreground pixel
    cols = np.any(arr, axis=0)  # cols that have at least one foreground pixel
    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return x_min, y_min, x_max, y_max



def flood_fill_interior(char_matrix: np.ndarray) -> np.ndarray:
    """
    Given a binary 100×100 matrix (1=foreground, 0=background), return a matrix
    where enclosed background regions are also filled to 1.

    Steps:
      1. Invert so background=1, foreground=0.
      2. Flood-fill from top-left corner, marking reachable background as 2.
      3. Any remaining 1s are enclosed interiors unreachable from the border.
      4. OR them back into the original to produce the filled version.
    """
    inverted = (1 - char_matrix).astype(np.uint8)
    h, w = inverted.shape
    flood = inverted.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 2)
    interior = (flood == 1).astype(np.uint8)
    return (char_matrix | interior).astype(np.uint8)


# ---------------------------------------------------------------------------
# Disqualification masks — built once at module load
# ---------------------------------------------------------------------------

DISQUALIFICATION_MASKS: dict[str, list[np.ndarray]] = {}


def _build_disqualification_masks() -> None:
    """
    Compute all disqualification masks once at module load.
    Each entry maps a character to a list of binary 100×100 masks.
    An episode ends immediately if the agent's canvas covers > INTEGRITY_THRESHOLD
    of any mask's pixels.
    """
    # Diff-based masks: pixels present in the "bad" character but absent in the target.
    # These are the regions the agent must NOT draw into.
    diff_pairs: dict[str, str] = {
        "C": "O",  # right-side closing arc that would turn C into O
        "S": "8",  # two bridge regions between S loops that would form 8
        "G": "O",  # right-side closing arc that would turn G into O
    }
    for good_char, bad_char in diff_pairs.items():
        good = render_target_character(good_char)
        bad = render_target_character(bad_char)
        mask = np.clip(bad.astype(np.int16) - good.astype(np.int16), 0, 1).astype(np.uint8)
        if mask.sum() > 0:
            DISQUALIFICATION_MASKS[good_char] = [mask]

    # Flood-fill interior masks: enclosed counter regions the agent must not fill.
    for char in ["A", "B", "O", "Q"]:
        target = render_target_character(char)
        filled = flood_fill_interior(target)
        interior = np.clip(filled.astype(np.int16) - target.astype(np.int16), 0, 1).astype(np.uint8)

        if char == "B":
            # B has two separate lobe interiors — treat each as an independent constraint.
            n_labels, labels = cv2.connectedComponents(interior)
            masks = []
            for label_id in range(1, n_labels):
                component = (labels == label_id).astype(np.uint8)
                if component.sum() > 50:  # discard noise
                    masks.append(component)
            if masks:
                DISQUALIFICATION_MASKS[char] = masks
        else:
            if interior.sum() > 0:
                DISQUALIFICATION_MASKS[char] = [interior]


_build_disqualification_masks()
