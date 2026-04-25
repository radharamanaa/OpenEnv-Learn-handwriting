#!/usr/bin/env python3
"""
Export every glyph in TASK_CHARACTERS to characters/ for visual reference.

Uses the same binary raster as the environment (Roboto Bold via server.renderer).
Run from the project root:

    python export_character_images.py

Output: characters/<LETTER>.png (100×100 upscaled with nearest-neighbor for readability).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from server.learn_handwriting_environment import TASK_CHARACTERS
from server.renderer import render_target_character

OUT_DIR = Path(__file__).resolve().parent / "characters"
SCALE = 8  # 100×100 → 800×800


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chars = sorted({c for pool in TASK_CHARACTERS.values() for c in pool})
    for char in chars:
        arr = render_target_character(char)
        assert arr.shape == (100, 100)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[arr > 0] = (255, 255, 255)
        img = Image.fromarray(rgb, mode="RGB")
        out = img.resize((100 * SCALE, 100 * SCALE), resample=Image.NEAREST)
        path = OUT_DIR / f"{char}.png"
        out.save(path)
        px = int(arr.sum())
        print(f"  {char}  {px:4d} px  →  {path.relative_to(OUT_DIR.parent)}")
    print(f"\nWrote {len(chars)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
