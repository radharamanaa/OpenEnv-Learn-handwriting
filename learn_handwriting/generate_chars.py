"""
Regenerate character images in learn_handwriting/characters/.
Each letter is rendered at large size with a bold font on a 100x100
black canvas, then binarised: pixels > 50 → 240, else 0.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARACTERS = ["A", "B", "C", "L", "O", "V", "Z"]
OUT_DIR = Path("learn_handwriting/characters")
FONT_PATH = "/System/Library/Fonts/Supplemental/Impact.ttf"
SIZE = 100

def generate(char: str):
    # Start large to measure, then fit tightly
    font_size = 90
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Measure bounding box
    tmp = Image.new("L", (SIZE * 4, SIZE * 4), 0)
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Scale font to fill ~85% of canvas
    scale = (SIZE * 0.85) / max(w, h)
    font_size = int(font_size * scale)
    font = ImageFont.truetype(FONT_PATH, font_size)

    # Re-measure with scaled font
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Centre on canvas
    x = (SIZE - w) // 2 - bbox[0]
    y = (SIZE - h) // 2 - bbox[1]

    img = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), char, fill=255, font=font)

    # Binarise: > 50 → 255, else 0
    arr = np.array(img)
    arr = np.where(arr > 50, 255, 0).astype(np.uint8)
    img = Image.fromarray(arr)

    # Erode ~30% of stroke width (2 passes of MinFilter(3) removes ~2px per edge)
    from PIL import ImageFilter
    img = img.filter(ImageFilter.MinFilter(3))
    img = img.filter(ImageFilter.MinFilter(3))

    # Final binarise: > 50 → 240, else 0
    arr = np.array(img)
    arr = np.where(arr > 50, 240, 0).astype(np.uint8)

    out_path = OUT_DIR / f"{char}.jpg"
    Image.fromarray(arr).save(out_path)

    white_pixels = int(np.sum(arr == 240))
    print(f"{char}.jpg  →  {white_pixels} white pixels  (saved to {out_path})")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ch in CHARACTERS:
        generate(ch)
    print("\nDone.")
