"""
Renders all 13 task characters and their disqualification masks to images/
for visual inspection.

Run from the project root:
    python preview_characters.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from server.renderer import (
    DISQUALIFICATION_MASKS,
    FONT_PATH,
    FONT_SIZE,
    render_target_character,
)

SCALE = 6  # 100×100 → 600×600 for easy viewing
OUT_DIR = Path(__file__).parent / "images"
OUT_DIR.mkdir(exist_ok=True)

ALL_CHARS = ["L", "T", "V", "X", "A", "N", "Z", "E", "B", "C", "S", "O", "G", "Q"]


def arr_to_rgb(arr: np.ndarray, fg=(255, 255, 255), bg=(30, 30, 30)) -> Image.Image:
    """Convert a binary 100×100 array to an RGB PIL image."""
    h, w = arr.shape
    img = Image.new("RGB", (w, h), color=bg)
    for y in range(h):
        for x in range(w):
            if arr[y, x]:
                img.putpixel((x, y), fg)
    return img.resize((w * SCALE, h * SCALE), resample=Image.NEAREST)


def overlay_mask(base: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Render character in white and mask pixels in red on a dark background."""
    h, w = base.shape
    img = Image.new("RGB", (w, h), color=(30, 30, 30))
    for y in range(h):
        for x in range(w):
            if base[y, x] and mask[y, x]:
                img.putpixel((x, y), (255, 120, 120))  # shouldn't happen
            elif base[y, x]:
                img.putpixel((x, y), (255, 255, 255))  # character stroke
            elif mask[y, x]:
                img.putpixel((x, y), (220, 60, 60))    # protected zone in red
    return img.resize((w * SCALE, h * SCALE), resample=Image.NEAREST)


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Add a text label at the bottom of the image."""
    label_h = 40
    canvas = Image.new("RGB", (img.width, img.height + label_h), color=(15, 15, 15))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(str(FONT_PATH), 26)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tx = (canvas.width - (bbox[2] - bbox[0])) // 2
    ty = img.height + (label_h - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), text, font=font, fill=(200, 200, 200))
    return canvas


print(f"Rendering {len(ALL_CHARS)} characters at {SCALE}x scale → {OUT_DIR}/\n")

for char in ALL_CHARS:
    target = render_target_character(char)
    px = int(np.sum(target))
    masks = DISQUALIFICATION_MASKS.get(char, [])

    # Plain character image
    plain = arr_to_rgb(target)
    plain = add_label(plain, f"{char}  ({px}px foreground)")
    plain_path = OUT_DIR / f"{char}_target.png"
    plain.save(plain_path)

    if masks:
        # Combined mask overlay — all protected zones in red
        combined_mask = np.zeros((100, 100), dtype=np.uint8)
        for m in masks:
            combined_mask = np.maximum(combined_mask, m)
        total_protected = int(np.sum(combined_mask))

        overlay = overlay_mask(target, combined_mask)
        overlay = add_label(overlay, f"{char}  — red = protected zone ({total_protected}px)")
        overlay_path = OUT_DIR / f"{char}_with_mask.png"
        overlay.save(overlay_path)

        # Individual mask images — white on dark, one per component
        saved_masks = []
        for i, m in enumerate(masks):
            mask_img = arr_to_rgb(m, fg=(220, 60, 60), bg=(30, 30, 30))
            suffix = f"_mask_{i+1}" if len(masks) > 1 else "_mask"
            label_text = f"{char} disqualifier {i+1}/{len(masks)}  ({int(np.sum(m))}px)"
            mask_img = add_label(mask_img, label_text)
            mask_path = OUT_DIR / f"{char}{suffix}.png"
            mask_img.save(mask_path)
            saved_masks.append(mask_path.name)

        print(
            f"  {char}  → {plain_path.name}  +  {overlay_path.name}"
            f"  +  {', '.join(saved_masks)}  [{total_protected}px protected]"
        )
    else:
        print(f"  {char}  → {plain_path.name}  (no integrity constraint)")

print(f"\nDone. {len(list(OUT_DIR.iterdir()))} images written to {OUT_DIR}")
