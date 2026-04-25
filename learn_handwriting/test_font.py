import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

FONT_PATH = Path("Roboto-Regular.ttf")
FONT_URL = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"

if not FONT_PATH.exists():
    print("Downloading font...")
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)

font = ImageFont.truetype(str(FONT_PATH), 80)
img = Image.new("L", (100, 100), color=0)
draw = ImageDraw.Draw(img)

character = "L"
bbox = draw.textbbox((0, 0), character, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (100 - text_w) / 2 - bbox[0]
y = (100 - text_h) / 2 - bbox[1]

draw.text((x, y), character, font=font, fill=255)

arr = np.array(img, dtype=np.uint8)

# Check thickness before dilation
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(arr, kernel, iterations=1)

Image.fromarray(arr).save("test_normal.png")
Image.fromarray(dilated).save("test_dilated.png")

print("Done. Check test_normal.png and test_dilated.png")
