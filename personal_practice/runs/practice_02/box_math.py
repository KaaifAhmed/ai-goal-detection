
from pathlib import Path
from PIL import Image

# Load our practice image to get its real dimensions.
image_path = Path(__file__).resolve().parent / "sample.jpg"

with Image.open(image_path) as image:
    image_width, image_height = image.size

print("Image dimensions:", image_width, "x", image_height)

# Bounding box from our YOLO prediction.
x1 = 417.4
y1 = 191.8
x2 = 449.3
y2 = 221.0

# 1. Calculate pixel width and height.
box_width = x2 - x1
box_height = y2 - y1

# 2. Calculate the centre in pixels.
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2

print("\nPixel measurements")
print("Width:", round(box_width, 2))
print("Height:", round(box_height, 2))
print("Centre:", (round(center_x, 2), round(center_y, 2)))

# 3. Convert values into the range 0 to 1.
# This is useful for YOLO training annotations.
normalized = (
    center_x / image_width,
    center_y / image_height,
    box_width / image_width,
    box_height / image_height,
)

print("\nNormalized YOLO coordinates")
print("center_x, center_y, width, height:")
print(tuple(round(value, 6) for value in normalized))
