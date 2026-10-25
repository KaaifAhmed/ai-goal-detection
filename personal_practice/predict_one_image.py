from pathlib import Path
from ultralytics import YOLO

# 1. Find the folder containing this Python file.
PROJECT_DIR = Path(__file__).resolve().parent

# 2. Select one image.
IMAGE_PATH = PROJECT_DIR / "data" / "sample.png"
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

# 3. Load a pretrained object-detection model.
model = YOLO("yolo11n.pt")

# 4. Ask the model to make predictions.
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=640,
    save=True,
    project=str(PROJECT_DIR / "runs"),
    name="practice_02",
    exist_ok=True,
    verbose=False,
)

# 5. Read and print the predictions.
for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected above the threshold.")
        continue

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = [
            round(float(value), 1)
            for value in box.xyxy[0].tolist()
        ]

        print(f"Object: {class_name}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Bounding box: {(x1, y1, x2, y2)}")
        print("-" * 30)

print("Finished inference.")
print("Check the runs/practice_02 folder for the output image.")
