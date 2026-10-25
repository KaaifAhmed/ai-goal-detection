"""Experimental football goal detector using a drawn pixel-size reference.

Uses pretrained YOLO26x. Does not train or fine-tune.
Outputs a GOAL CANDIDATE for review, not a physically confirmed goal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("Data"))
    parser.add_argument("--video", default=None)
    parser.add_argument("--weights", default="yolo26x.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/goal_size_gate"))
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size-tolerance", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--setup-frame", type=int, default=0)
    return parser.parse_args()


def ask_measurement(question, unit):
    while True:
        try:
            value = float(input(f"{question} ({unit}): ").replace(",", "."))
            if math.isfinite(value) and value > 0:
                return value
        except ValueError:
            pass
        print("Enter a number greater than zero.")


def choose_video(folder, filename):
    if not folder.is_dir():
        raise FileNotFoundError(f"Video folder not found: {folder.resolve()}")

    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    if filename:
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path.resolve()}")
        return path

    if not videos:
        raise FileNotFoundError(f"No videos found in {folder.resolve()}")

    print("\nVideos:")
    for index, path in enumerate(videos, start=1):
        print(f"{index}. {path.name}")

    choice = input("Enter video filename or list number: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(videos):
        return videos[int(choice) - 1]

    path = folder / choice
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path.resolve()}")
    return path


def preview_scale(frame, max_width=1280, max_height=760):
    height, width = frame.shape[:2]
    return min(1.0, max_width / width, max_height / height)


def resize_preview(frame, scale):
    if scale >= 0.999:
        return frame.copy()

    return cv2.resize(
        frame,
        (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
    )


def draw_goal_area(frame):
    """Click four corners in order:
    top-left, top-right, bottom-right, bottom-left.
    """
    scale = preview_scale(frame)
    points = []
    window = "Draw Virtual Goal Area"

    def redraw():
        image = resize_preview(frame, scale)

        cv2.putText(
            image,
            "Click: top-left, top-right, bottom-right, bottom-left",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image,
            "ENTER=accept | R=reset | Q=cancel",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        for index, (x, y) in enumerate(points):
            px, py = int(x * scale), int(y * scale)
            cv2.circle(image, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(
                image, str(index + 1), (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        for index in range(1, len(points)):
            p1 = tuple(int(v * scale) for v in points[index - 1])
            p2 = tuple(int(v * scale) for v in points[index])
            cv2.line(image, p1, p2, (0, 255, 0), 2)

        if len(points) == 4:
            p1 = tuple(int(v * scale) for v in points[-1])
            p2 = tuple(int(v * scale) for v in points[0])
            cv2.line(image, p1, p2, (0, 255, 0), 2)

        cv2.imshow(window, image)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x / scale), int(y / scale)))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError("Goal-area selection cancelled.")

        if key == ord("r"):
            points.clear()
            redraw()

        if key in (10, 13) and len(points) == 4:
            break

    cv2.destroyWindow(window)

    polygon = np.asarray(points, dtype=np.int32)
    if abs(cv2.contourArea(polygon)) < 20:
        raise ValueError("Goal area is too small.")

    return polygon


def draw_reference_size(frame):
    """Draw a reference rectangle, which need not surround a real ball."""
    scale = preview_scale(frame)
    display = resize_preview(frame, scale)

    print("\nREFERENCE BALL SIZE")
    print("Draw a rectangle representing the expected pixel size")
    print("of the ball at your chosen goal-crossing state.")
    print("It may be drawn over an empty patch near the goal plane.")

    roi = cv2.selectROI(
        "Draw Expected Ball Size",
        display,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Draw Expected Ball Size")

    x, y, width, height = map(int, roi)

    if width < 3 or height < 3:
        raise ValueError("Reference rectangle is missing or too small.")

    width = int(round(width / scale))
    height = int(round(height / scale))
    x = int(round(x / scale))
    y = int(round(y / scale))

    return width, height, (x, y, width, height)


def find_ball_class(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)

    for class_id, name in items:
        if str(name).strip().lower() == "sports ball":
            return int(class_id)

    raise RuntimeError("Model has no 'sports ball' class.")


def point_inside_goal(cx, cy, polygon):
    return cv2.pointPolygonTest(
        polygon, (float(cx), float(cy)), False
    ) >= 0


def main():
    args = parse_args()

    if not 0 < args.conf < 1:
        raise ValueError("--conf must be between 0 and 1.")

    if not 0.05 <= args.size_tolerance <= 0.60:
        raise ValueError("--size-tolerance must be between 0.05 and 0.60.")

    video = choose_video(args.input, args.video)

    print("\nEnter setup measurements.")
    ball_diameter_cm = ask_measurement("Ball diameter", "cm")
    block_gap_cm = ask_measurement("Clear gap between blocks", "cm")
    goal_height_cm = ask_measurement("Virtual goal height", "cm")
    camera_distance_m = ask_measurement(
        "Camera lens to goal plane distance", "metres"
    )

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    setup_index = min(args.setup_frame, max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, setup_index)
    ok, setup_frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Cannot read setup frame.")

    print("\nDraw the goal area's four corners.")
    print("If there is no crossbar, use your team's agreed virtual height.")

    goal_polygon = draw_goal_area(setup_frame)
    ref_w, ref_h, ref_rect = draw_reference_size(setup_frame)

    # Use the geometric mean so small box width/height differences
    # do not dominate the comparison.
    reference_size_px = math.sqrt(ref_w * ref_h)

    # A ball may be slightly smaller than the drawn reference because
    # detector boxes and manual drawing are not perfectly exact.
    size_threshold_px = reference_size_px * (1 - args.size_tolerance)

    print(f"\nReference apparent size: {reference_size_px:.1f} pixels")
    print(f"Size threshold: {size_threshold_px:.1f} pixels")

    print("Loading pretrained model. First run may download weights.")
    model = YOLO(args.weights)
    class_id = find_ball_class(model.names)

    args.output.mkdir(parents=True, exist_ok=True)
    annotated_dir = args.output / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    output_video = annotated_dir / f"{video.stem}_goal_check.mp4"
    csv_path = args.output / "detections.csv"
    result_path = args.output / "goal_result.json"
    setup_path = args.output / "setup.json"

    setup_path.write_text(
        json.dumps(
            {
                "video": video.name,
                "weights": args.weights,
                "ball_diameter_cm": ball_diameter_cm,
                "block_gap_cm": block_gap_cm,
                "virtual_goal_height_cm": goal_height_cm,
                "camera_to_goal_plane_m": camera_distance_m,
                "goal_polygon_xy": goal_polygon.tolist(),
                "reference_rectangle_xywh": list(ref_rect),
                "reference_size_px": reference_size_px,
                "size_threshold_px": size_threshold_px,
                "size_tolerance": args.size_tolerance,
                "note": (
                    "The drawn reference sets a pixel threshold. Physical "
                    "measurements are recorded but do not themselves calculate "
                    "exact 3D geometry in this heuristic."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not cap.isOpened() or not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Cannot open input or output video.")

    fields = [
        "frame", "timestamp_s", "confidence",
        "center_x", "center_y", "box_width_px", "box_height_px",
        "ball_size_px", "size_ratio", "inside_goal_area", "status",
    ]

    total_detections = 0
    goal_candidate_frame = None
    frames_processed = 0
    previous = None

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fields)
        csv_writer.writeheader()

        try:
            while True:
                if args.max_frames and frames_processed >= args.max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                result = model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    classes=[class_id],
                    max_det=20,
                    device=args.device,
                    verbose=False,
                )[0]

                detections = []

                if result.boxes is not None:
                    for coords, confidence in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                    ):
                        x1, y1, x2, y2 = map(float, coords)
                        box_w = max(1.0, x2 - x1)
                        box_h = max(1.0, y2 - y1)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        size_px = math.sqrt(box_w * box_h)

                        detections.append({
                            "box": (x1, y1, x2, y2),
                            "cx": cx,
                            "cy": cy,
                            "w": box_w,
                            "h": box_h,
                            "size": size_px,
                            "ratio": size_px / reference_size_px,
                            "confidence": float(confidence),
                            "inside": point_inside_goal(cx, cy, goal_polygon),
                        })

                total_detections += len(detections)
                annotated = frame.copy()

                cv2.polylines(
                    annotated,
                    [goal_polygon.reshape((-1, 1, 2))],
                    True, (0, 255, 255), 2,
                )

                rx, ry, rw, rh = ref_rect
                cv2.rectangle(
                    annotated, (rx, ry), (rx + rw, ry + rh),
                    (255, 0, 255), 2,
                )
                cv2.putText(
                    annotated,
                    f"SIZE REFERENCE: {reference_size_px:.1f}px",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 255), 2,
                )

                for det in detections:
                    x1, y1, x2, y2 = map(int, det["box"])

                    reached_size = det["size"] >= size_threshold_px
                    candidate = (
                        det["inside"]
                        and reached_size
                        and previous is not None
                        and det["size"] >= previous["size"] * 1.01
                        and math.hypot(
                            det["cx"] - previous["cx"],
                            det["cy"] - previous["cy"],
                        ) >= 2
                    )

                    if candidate and goal_candidate_frame is None:
                        goal_candidate_frame = frames_processed

                    if det["inside"] and reached_size:
                        color = (0, 220, 0)
                        status = "SIZE_REACHED_INSIDE_AREA"
                    else:
                        color = (0, 165, 255)
                        status = "NOT_CANDIDATE"

                    if candidate:
                        status = "GOAL_CANDIDATE_REVIEW_REQUIRED"
                        color = (0, 0, 255)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ball? {det['confidence']:.2f} "
                        f"size={det['size']:.1f}px",
                        (max(0, x1), max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 2,
                    )

                    csv_writer.writerow({
                        "frame": frames_processed,
                        "timestamp_s": round(frames_processed / fps, 4),
                        "confidence": round(det["confidence"], 4),
                        "center_x": round(det["cx"], 2),
                        "center_y": round(det["cy"], 2),
                        "box_width_px": round(det["w"], 2),
                        "box_height_px": round(det["h"], 2),
                        "ball_size_px": round(det["size"], 2),
                        "size_ratio": round(det["ratio"], 3),
                        "inside_goal_area": int(det["inside"]),
                        "status": status,
                    })

                    # Simple single-target comparison for the next frame.
                    # This can switch objects if there are multiple detections;
                    # inspect the saved video carefully.
                    if previous is None or det["confidence"] > previous["confidence"]:
                        previous = {
                            "cx": det["cx"],
                            "cy": det["cy"],
                            "size": det["size"],
                            "confidence": det["confidence"],
                        }

                writer.write(annotated)
                frames_processed += 1

                if frames_processed % 50 == 0:
                    print(f"Processed {frames_processed} frames...")

        finally:
            cap.release()
            writer.release()

    if goal_candidate_frame is not None:
        decision = "GOAL_CANDIDATE_REVIEW_REQUIRED"
        explanation = (
            "A detected sports-ball box reached the size threshold inside "
            "the drawn area. Review the replay; this is not proof of a goal."
        )
    elif total_detections == 0:
        decision = "UNCERTAIN"
        explanation = "No sports-ball detections. This does not prove no goal."
    else:
        decision = "NO_GOAL_CANDIDATE_FOUND"
        explanation = (
            "The heuristic did not find a candidate. Missed detections or "
            "occlusion may still hide a true goal."
        )

    result_path.write_text(
        json.dumps(
            {
                "video": video.name,
                "decision": decision,
                "candidate_frame": goal_candidate_frame,
                "frames_processed": frames_processed,
                "sports_ball_detection_count": total_detections,
                "explanation": explanation,
                "warning": (
                    "Single-camera heuristic, not validated goal-line technology. "
                    "A slipper classified as sports ball can cause false results."
                ),
                "annotated_video": str(output_video.resolve()),
                "detections_csv": str(csv_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPROCESSING COMPLETE")
    print("Decision:", decision)
    print(explanation)
    print("Annotated video:", output_video.resolve())
    print("Detection CSV:", csv_path.resolve())
    print("Result JSON:", result_path.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())