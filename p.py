"""Run a pretrained YOLO detector on local football videos.

This is an inference baseline only: it does not train or fine-tune a model.
It searches the model's COCO classes for 'sports ball', processes each video,
then saves annotated videos, raw detections, and a per-video summary.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrained football detection baseline")
    parser.add_argument("--input", type=Path, default=Path("Data/"),
                        help="Folder containing original videos (default: Data/)")
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"),
                        help="Output folder (default: runs/baseline)")
    parser.add_argument("--weights", default="yolo26n.pt",
                        help="Pretrained Ultralytics weights (default: yolo26n.pt)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Detection confidence threshold; lower can find more candidates (default: 0.10)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640; try 960 if the ball is tiny)")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu, or a CUDA device such as 0 if properly installed")
    return parser.parse_args()


def find_ball_class(names: dict[int, str] | list[str]) -> int:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in items:
        if str(name).strip().lower() == "sports ball":
            return int(class_id)
    available = ", ".join(str(name) for name in names.values()) if isinstance(names, dict) else ", ".join(names)
    raise RuntimeError(
        "The selected model has no class named 'sports ball'. "
        f"Available classes include: {available[:500]}"
    )


def get_video_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def safe_fps(value: float) -> float:
    return value if value and value > 0 else 30.0


def process_video(
    video_path: Path,
    output_dir: Path,
    model: YOLO,
    ball_class_id: int,
    conf: float,
    imgsz: int,
    device: str,
    detections_writer: csv.DictWriter,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return {
            "video": video_path.name, "status": "failed_to_open", "frames": 0,
            "frames_with_detection": 0, "frame_detection_percent": 0.0,
            "detection_count": 0, "fps": "", "width": "", "height": "",
        }

    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotated_dir = output_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotated_dir / f"{video_path.stem}_baseline.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    frame_idx = 0
    frames_with_detection = 0
    detection_count = 0

    print(f"\n[VIDEO] {video_path.name} | {width}x{height} | {fps:.2f} FPS | declared frames={declared_frames}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps
        prediction = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            classes=[ball_class_id],
            max_det=10,
            device=device,
            verbose=False,
        )[0]

        boxes = prediction.boxes
        frame_rows: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            xyxy_values = boxes.xyxy.cpu().numpy()
            confidence_values = boxes.conf.cpu().numpy()
            for xyxy, confidence in zip(xyxy_values, confidence_values):
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                frame_rows.append({
                    "video": video_path.name,
                    "frame_index": frame_idx,
                    "timestamp_s": round(timestamp_s, 4),
                    "class_name": "sports ball",
                    "confidence": round(float(confidence), 6),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "center_x": round((x1 + x2) / 2.0, 2),
                    "center_y": round((y1 + y2) / 2.0, 2),
                    "box_width_px": round(x2 - x1, 2),
                    "box_height_px": round(y2 - y1, 2),
                    "detected": 1,
                })

        if frame_rows:
            frames_with_detection += 1
            detection_count += len(frame_rows)
            for row in frame_rows:
                detections_writer.writerow(row)
        else:
            detections_writer.writerow({
                "video": video_path.name,
                "frame_index": frame_idx,
                "timestamp_s": round(timestamp_s, 4),
                "class_name": "", "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "center_x": "", "center_y": "",
                "box_width_px": "", "box_height_px": "",
                "detected": 0,
            })

        # The model annotation is only a visual aid; a box is not proof of a goal.
        annotated = prediction.plot()
        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  processed {frame_idx} frames...")

    cap.release()
    writer.release()
    pct = (100.0 * frames_with_detection / frame_idx) if frame_idx else 0.0
    summary = {
        "video": video_path.name,
        "status": "ok",
        "frames": frame_idx,
        "frames_with_detection": frames_with_detection,
        "frame_detection_percent": round(pct, 2),
        "detection_count": detection_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "annotated_video": str(out_path),
    }
    print(
        f"  result: {frames_with_detection}/{frame_idx} frames had a sports-ball detection "
        f"({pct:.1f}%), boxes={detection_count}"
    )
    print(f"  annotated video: {out_path}")
    return summary


def main() -> int:
    args = parse_args()
    if not 0.0 < args.conf < 1.0:
        print("ERROR: --conf must be greater than 0 and less than 1.", file=sys.stderr)
        return 2
    if args.imgsz < 160:
        print("ERROR: --imgsz must be at least 160.", file=sys.stderr)
        return 2

    videos = get_video_files(args.input)
    if not videos:
        args.input.mkdir(parents=True, exist_ok=True)
        print(f"No supported video files found in: {args.input.resolve()}")
        print("Put your original .mp4/.mov/.avi/.mkv videos in that folder, then run this script again.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading pretrained model: {args.weights}")
    print("First run may download model weights and requires internet access.")
    model = YOLO(args.weights)
    ball_class_id = find_ball_class(model.names)
    print(f"Model class selected: sports ball (class id {ball_class_id})")
    print("This run is inference only: no fine-tuning is happening.")

    detections_path = args.output / "detections.csv"
    summary_path = args.output / "video_summary.csv"
    fieldnames = [
        "video", "frame_index", "timestamp_s", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y",
        "box_width_px", "box_height_px", "detected",
    ]
    summaries: list[dict[str, Any]] = []
    with detections_path.open("w", newline="", encoding="utf-8-sig") as f:
        detections_writer = csv.DictWriter(f, fieldnames=fieldnames)
        detections_writer.writeheader()
        for video in videos:
            summary = process_video(
                video_path=video,
                output_dir=args.output,
                model=model,
                ball_class_id=ball_class_id,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                detections_writer=detections_writer,
            )
            summaries.append(summary)

    summary_fields = [
        "video", "status", "frames", "frames_with_detection", "frame_detection_percent",
        "detection_count", "fps", "width", "height", "annotated_video",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        summary_writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        summary_writer.writeheader()
        summary_writer.writerows(summaries)

    print("\nBASELINE RUN COMPLETE")
    print(f"Detections CSV: {detections_path.resolve()}")
    print(f"Per-video summary: {summary_path.resolve()}")
    print(f"Annotated videos: {(args.output / 'annotated_videos').resolve()}")
    print("Next step: watch the annotated clips and record misses and false detections. These percentages are NOT accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
