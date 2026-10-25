"""Run pretrained YOLO26x on local football videos (inference only).

- Looks for the COCO class named 'sports ball'.
- Saves annotated MP4 videos, detections.csv, and video_summary.csv.
- Does not train or fine-tune the model.

Example:
    python run_yolo26x_test.py --input Data --only-video "WhatsApp Video 2026-10-15 at 3.24.29 PM.mp4" --max-frames 100
    python run_yolo26x_test.py --input Data
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
    parser = argparse.ArgumentParser(
        description="Test pretrained YOLO26x sports-ball detection on local videos."
    )
    parser.add_argument("--input", type=Path, default=Path("Data"),
                        help="Folder containing videos (default: Data)")
    parser.add_argument("--output", type=Path, default=Path("runs/yolo26x_test"),
                        help="Output folder (default: runs/yolo26x_test)")
    parser.add_argument("--weights", default="yolo26x.pt",
                        help="Ultralytics model weights (default: yolo26x.pt)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Detection confidence threshold from 0 to 1 (default: 0.10)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Inference image size (default: 960)")
    parser.add_argument("--device", default="cpu",
                        help="Use cpu, or CUDA device such as 0 if correctly installed")
    parser.add_argument("--only-video", default=None,
                        help="Optional exact video filename inside --input")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process at most this many frames per video; 0 means all frames")
    return parser.parse_args()


def find_sports_ball_class(names: dict[int, str] | list[str]) -> int:
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, name in pairs:
        if str(name).strip().lower() == "sports ball":
            return int(class_id)
    available = ", ".join(str(v) for v in (names.values() if isinstance(names, dict) else names))
    raise RuntimeError(
        "This model does not have a class named 'sports ball'. "
        f"Available classes: {available[:500]}"
    )


def get_videos(folder: Path, only_video: str | None) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder.resolve()}")
    if only_video:
        path = folder / only_video
        if not path.is_file():
            raise FileNotFoundError(f"Requested video not found: {path.resolve()}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {path.suffix}")
        return [path]
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


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
    max_frames: int,
    detections_writer: csv.DictWriter,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return {
            "video": video_path.name, "status": "failed_to_open", "frames": 0,
            "frames_with_detection": 0, "frame_detection_percent": 0.0,
            "detection_count": 0, "fps": "", "width": "", "height": "",
            "annotated_video": "",
        }

    fps = safe_fps(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotated_dir = output_dir / "annotated_videos"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    output_video = annotated_dir / f"{video_path.stem}_yolo26x.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_video.resolve()}")

    frame_index = 0
    frames_with_detection = 0
    detection_count = 0

    print(f"\n[VIDEO] {video_path.name} | {width}x{height} | {fps:.2f} FPS | frames={declared_frames}")
    try:
        while True:
            if max_frames > 0 and frame_index >= max_frames:
                print(f"  Stopping at requested limit of {max_frames} frames.")
                break

            ok, frame = cap.read()
            if not ok:
                break

            timestamp_s = frame_index / fps
            result = model.predict(
                source=frame,
                conf=conf,
                imgsz=imgsz,
                classes=[ball_class_id],
                max_det=10,
                device=device,
                verbose=False,
            )[0]

            boxes = result.boxes
            found_any = boxes is not None and len(boxes) > 0
            if found_any:
                frames_with_detection += 1
                xyxy_array = boxes.xyxy.cpu().numpy()
                confidence_array = boxes.conf.cpu().numpy()
                for xyxy, score in zip(xyxy_array, confidence_array):
                    x1, y1, x2, y2 = (float(v) for v in xyxy)
                    detection_count += 1
                    detections_writer.writerow({
                        "video": video_path.name,
                        "frame_index": frame_index,
                        "timestamp_s": round(timestamp_s, 4),
                        "class_name": "sports ball",
                        "confidence": round(float(score), 6),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "center_x": round((x1 + x2) / 2, 2),
                        "center_y": round((y1 + y2) / 2, 2),
                        "box_width_px": round(x2 - x1, 2),
                        "box_height_px": round(y2 - y1, 2),
                        "detected": 1,
                    })
            else:
                detections_writer.writerow({
                    "video": video_path.name,
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp_s, 4),
                    "class_name": "", "confidence": "",
                    "x1": "", "y1": "", "x2": "", "y2": "",
                    "center_x": "", "center_y": "",
                    "box_width_px": "", "box_height_px": "",
                    "detected": 0,
                })

            # Draw only sports-ball detections to keep output easy to inspect.
            writer.write(result.plot())
            frame_index += 1
            if frame_index % 10 == 0:
                print(f"  processed {frame_index} frames...")
    finally:
        cap.release()
        writer.release()

    pct = 100.0 * frames_with_detection / frame_index if frame_index else 0.0
    summary = {
        "video": video_path.name,
        "status": "ok",
        "frames": frame_index,
        "frames_with_detection": frames_with_detection,
        "frame_detection_percent": round(pct, 2),
        "detection_count": detection_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "annotated_video": str(output_video),
    }
    print(f"  result: {frames_with_detection}/{frame_index} frames had a sports-ball box ({pct:.2f}%), boxes={detection_count}")
    print(f"  annotated video: {output_video.resolve()}")
    return summary


def main() -> int:
    args = parse_args()
    if not 0 < args.conf < 1:
        print("ERROR: --conf must be greater than 0 and less than 1.", file=sys.stderr)
        return 2
    if args.imgsz < 160:
        print("ERROR: --imgsz must be at least 160.", file=sys.stderr)
        return 2
    if args.max_frames < 0:
        print("ERROR: --max-frames cannot be negative.", file=sys.stderr)
        return 2

    videos = get_videos(args.input, args.only_video)
    if not videos:
        print(f"No supported videos found in: {args.input.resolve()}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading model weights: {args.weights}")
    print("This is inference only: no training or fine-tuning.")
    print("The first run needs internet to download weights if they are not cached.")
    model = YOLO(args.weights)
    ball_class_id = find_sports_ball_class(model.names)
    print(f"Using class 'sports ball' (class id {ball_class_id})")
    print(f"Videos selected: {len(videos)} | device={args.device} | imgsz={args.imgsz} | conf={args.conf}")
    print("Note: YOLO26x can be very slow on CPU. Try --max-frames 100 for a quick test.")

    fields = ["video", "frame_index", "timestamp_s", "class_name", "confidence",
              "x1", "y1", "x2", "y2", "center_x", "center_y",
              "box_width_px", "box_height_px", "detected"]
    summaries: list[dict[str, Any]] = []
    detections_path = args.output / "detections.csv"
    summary_path = args.output / "video_summary.csv"

    with detections_path.open("w", newline="", encoding="utf-8-sig") as f:
        detection_writer = csv.DictWriter(f, fieldnames=fields)
        detection_writer.writeheader()
        for video in videos:
            summaries.append(process_video(
                video_path=video, output_dir=args.output, model=model,
                ball_class_id=ball_class_id, conf=args.conf, imgsz=args.imgsz,
                device=args.device, max_frames=args.max_frames,
                detections_writer=detection_writer,
            ))

    summary_fields = ["video", "status", "frames", "frames_with_detection",
                      "frame_detection_percent", "detection_count", "fps",
                      "width", "height", "annotated_video"]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nYOLO26x test complete.")
    print(f"Detections: {detections_path.resolve()}")
    print(f"Summary:    {summary_path.resolve()}")
    print(f"Videos:     {(args.output / 'annotated_videos').resolve()}")
    print("Remember: detection percentage is not accuracy; inspect the video for missed balls and false detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
