"""
Test script for Phase 2: Video Ingestion & Base Model Perception.
Processes the critical shot window (e.g. frames 165-195) in ground (1).mp4.
"""
import sys
from pathlib import Path
import yaml

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.video_loader import VideoLoader
from src.detector import BallDetector


def main():
    config_path = current_dir / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = current_dir.parent.parent / "data"
    sample_video = data_dir / "ground (1).mp4"
    if not sample_video.exists():
        print(f"Sample video not found at: {sample_video}")
        return

    print(f"--- Phase 2 Test: Video Ingestion & Perception ---")
    print(f"Loading video: {sample_video.name}")
    loader = VideoLoader(sample_video)
    metadata = loader.get_metadata()
    print(f"Resolution: {metadata['width']}x{metadata['height']} | FPS: {metadata['fps']:.2f} | Frames: {metadata['total_frames']}")

    detector = BallDetector(
        conf_threshold=config.get("ball_detection_conf", 0.25),
        iou_threshold=config.get("iou_threshold", 0.45),
        roi_box=(0.0, 0.15, 1.0, 1.0),  # Top 15% is sky, keep rest
    )

    detections = []
    print("\nScanning shot window (frames 165-192)...")
    for frame_idx, timestamp_s, frame in loader.frames():
        if frame_idx < 165:
            continue
        if frame_idx > 192:
            break

        balls = detector.detect(frame)
        if balls:
            best_ball = max(balls, key=lambda b: b["confidence"])
            detections.append((frame_idx, timestamp_s, best_ball))
            print(f"Frame {frame_idx:03d} [{timestamp_s:.3f}s]: center=({best_ball['center'][0]:.1f}, {best_ball['center'][1]:.1f}), radius={best_ball['radius']:.1f}px, conf={best_ball['confidence']:.2f}")

    loader.release()
    print(f"\nPhase 2 Verification Complete!")
    print(f"Successfully detected ball trajectory across {len(detections)} frames during the shot.")


if __name__ == "__main__":
    main()
