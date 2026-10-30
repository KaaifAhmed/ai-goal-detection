"""
Test script for Phase 3: Tracking & Trajectory Continuity.
Tests Kalman tracking, gap-filling, and state maintenance (OBSERVED vs PREDICTED).
"""
import sys
from pathlib import Path
import yaml

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.video_loader import VideoLoader
from src.detector import BallDetector
from src.tracker import BallTrajectoryTracker, TrackState
from src.visualizer import VideoVisualizer


def main():
    config_path = current_dir / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = current_dir.parent.parent / "data"
    input_video = data_dir / "ground (2).mp4"
    output_video = current_dir / "output" / "tracked_ground_1.mp4"

    loader = VideoLoader(input_video)
    meta = loader.get_metadata()
    print(f"Loaded: {input_video.name} ({meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps)")

    detector = BallDetector(
        model_name=config.get("model_name", "yolo11s.pt"),
        conf_threshold=config.get("ball_detection_conf", 0.28),
        iou_threshold=config.get("iou_threshold", 0.45),
        max_aspect_ratio_deviation=config.get("max_aspect_ratio_deviation", 0.75),
    )

    tracker = BallTrajectoryTracker(
        max_prediction_gap=100,
        max_association_distance=120.0,
    )

    visualizer = VideoVisualizer(
        output_path=output_video,
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        trail_length=30,
    )

    observed_count = 0
    predicted_count = 0

    print("Running tracking pipeline and generating video...")
    for frame_idx, timestamp_s, frame in loader.frames():
        # Get raw detections
        raw_balls = detector.detect(frame)

        # Update tracker
        tracked_ball = tracker.update(raw_balls, frame_idx, timestamp_s)

        if tracked_ball is not None:
            if tracked_ball["state"] == TrackState.OBSERVED:
                observed_count += 1
            else:
                predicted_count += 1

        visualizer.write_frame(frame, tracked_ball, raw_balls, frame_idx, timestamp_s)

    visualizer.release()
    loader.release()

    print(f"\n--- Phase 3 Tracking Summary ---")
    print(f"Output video: {output_video}")
    print(f"Total OBSERVED states: {observed_count}")
    print(f"Total PREDICTED (gap-filled) states: {predicted_count}")
    print(f"Total continuous tracked frames: {observed_count + predicted_count}")


if __name__ == "__main__":
    main()
