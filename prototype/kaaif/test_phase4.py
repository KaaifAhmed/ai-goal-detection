"""
Verification script for Phase 4: Proportional Goal Geometry & Plane Modeling.
1. Automatically detects physical goal markers (concrete blocks) in the video.
2. Computes skewed ground baseline, perpendicular upright posts, and proportional crossbar.
3. Renders full annotated verification video with the virtual goal erected on the actual goalposts.
"""
import sys
from pathlib import Path
import numpy as np
import yaml

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.goal_geometry import VirtualGoalGeometry
from src.marker_detector import GoalMarkerDetector
from src.video_loader import VideoLoader
from src.detector import BallDetector
from src.tracker import BallTrajectoryTracker
from src.visualizer import VideoVisualizer


def run_unit_tests():
    print("=== 1. Testing VirtualGoalGeometry Mathematical Invariants ===")

    # Test Case 1: Standard horizontal ground markers
    p_left = (200.0, 800.0)
    p_right = (1000.0, 800.0)
    goal = VirtualGoalGeometry(p_left, p_right, ratio_a=0.75)
    assert abs(goal.width - 800.0) < 1e-3
    assert abs(goal.height - 600.0) < 1e-3
    print("  [PASS] Width = 800px, Height = 600px (0.75 * W).")

    # Test Case 2: Tilted ground skew compensation
    p_left_tilt = (440.0, 870.0)
    p_right_tilt = (1710.0, 940.0)
    tilted_goal = VirtualGoalGeometry(p_left_tilt, p_right_tilt, ratio_a=0.50)
    assert tilted_goal.skew_angle_deg > 0.0
    # Upward vector must point into negative y
    assert tilted_goal.u_up[1] < 0.0
    print(f"  [PASS] Tilted Ground: Tilt = {tilted_goal.skew_angle_deg:.2f} deg, Upward normal correctly oriented.")


def render_video_with_detected_markers():
    print("\n=== 2. Detecting Physical Markers & Rendering Virtual Goal ===")
    config_path = current_dir / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = current_dir.parent.parent / "data"
    input_video = data_dir / "ground (1).mp4"
    output_video = current_dir / "output" / "virtual_goal_ground_1.mp4"

    loader = VideoLoader(input_video)
    meta = loader.get_metadata()

    # Step 1: Detect physical goal markers from first frame
    frame_idx, ts, first_frame = next(loader.frames())
    marker_det = GoalMarkerDetector()
    p_left, p_right, debug_info = marker_det.detect_markers(first_frame)

    ratio_a = config.get("crossbar_ratio_a", 0.75)
    virtual_goal = VirtualGoalGeometry(
        left_marker=p_left,
        right_marker=p_right,
        ratio_a=ratio_a,
        left_bbox=debug_info.get("left_marker_bbox"),
        right_bbox=debug_info.get("right_marker_bbox"),
    )

    print(f"Detected Physical Markers:")
    print(f"  Left Post:  {p_left} (Box: {debug_info.get('left_marker_bbox')})")
    print(f"  Right Post: {p_right} (Box: {debug_info.get('right_marker_bbox')})")
    print(f"  Goal Width: {virtual_goal.width:.1f} px")
    print(f"  Goal Height:{virtual_goal.height:.1f} px (ratio a = {ratio_a})")
    print(f"  Ground Tilt:{virtual_goal.skew_angle_deg:+.2f} degrees")

    # Step 2: Initialize tracker, detector, and visualizer
    detector = BallDetector(
        model_name=config.get("model_name", "yolo11s.pt"),
        conf_threshold=config.get("ball_detection_conf", 0.28),
        iou_threshold=config.get("iou_threshold", 0.45),
        max_aspect_ratio_deviation=config.get("max_aspect_ratio_deviation", 0.75),
    )

    tracker = BallTrajectoryTracker(
        max_prediction_gap=5,
        max_association_distance=120.0,
    )

    visualizer = VideoVisualizer(
        output_path=output_video,
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        trail_length=30,
        virtual_goal=virtual_goal,
    )

    print(f"Rendering updated video to: {output_video} ...")
    # Reset loader to beginning
    loader.release()
    loader = VideoLoader(input_video)

    for frame_idx, timestamp_s, frame in loader.frames():
        raw_balls = detector.detect(frame)
        tracked_ball = tracker.update(raw_balls, frame_idx, timestamp_s)
        visualizer.write_frame(frame, tracked_ball, raw_balls, frame_idx, timestamp_s)

    visualizer.release()
    loader.release()

    print(f"\n[DONE] Rendered video saved to: {output_video}")


def main():
    run_unit_tests()
    render_video_with_detected_markers()


if __name__ == "__main__":
    main()
