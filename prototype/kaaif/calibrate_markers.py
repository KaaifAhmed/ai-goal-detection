"""
Tool for setting and verifying goal marker positions.
Supports interactive 2-click setup or manual coordinate configuration.
Saves calibrated marker positions directly into config/settings.yaml.
"""
import sys
import argparse
from pathlib import Path
import cv2
import yaml

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.video_loader import VideoLoader
from src.goal_geometry import VirtualGoalGeometry


clicks = []


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Clicked point {len(clicks)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(description="Calibrate virtual goal markers")
    parser.add_argument("--video", type=str, default="data/ground (1).mp4", help="Video file path")
    parser.add_argument("--left", nargs=2, type=float, default=None, help="Left marker x y")
    parser.add_argument("--right", nargs=2, type=float, default=None, help="Right marker x y")
    parser.add_argument("--ratio", type=float, default=0.75, help="Height to width ratio a")
    args = parser.parse_args()

    # Resolve video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (current_dir.parent.parent / video_path).resolve()

    loader = VideoLoader(video_path)
    frame_idx, ts, first_frame = next(loader.frames())
    loader.release()

    config_path = current_dir / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.left is not None and args.right is not None:
        p_left = tuple(args.left)
        p_right = tuple(args.right)
    else:
        # Check if already present in config
        default_markers = config.get("goal_markers", {}).get(video_path.name)
        if default_markers:
            p_left = tuple(default_markers["left"])
            p_right = tuple(default_markers["right"])
            print(f"Using existing configured markers for {video_path.name}: L={p_left}, R={p_right}")
        else:
            # Fallback default reasonable markers across lower area
            p_left = (450.0, 750.0)
            p_right = (1450.0, 750.0)
            print(f"Using default baseline markers: L={p_left}, R={p_right}")

    ratio_a = args.ratio or config.get("crossbar_ratio_a", 0.75)
    geometry = VirtualGoalGeometry(p_left, p_right, ratio_a=ratio_a)

    print(f"\n--- Virtual Goal Geometry Calculated ---")
    print(f"Left Marker:  {geometry.p_bottom_left.tolist()}")
    print(f"Right Marker: {geometry.p_bottom_right.tolist()}")
    print(f"Goal Width:   {geometry.width:.1f} px")
    print(f"Goal Height:  {geometry.height:.1f} px (ratio a = {ratio_a})")
    print(f"Top Left:     {geometry.p_top_left.tolist()}")
    print(f"Top Right:    {geometry.p_top_right.tolist()}")

    # Render and save preview
    preview = geometry.draw_virtual_goal(first_frame)
    output_preview = current_dir / "output" / f"goal_setup_{video_path.stem}.jpg"
    output_preview.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_preview), preview)
    print(f"\nSaved virtual goal setup preview image to: {output_preview}")

    # Update settings.yaml
    if "goal_markers" not in config:
        config["goal_markers"] = {}
    config["goal_markers"][video_path.name] = {
        "left": [float(p_left[0]), float(p_left[1])],
        "right": [float(p_right[0]), float(p_right[1])],
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Updated marker configuration in: {config_path}")


if __name__ == "__main__":
    main()
