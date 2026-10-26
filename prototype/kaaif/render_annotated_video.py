"""
Renders updated annotated verification video for ground (1).mp4
with enhanced false-positive filtering (slipper & shoulder suppression)
and motion-blur tolerance.
"""
import sys
from pathlib import Path
import yaml

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from src.video_loader import VideoLoader
from src.detector import BallDetector
from src.visualizer import VideoVisualizer


def main():
    config_path = current_dir / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = current_dir.parent.parent / "data"
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}
    videos = sorted(
        (
            path
            for path in data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in video_extensions
        ),
        key=lambda path: path.name.lower(),
    )

    if not videos:
        print(f"No videos found in: {data_dir}")
        return
    while True:
        print("\nChoose videos to process:")
        print("1. Process all videos")
        print("2. Process a specific video")
        choice = input("Enter 1 or 2: ").strip()
        if choice in {"1", "2"}:
            break
        print("Invalid choice. Please enter 1 or 2.")
    if choice == "1":
        selected_videos = videos
    else:
        print("\nAvailable videos:")
        for index, video in enumerate(videos, start=1):
            print(f"{index}. {video.name}")
        while True:
            selection = input(f"Enter a video number (1-{len(videos)}): ").strip()
            if selection.isdigit() and 1 <= int(selection) <= len(videos):
                selected_videos = [videos[int(selection) - 1]]
                break
            print(f"Invalid selection. Please enter a number from 1 to {len(videos)}.")
    for input_video in selected_videos:
        output_name = input_video.stem.replace(" ", "_").replace("(", "").replace(")", "")
        output_video = current_dir / "output" / f"annotated_{output_name}.mp4"
        process_video(input_video, output_video, config)

def process_video(input_video, output_video, config):

    print(f"Loading input: {input_video}")
    loader = VideoLoader(input_video)
    meta = loader.get_metadata()
    print(f"Video {meta['width']}x{meta['height']} @ {meta['fps']:.2f} FPS ({meta['total_frames']} frames)")

    detector = BallDetector(
        model_name=config.get("model_name", "yolo11s.pt"),
        conf_threshold=config.get("ball_detection_conf", 0.28),
        iou_threshold=config.get("iou_threshold", 0.45),
        max_aspect_ratio_deviation=config.get("max_aspect_ratio_deviation", 0.75),
        filter_person_parts=True,
        filter_stationary=config.get("suppress_stationary_objects", True),
        stationary_radius_px=config.get("stationary_radius_px", 14.0),
        stationary_frame_persistence=config.get("stationary_frame_persistence", 4),
        roi_box=(0.0, 0.15, 1.0, 1.0),
    )

    visualizer = VideoVisualizer(
        output_path=output_video,
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        trail_length=20,
    )

    print(f"Rendering updated annotations to: {output_video} ...")
    total_detections = 0
    for frame_idx, timestamp_s, frame in loader.frames():
        balls = detector.detect(frame)
        if balls:
            total_detections += len(balls)
        visualizer.write_frame(frame, balls, frame_idx, timestamp_s)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}/{meta['total_frames']}...")

    visualizer.release()
    loader.release()

    print(f"\nDone! Updated video saved to: {output_video}")
    print(f"Total ball frames detected: {total_detections}")


if __name__ == "__main__":
    main()
