import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class VideoVisualizer:
    """
    Renders visual annotations (bounding boxes, labels, trajectory trails)
    onto frames and writes out an annotated video clip.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        trail_length: int = 15,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.trail_length = trail_length
        self.trail_points: List[tuple] = []

        # Use mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

    def draw_frame(
        self,
        frame: np.ndarray,
        balls: List[Dict[str, Any]],
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> np.ndarray:
        """
        Draws bounding box, center circle, trail, and info text onto frame.
        """
        annotated = frame.copy()

        # Update and draw trajectory trail
        if balls:
            best_ball = max(balls, key=lambda b: b.get("confidence", 0.0))
            cx, cy = int(best_ball["center"][0]), int(best_ball["center"][1])
            self.trail_points.append((cx, cy))
            if len(self.trail_points) > self.trail_length:
                self.trail_points.pop(0)

        # Draw motion trail
        for i in range(1, len(self.trail_points)):
            alpha = i / len(self.trail_points)
            thickness = int(2 + 4 * alpha)
            cv2.line(
                annotated,
                self.trail_points[i - 1],
                self.trail_points[i],
                (0, 255, 255),  # Yellow trail
                thickness,
            )

        # Draw bounding boxes for detected balls
        for b in balls:
            x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
            conf = b.get("confidence", 0.0)
            cx, cy = int(b["center"][0]), int(b["center"][1])
            radius = int(b.get("radius", 0))

            # Bounding box (bright green)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Center point
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            # Ball circle outline
            if radius > 0:
                cv2.circle(annotated, (cx, cy), radius, (255, 200, 0), 1)

            # Confidence label
            label = f"Ball: {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - lh - 8)),
                (x1 + lw + 6, max(lh + 8, y1)),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(lh + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Frame and time watermark
        info_text = ""
        if frame_idx is not None:
            info_text += f"Frame: {frame_idx} "
        if timestamp_s is not None:
            info_text += f"| Time: {timestamp_s:.2f}s "
        if balls:
            info_text += f"| Ball detected ({len(balls)})"
        else:
            info_text += "| No ball"

        cv2.putText(
            annotated,
            info_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated

    def write_frame(
        self,
        frame: np.ndarray,
        balls: List[Dict[str, Any]],
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ):
        annotated = self.draw_frame(frame, balls, frame_idx, timestamp_s)
        self.writer.write(annotated)

    def release(self):
        if self.writer:
            self.writer.release()
