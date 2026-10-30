import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from .tracker import TrackState
from .goal_geometry import VirtualGoalGeometry


class VideoVisualizer:
    """
    Renders visual annotations:
    - Virtual Goalposts, Proportional Crossbar (H = a * W), and semi-transparent goal plane
    - Bounding boxes & confidence
    - Differentiated trajectory trails (Green = OBSERVED, Orange = PREDICTED)
    - Boundary metrics & State flags
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        trail_length: int = 25,
        virtual_goal: Optional[VirtualGoalGeometry] = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.trail_length = trail_length
        self.virtual_goal = virtual_goal
        self.trail_points: List[Dict[str, Any]] = []

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
        tracked_ball: Optional[Dict[str, Any]] = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> np.ndarray:
        annotated = frame.copy()

        # 1. Render Virtual Goal first (as background/ground element)
        if self.virtual_goal is not None:
            annotated = self.virtual_goal.draw_virtual_goal(annotated)

        # 2. Update and draw trajectory trail
        if tracked_ball is not None:
            cx, cy = int(tracked_ball["center"][0]), int(tracked_ball["center"][1])
            self.trail_points.append({
                "pos": (cx, cy),
                "state": tracked_ball["state"],
            })
            if len(self.trail_points) > self.trail_length:
                self.trail_points.pop(0)

        for i in range(1, len(self.trail_points)):
            p1 = self.trail_points[i - 1]["pos"]
            p2 = self.trail_points[i]["pos"]
            st = self.trail_points[i]["state"]

            if st == TrackState.OBSERVED:
                cv2.line(annotated, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.line(annotated, p1, p2, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(annotated, p2, 2, (0, 140, 255), -1)

        # 3. Draw tracked ball & boundary status
        boundary_info = None
        if tracked_ball is not None:
            cx, cy = int(tracked_ball["center"][0]), int(tracked_ball["center"][1])
            radius = int(tracked_ball.get("radius", 15))
            state = tracked_ball["state"]
            conf = tracked_ball.get("confidence", 0.0)

            if state == TrackState.OBSERVED:
                box_color = (0, 255, 0)
                state_label = f"BALL ({conf:.2f})"
            else:
                box_color = (0, 140, 255)
                state_label = "PREDICTED (Kalman)"

            # Ball outline and center
            cv2.circle(annotated, (cx, cy), radius, box_color, 2)
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            # Check position relative to virtual goal
            if self.virtual_goal is not None:
                boundary_info = self.virtual_goal.check_ball_boundaries((cx, cy), radius)

            # Label tag
            (lw, lh), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (cx - radius, max(0, cy - radius - lh - 6)), (cx - radius + lw + 6, cy - radius), box_color, -1)
            cv2.putText(
                annotated,
                state_label,
                (cx - radius + 3, max(lh, cy - radius - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # 4. Watermark information
        info_lines = []
        if frame_idx is not None and timestamp_s is not None:
            info_lines.append(f"Frame: {frame_idx:03d} | Time: {timestamp_s:.2f}s")

        if self.virtual_goal is not None:
            info_lines.append(
                f"Virtual Goal: Width={self.virtual_goal.width:.0f}px | Height={self.virtual_goal.height:.0f}px ({self.virtual_goal.ratio_a}*W)"
            )

        if boundary_info is not None:
            status_str = "INSIDE VIRTUAL GOAL" if boundary_info["is_fully_within"] else "OUTSIDE GOAL BOUNDARY"
            info_lines.append(
                f"Position: {status_str} | Lateral: {boundary_info['t_lateral']:.0f}px | Height: {boundary_info['h_vertical']:.0f}px"
            )

        for idx, line in enumerate(info_lines):
            cv2.putText(
                annotated,
                line,
                (20, 35 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def write_frame(
        self,
        frame: np.ndarray,
        tracked_ball: Optional[Dict[str, Any]] = None,
        raw_detections: Optional[List[Dict[str, Any]]] = None,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ):
        annotated = self.draw_frame(frame, tracked_ball, raw_detections, frame_idx, timestamp_s)
        self.writer.write(annotated)

    def release(self):
        if self.writer:
            self.writer.release()
