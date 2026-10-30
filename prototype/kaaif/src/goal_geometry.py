import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import cv2


class VirtualGoalGeometry:
    """
    Constructs and models the virtual goal from the physical ground markers:
    P_left = (x_L, y_L), P_right = (x_R, y_R)

    Accounts for ground slope, camera skewness, and lens perspective:
    - Calculates baseline width W and skew angle theta
    - Erects virtual goalposts perpendicular to the ground baseline
    - Applies proportional crossbar rule: Height H = a * W
    """

    def __init__(
        self,
        left_marker: Tuple[float, float],
        right_marker: Tuple[float, float],
        ratio_a: float = 0.75,
        left_bbox: Optional[List[int]] = None,
        right_bbox: Optional[List[int]] = None,
    ):
        self.p_left = np.array(left_marker, dtype=np.float32)
        self.p_right = np.array(right_marker, dtype=np.float32)
        self.ratio_a = ratio_a
        self.left_bbox = left_bbox
        self.right_bbox = right_bbox

        # Baseline vector and width W
        self.baseline_vec = self.p_right - self.p_left
        self.width = float(np.linalg.norm(self.baseline_vec))
        if self.width < 1e-4:
            raise ValueError("Left and right markers cannot be at the same point.")

        # Baseline tilt / skew angle relative to image horizontal
        self.skew_angle_deg = float(np.degrees(np.arctan2(self.baseline_vec[1], self.baseline_vec[0])))

        # Unit vector along tilted ground baseline (from left to right)
        self.u_base = self.baseline_vec / self.width

        # Perpendicular upward unit vector (accounting for ground tilt)
        # Vector perpendicular counter-clockwise is (dy, -dx)
        self.u_up = np.array([self.u_base[1], -self.u_base[0]], dtype=np.float32)
        if self.u_up[1] > 0:  # Ensure it always points upwards into the sky (-y)
            self.u_up = -self.u_up

        # Proportional crossbar height
        self.height = self.ratio_a * self.width

        # 4 corners of the virtual goal quad:
        # [Bottom-Left, Bottom-Right, Top-Right, Top-Left]
        self.p_bottom_left = self.p_left
        self.p_bottom_right = self.p_right
        self.p_top_right = self.p_right + self.u_up * self.height
        self.p_top_left = self.p_left + self.u_up * self.height

        # Plane line equation Ax + By + C = 0 for the ground goal-line
        dx = self.p_right[0] - self.p_left[0]
        dy = self.p_right[1] - self.p_left[1]
        self.A = -dy
        self.B = dx
        self.C = dy * self.p_left[0] - dx * self.p_left[1]
        norm = math.hypot(self.A, self.B)
        self.A /= norm
        self.B /= norm
        self.C /= norm

    def get_goal_corners(self) -> np.ndarray:
        return np.array([
            self.p_bottom_left,
            self.p_bottom_right,
            self.p_top_right,
            self.p_top_left,
        ], dtype=np.int32)

    def signed_distance_to_goal_line(self, point: Tuple[float, float]) -> float:
        return float(self.A * point[0] + self.B * point[1] + self.C)

    def project_onto_goal_frame(
        self,
        point: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Projects a point (x, y) into the tilted goal's local coordinate frame:
        Returns:
            t_lateral: Distance in pixels along the goal line from left post (0 to W).
            h_vertical: Height in pixels perpendicular to the ground baseline (0 to H).
        """
        pt = np.array(point, dtype=np.float32)
        v = pt - self.p_bottom_left
        t_lateral = float(np.dot(v, self.u_base))
        h_vertical = float(np.dot(v, self.u_up))
        return t_lateral, h_vertical

    def check_ball_boundaries(
        self,
        center: Tuple[float, float],
        radius: float,
    ) -> Dict[str, Any]:
        t_lat, h_vert = self.project_onto_goal_frame(center)

        inside_lateral = (t_lat - radius >= 0) and (t_lat + radius <= self.width)
        below_crossbar = (h_vert + radius <= self.height)
        above_ground = (h_vert - radius >= -radius * 0.5)

        is_fully_within = inside_lateral and below_crossbar and above_ground

        lateral_margin_left = t_lat - radius
        lateral_margin_right = self.width - (t_lat + radius)
        vertical_margin_crossbar = self.height - (h_vert + radius)

        return {
            "t_lateral": t_lat,
            "h_vertical": h_vert,
            "goal_width": self.width,
            "goal_height": self.height,
            "inside_lateral": inside_lateral,
            "below_crossbar": below_crossbar,
            "above_ground": above_ground,
            "is_fully_within": is_fully_within,
            "min_boundary_margin": min(lateral_margin_left, lateral_margin_right, vertical_margin_crossbar),
        }

    def draw_virtual_goal(
        self,
        frame: np.ndarray,
        alpha: float = 0.18,
        line_thickness: int = 4,
        color_posts: Tuple[int, int, int] = (0, 255, 255),  # Yellow
        color_fill: Tuple[int, int, int] = (0, 200, 255),   # Amber
    ) -> np.ndarray:
        annotated = frame.copy()
        corners = self.get_goal_corners()

        # 1. Highlight detected physical marker blocks (cinder blocks / bricks)
        if self.left_bbox is not None:
            lx1, ly1, lx2, ly2 = [int(v) for v in self.left_bbox]
            cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
            cv2.putText(annotated, "POST L", (lx1, ly1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        if self.right_bbox is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in self.right_bbox]
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(annotated, "POST R", (rx1, ry1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 2. Semi-transparent amber goal plane
        overlay = frame.copy()
        cv2.fillPoly(overlay, [corners], color_fill)
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

        p_bl = tuple(corners[0])
        p_br = tuple(corners[1])
        p_tr = tuple(corners[2])
        p_tl = tuple(corners[3])

        # 3. Ground baseline (tilted to match pitch skew)
        cv2.line(annotated, p_bl, p_br, (255, 255, 255), 3, cv2.LINE_AA)

        # 4. Upright virtual goalposts (perpendicular to tilted ground line)
        cv2.line(annotated, p_bl, p_tl, color_posts, line_thickness, cv2.LINE_AA)
        cv2.line(annotated, p_br, p_tr, color_posts, line_thickness, cv2.LINE_AA)

        # 5. Virtual crossbar
        cv2.line(annotated, p_tl, p_tr, color_posts, line_thickness, cv2.LINE_AA)

        # Marker ground anchor points
        cv2.circle(annotated, p_bl, 7, (0, 0, 255), -1)
        cv2.circle(annotated, p_br, 7, (0, 0, 255), -1)

        # Goal geometry tag
        tag_text = f"VIRTUAL GOAL (W={self.width:.0f}px, H={self.height:.0f}px, Tilt={self.skew_angle_deg:+.1f} deg)"
        cv2.putText(
            annotated,
            tag_text,
            (max(20, p_tl[0] - 80), max(40, p_tl[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color_posts,
            2,
            cv2.LINE_AA,
        )

        return annotated
