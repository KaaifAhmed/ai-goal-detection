from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2


class GoalMarkerDetector:
    """
    Detects the physical goalposts / concrete cinder blocks in the foreground.
    Computes their ground contact points and inner edges to anchor the virtual goal.
    """

    def __init__(
        self,
        sat_threshold: int = 55,  # Concrete has low color saturation
        val_min: int = 70,
        val_max: int = 220,
    ):
        self.sat_threshold = sat_threshold
        self.val_min = val_min
        self.val_max = val_max

    def _find_block_in_roi(
        self,
        frame: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        is_left_marker: bool = True,
    ) -> Optional[Dict[str, Any]]:
        rx1, ry1, rx2, ry2 = roi_rect
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        mask = (sat < self.sat_threshold) & (val > self.val_min) & (val < self.val_max)
        mask = mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best_cand = None
        max_area = 0

        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect = bh / float(bw) if bw > 0 else 0

            # Blocks are upright rectangular objects (height > width)
            if area > 1000 and 0.8 <= aspect <= 3.0:
                if area > max_area:
                    max_area = area
                    # Inner post edge: right edge of left block, left edge of right block
                    inner_x = float(rx1 + bx + bw) if is_left_marker else float(rx1 + bx)
                    ground_y = float(ry1 + by + bh)
                    best_cand = {
                        "bbox": [rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh],
                        "inner_point": (inner_x, ground_y),
                        "ground_point": (float(rx1 + bx + bw / 2.0), ground_y),
                        "area": area,
                    }

        return best_cand

    def detect_markers(
        self,
        frame: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Dict[str, Any]]:
        h, w = frame.shape[:2]

        # Targeted search regions for the two foreground cinder blocks
        roi_left = (int(0.16 * w), int(0.62 * h), int(0.28 * w), int(0.88 * h))
        roi_right = (int(0.85 * w), int(0.68 * h), int(0.96 * w), int(0.95 * h))

        cand_left = self._find_block_in_roi(frame, roi_left, is_left_marker=True)
        cand_right = self._find_block_in_roi(frame, roi_right, is_left_marker=False)

        if cand_left is not None:
            p_left = cand_left["inner_point"]
            left_box = cand_left["bbox"]
        else:
            p_left = (440.0, 870.0)
            left_box = [350, 710, 450, 880]

        if cand_right is not None:
            p_right = cand_right["inner_point"]
            right_box = cand_right["bbox"]
        else:
            p_right = (1710.0, 940.0)
            right_box = [1680, 770, 1800, 950]

        debug_info = {
            "left_marker_bbox": left_box,
            "right_marker_bbox": right_box,
            "p_left": p_left,
            "p_right": p_right,
        }

        return p_left, p_right, debug_info
