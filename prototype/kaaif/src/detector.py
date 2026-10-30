from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BallDetector:
    """
    Detects footballs with false-positive suppression:
    1. Rejects stationary clutter (e.g., slippers, shoes, stationary debris).
    2. Suppresses upper-body/shoulder detections on players.
    3. Permits motion-blur elongation for fast balls approaching the goal.
    """

    def __init__(
        self,
        model_name: str = "yolo26x.pt",
        conf_threshold: float = 0.28,
        iou_threshold: float = 0.45,
        roi_box: Optional[Tuple[float, float, float, float]] = None,
        min_ball_px: float = 8.0,
        max_ball_px: float = 350.0,
        max_aspect_ratio_deviation: float = 0.75,
        filter_person_parts: bool = True,
        filter_stationary: bool = True,
        stationary_radius_px: float = 14.0,
        stationary_frame_persistence: int = 4,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.roi_box = roi_box or (0.0, 0.15, 1.0, 1.0)
        self.min_ball_px = min_ball_px
        self.max_ball_px = max_ball_px
        self.max_aspect_ratio_deviation = max_aspect_ratio_deviation
        self.filter_person_parts = filter_person_parts
        self.filter_stationary = filter_stationary
        self.stationary_radius_px = stationary_radius_px
        self.stationary_frame_persistence = stationary_frame_persistence

        # Memory for stationary candidate filtering: list of dicts {center: [x,y], count: int}
        self.stationary_anchors: List[Dict[str, Any]] = []

    def _update_stationary_and_check(self, center: Tuple[float, float]) -> bool:
        """
        Returns True if the detection is stationary (e.g. a slipper on ground),
        meaning it has stayed within stationary_radius_px across persistence frames.
        """
        cx, cy = center
        matched = False
        is_stationary = False

        for anchor in self.stationary_anchors:
            ax, ay = anchor["center"]
            dist = np.hypot(cx - ax, cy - ay)
            if dist <= self.stationary_radius_px:
                # Update anchor moving average and increment count
                anchor["center"] = [0.8 * ax + 0.2 * cx, 0.8 * ay + 0.2 * cy]
                anchor["count"] += 1
                anchor["last_seen_age"] = 0
                matched = True
                if anchor["count"] >= self.stationary_frame_persistence:
                    is_stationary = True
                break

        if not matched:
            self.stationary_anchors.append({
                "center": [cx, cy],
                "count": 1,
                "last_seen_age": 0,
            })

        return is_stationary

    def _prune_old_anchors(self):
        for a in self.stationary_anchors:
            a["last_seen_age"] += 1
        # Drop anchors not seen in 30 frames
        self.stationary_anchors = [
            a for a in self.stationary_anchors if a["last_seen_age"] < 30
        ]

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects balls while filtering out slippers, shoes, and clothing artifacts.
        """
        h, w = frame.shape[:2]
        rx1 = int(self.roi_box[0] * w)
        ry1 = int(self.roi_box[1] * h)
        rx2 = int(self.roi_box[2] * w)
        ry2 = int(self.roi_box[3] * h)

        roi_crop = frame[ry1:ry2, rx1:rx2]
        if roi_crop.size == 0:
            return []

        # Predict classes 0 (person) and 32 (sports ball)
        classes_to_detect = [0, 32] if self.filter_person_parts else [32]
        results = self.model.predict(
            source=roi_crop,
            conf=min(self.conf_threshold, 0.25),  # Lower threshold internally to inspect blur candidates
            iou=self.iou_threshold,
            classes=classes_to_detect,
            verbose=False,
        )

        persons = []
        raw_ball_candidates = []

        for r in results:
            if not r.boxes:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                x1 = float(box[0] + rx1)
                y1 = float(box[1] + ry1)
                x2 = float(box[2] + rx1)
                y2 = float(box[3] + ry1)

                if cls_id == 0:  # person
                    persons.append((x1, y1, x2, y2))
                elif cls_id == 32:  # sports ball
                    raw_ball_candidates.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                    })

        self._prune_old_anchors()

        valid_detections = []
        for cand in raw_ball_candidates:
            x1, y1, x2, y2 = cand["bbox"]
            conf = cand["conf"]
            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            dim = max(bw, bh)

            # 1. Size bounds
            if dim < self.min_ball_px or dim > self.max_ball_px:
                continue

            # 2. Aspect ratio (accommodates motion blur elongation along flight path)
            aspect_ratio = bw / bh if bh > 0 else 999.0
            if abs(aspect_ratio - 1.0) > self.max_aspect_ratio_deviation:
                continue

            # 3. Person shoulder / upper torso false-positive suppression
            is_on_person = False
            if self.filter_person_parts:
                for px1, py1, px2, py2 in persons:
                    # Check if candidate is inside the person's upper 60% (shoulders / chest / head)
                    pw = px2 - px1
                    ph = py2 - py1
                    upper_y2 = py1 + 0.60 * ph
                    if (px1 - 5 <= cx <= px2 + 5) and (py1 <= cy <= upper_y2):
                        is_on_person = True
                        break
            if is_on_person:
                continue

            # 4. Stationary clutter filter (e.g. slippers, shoes lying on ground)
            if self.filter_stationary:
                is_stationary = self._update_stationary_and_check((cx, cy))
                if is_stationary:
                    continue

            # Apply final confidence gate
            if conf < self.conf_threshold:
                continue

            valid_detections.append({
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "radius": float(dim / 2.0),
                "confidence": conf,
            })

        return valid_detections
