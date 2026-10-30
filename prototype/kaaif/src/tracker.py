from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class TrackState(str, Enum):
    OBSERVED = "OBSERVED"    # Direct visual detection present
    PREDICTED = "PREDICTED"  # Visual detection missing; state estimated via kinematic prediction


class KalmanBallTracker:
    """
    2D Constant-Velocity Kalman Filter for tracking a football:
    State vector: x = [u, v, u_dot, v_dot]^T
    Measurement:  z = [u, v]^T
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 15.0,
        measurement_noise_std: float = 3.0,
    ):
        self.dt = dt
        # State: [u, v, du, dv]
        self.x = np.zeros((4, 1), dtype=np.float32)

        # State transition matrix F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance Q
        q = process_noise_std ** 2
        self.Q = np.array([
            [0.25 * (dt ** 4) * q, 0, 0.5 * (dt ** 3) * q, 0],
            [0, 0.25 * (dt ** 4) * q, 0, 0.5 * (dt ** 3) * q],
            [0.5 * (dt ** 3) * q, 0, (dt ** 2) * q, 0],
            [0, 0.5 * (dt ** 3) * q, 0, (dt ** 2) * q],
        ], dtype=np.float32)

        # Measurement noise covariance R
        r = measurement_noise_std ** 2
        self.R = np.eye(2, dtype=np.float32) * r

        # Estimation error covariance P
        self.P = np.eye(4, dtype=np.float32) * 50.0

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_u: float, z_v: float):
        z = np.array([[z_u], [z_v]], dtype=np.float32)
        y = z - self.H @ self.x  # Innovation / residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


class BallTrajectoryTracker:
    """
    Manages football track association, Kalman filtering, gap-filling,
    and maintains OBSERVED vs PREDICTED states per frame.
    """

    def __init__(
        self,
        max_prediction_gap: int = 5,
        max_association_distance: float = 120.0,
        min_hits_to_confirm: int = 2,
    ):
        self.max_prediction_gap = max_prediction_gap
        self.max_association_distance = max_association_distance
        self.min_hits_to_confirm = min_hits_to_confirm

        self.kalman: Optional[KalmanBallTracker] = None
        self.last_radius: float = 15.0
        self.consecutive_misses: int = 0
        self.hits: int = 0
        self.history: List[Dict[str, Any]] = []

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Updates the track with detections from current frame.

        Returns:
            Dict representing the active tracked ball state, or None if no active track.
            {
                "frame_idx": int,
                "timestamp_s": float,
                "center": (u, v),
                "radius": float,
                "velocity": (du, dv),
                "state": TrackState.OBSERVED or TrackState.PREDICTED,
                "confidence": float,
                "can_trigger_goal": bool,  # False if purely predicted
            }
        """
        best_detection = None
        matched = False

        if self.kalman is not None:
            # 1. Predict next position
            pred_u, pred_v = self.kalman.predict()

            # 2. Match with closest detection within distance threshold
            best_dist = float("inf")
            for det in detections:
                cu, cv = det["center"]
                dist = np.hypot(cu - pred_u, cv - pred_v)
                if dist < self.max_association_distance and dist < best_dist:
                    best_dist = dist
                    best_detection = det

            if best_detection is not None:
                # Matched with visual detection -> OBSERVED
                cu, cv = best_detection["center"]
                self.kalman.update(cu, cv)
                self.last_radius = best_detection.get("radius", self.last_radius)
                self.consecutive_misses = 0
                self.hits += 1
                matched = True

                tracked_entry = {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "center": self.kalman.position,
                    "radius": self.last_radius,
                    "velocity": self.kalman.velocity,
                    "state": TrackState.OBSERVED,
                    "confidence": best_detection.get("confidence", 1.0),
                    "can_trigger_goal": True,
                }
            else:
                # Detection missing -> PREDICTED (fill short gap)
                self.consecutive_misses += 1
                if self.consecutive_misses <= self.max_prediction_gap:
                    tracked_entry = {
                        "frame_idx": frame_idx,
                        "timestamp_s": timestamp_s,
                        "center": (pred_u, pred_v),
                        "radius": self.last_radius,
                        "velocity": self.kalman.velocity,
                        "state": TrackState.PREDICTED,
                        "confidence": 0.0,
                        "can_trigger_goal": False,  # Safety check: pure prediction cannot trigger a goal
                    }
                else:
                    # Lost track beyond gap limit
                    self.kalman = None
                    self.hits = 0
                    return None
        else:
            # Track not initialized yet: initialize from highest confidence detection
            if detections:
                best_det = max(detections, key=lambda d: d.get("confidence", 0.0))
                cu, cv = best_det["center"]
                self.kalman = KalmanBallTracker()
                self.kalman.x[0, 0] = cu
                self.kalman.x[1, 0] = cv
                self.last_radius = best_det.get("radius", 15.0)
                self.consecutive_misses = 0
                self.hits = 1

                tracked_entry = {
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "center": (cu, cv),
                    "radius": self.last_radius,
                    "velocity": (0.0, 0.0),
                    "state": TrackState.OBSERVED,
                    "confidence": best_det.get("confidence", 1.0),
                    "can_trigger_goal": True,
                }
            else:
                return None

        self.history.append(tracked_entry)
        return tracked_entry

    def get_full_trajectory(self) -> List[Dict[str, Any]]:
        return self.history
