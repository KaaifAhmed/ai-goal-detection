import cv2
from pathlib import Path
from typing import Generator, Tuple, Dict, Any, Optional


class VideoLoader:
    """Ingests videos and yields frames with monotonic timestamps."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_s": self.total_frames / self.fps if self.fps > 0 else 0.0,
        }

    def frames(self) -> Generator[Tuple[int, float, Any], None, None]:
        """
        Yields:
            frame_idx (int): 0-indexed frame number.
            timestamp_s (float): Monotonic timestamp in seconds.
            frame (np.ndarray): BGR image frame.
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            # Fallback to frame_idx / fps if POS_MSEC is unreliable
            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = (pos_msec / 1000.0) if pos_msec > 0 else (frame_idx / self.fps)

            yield frame_idx, timestamp_s, frame
            frame_idx += 1

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
