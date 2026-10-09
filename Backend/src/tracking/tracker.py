"""Stage 2 — Tracking.

Assigns a persistent track_id to each detected person across frames using
ByteTrack (via Ultralytics' built-in `.track()` API — no raw identity
features are stored, only a per-video integer ID that resets each run).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedObject:
    """A single tracked person in one frame."""

    frame_idx: int
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class PersonTracker:
    """Wraps Ultralytics YOLOv11 + ByteTrack for persistent person tracking.

    Note: detection and tracking are run together here (Ultralytics fuses
    them via `.track()`), but kept as a separate pipeline stage/module so
    Stage 1 (detector.py) can still be used standalone, and so the tracker
    can later be swapped (e.g. for DeepSORT) without touching detection.
    """

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        tracker_cfg: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.tracker_cfg = tracker_cfg
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0]
        self.device = device

    def track(self, frame: np.ndarray, frame_idx: int = 0) -> list[TrackedObject]:
        """Run detection+tracking on a single frame, return TrackedObjects.

        Must be called frame-by-frame, in order, on the same instance —
        ByteTrack keeps internal state (track buffer) between calls.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,  # keep track state across calls
            verbose=False,
        )

        tracked: list[TrackedObject] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue  # no confirmed tracks yet this frame
            ids = result.boxes.id.tolist()
            for box, track_id in zip(result.boxes, ids):
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                tracked.append(
                    TrackedObject(
                        frame_idx=frame_idx,
                        track_id=int(track_id),
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                    )
                )
        return tracked

    @classmethod
    def from_config(cls, config: dict) -> "PersonTracker":
        """Build a PersonTracker from `detection:` + `tracking:` config blocks."""
        det_cfg = config["detection"]
        trk_cfg = config["tracking"]
        return cls(
            model_path=det_cfg["model"],
            tracker_cfg=trk_cfg["tracker"],
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            classes=det_cfg.get("classes", [0]),
            device=det_cfg.get("device", "cpu"),
        )