"""Stage 1 — Detection.

Runs YOLOv11-small over each frame and returns person bounding boxes.
Restricted to COCO class 0 ("person") since the pipeline is identity-free
and only needs localization, not classification of object types.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected person in one frame."""

    frame_idx: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class PersonDetector:
    """Wraps Ultralytics YOLOv11-small, filtered to the "person" class."""

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes if classes is not None else [0]  # person
        self.device = device

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """Run detection on a single frame, return a list of Detection."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        frame_idx=frame_idx,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                    )
                )
        return detections

    @classmethod
    def from_config(cls, config: dict) -> "PersonDetector":
        """Build a PersonDetector from the `detection:` block of config.yaml."""
        det_cfg = config["detection"]
        return cls(
            model_path=det_cfg["model"],
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            classes=det_cfg.get("classes", [0]),
            device=det_cfg.get("device", "cpu"),
        )