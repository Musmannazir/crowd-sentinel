"""Stage 4 — Pose Estimation.

Runs YOLO11-Pose over each frame to extract 17 COCO keypoints per detected
person. Produces its own boxes/keypoints independently of Stage 2's
tracks — pose_matching.py links each pose detection back to a Stage 2
track_id via IoU, so Stage 2 (ByteTrack) stays the single source of truth
for track IDs across the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

# COCO 17-keypoint order, as returned by Ultralytics pose models.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class PoseDetection:
    """A single person's pose in one frame (not yet linked to a track_id)."""

    frame_idx: int
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, conf) x 17


class PoseEstimator:
    """Wraps Ultralytics YOLO11-Pose."""

    def __init__(
        self,
        model_path: str = "yolo11s-pose.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def estimate(self, frame: np.ndarray, frame_idx: int = 0) -> list[PoseDetection]:
        """Run pose estimation on a single frame, return a list of PoseDetection."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[PoseDetection] = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes = result.boxes
            kpts = result.keypoints  # .xy: (N, 17, 2), .conf: (N, 17)
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                kp_xy = kpts.xy[i].tolist()
                kp_conf = kpts.conf[i].tolist() if kpts.conf is not None else [1.0] * len(kp_xy)
                keypoints = [(kp_xy[j][0], kp_xy[j][1], kp_conf[j]) for j in range(len(kp_xy))]
                detections.append(
                    PoseDetection(
                        frame_idx=frame_idx,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=conf,
                        keypoints=keypoints,
                    )
                )
        return detections

    @classmethod
    def from_config(cls, config: dict) -> "PoseEstimator":
        """Build a PoseEstimator from the `pose:` block of config.yaml."""
        pose_cfg = config["pose"]
        return cls(
            model_path=pose_cfg["model"],
            conf_threshold=pose_cfg.get("conf_threshold", 0.35),
            iou_threshold=pose_cfg.get("iou_threshold", 0.45),
            device=pose_cfg.get("device", "cpu"),
        )