"""Drawing helpers for visually inspecting pipeline output."""
from __future__ import annotations

import cv2
import numpy as np


def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draw bounding boxes + track IDs for a list of TrackedObject.

    Deliberately draws only a box + numeric ID — no identity, no face
    crops — consistent with the project's identity-free design.
    """
    annotated = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = _color_for_id(t.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {t.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + confidence score for raw (untracked) Detections.

    Used for Stage 1 standalone testing — deliberately shows confidence
    instead of a track ID, since there is no ID yet at this stage.
    """
    annotated = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = (60, 200, 60)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def draw_motion_arrows(frame: np.ndarray, tracks_with_motion: list[dict]) -> np.ndarray:
    """Draw an arrow per track showing its average motion direction + speed.

    Each dict in tracks_with_motion needs: bbox, track_id, dx, dy, speed.
    Arrow length is exaggerated (ARROW_SCALE) purely for visibility — real
    per-frame pixel displacement is often just a few pixels and invisible
    otherwise. Not to scale; speed is also printed as text for the real number.
    """
    annotated = frame.copy()
    ARROW_SCALE = 8

    for t in tracks_with_motion:
        x1, y1, x2, y2 = map(int, t["bbox"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        end_x = int(cx + t["dx"] * ARROW_SCALE)
        end_y = int(cy + t["dy"] * ARROW_SCALE)

        color = (0, 140, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.arrowedLine(annotated, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
        label = f"ID{t['track_id']} {t['speed']:.1f}px/f"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return annotated


# COCO 17-keypoint skeleton edges (0-indexed, matches KEYPOINT_NAMES order
# in src/pose/pose_estimator.py).
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


def draw_poses(
    frame: np.ndarray, tracks_with_pose: list[dict], kp_conf_threshold: float = 0.5
) -> np.ndarray:
    """Draw skeleton lines + track ID for each pose-matched track.

    Each dict in tracks_with_pose needs: track_id, bbox, keypoints
    (list of (x, y, conf) tuples, COCO 17-keypoint order). Only draws an
    edge/point if its keypoint confidence clears kp_conf_threshold, so
    occluded/unsure joints don't draw nonsense lines across the frame.
    """
    annotated = frame.copy()
    for t in tracks_with_pose:
        color = _color_for_id(t["track_id"])
        kpts = t["keypoints"]

        for i, j in COCO_SKELETON:
            if i >= len(kpts) or j >= len(kpts):
                continue
            x1, y1, c1 = kpts[i]
            x2, y2, c2 = kpts[j]
            if c1 < kp_conf_threshold or c2 < kp_conf_threshold:
                continue
            cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        for x, y, c in kpts:
            if c >= kp_conf_threshold:
                cv2.circle(annotated, (int(x), int(y)), 3, color, -1)

        x1, y1, x2, y2 = map(int, t["bbox"])
        label = f"ID {t['track_id']}"
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return annotated


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct-ish BGR color per track ID."""
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))