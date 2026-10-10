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


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct-ish BGR color per track ID."""
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))