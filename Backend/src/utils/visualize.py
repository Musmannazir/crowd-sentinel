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


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic distinct-ish BGR color per track ID."""
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))
