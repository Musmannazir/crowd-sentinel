"""Converts a dense optical flow field into one motion feature per track.

Stage 6/7 (clustering, anomaly scoring) need "how fast and in what
direction is this person moving", not a full per-pixel flow map — this
module does that reduction: average the flow inside each person's bbox.
"""
from __future__ import annotations

import math

import numpy as np


def track_motion_from_flow(flow: np.ndarray, bbox: tuple[float, float, float, float]) -> dict:
    """Average the dense flow field inside bbox into one motion vector.

    Returns:
        dx, dy:    average pixel displacement per frame (image coords:
                   +x = right, +y = down)
        speed:     magnitude of (dx, dy), in pixels/frame
        angle_deg: direction of motion, 0=right, 90=down, -90=up, 180/-180=left
    """
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox (off-frame or zero-area) — no flow to average.
        return {"dx": 0.0, "dy": 0.0, "speed": 0.0, "angle_deg": 0.0}

    region = flow[y1:y2, x1:x2]
    dx = float(np.mean(region[..., 0]))
    dy = float(np.mean(region[..., 1]))
    speed = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx))

    return {"dx": dx, "dy": dy, "speed": speed, "angle_deg": angle_deg}