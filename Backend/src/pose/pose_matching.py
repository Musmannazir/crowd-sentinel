"""Links Stage 4 pose detections back to Stage 2 track IDs via IoU.

Stage 2 (ByteTrack) is the single source of truth for track_id across the
whole pipeline. The pose model produces its own boxes each frame; this
module matches each pose box to the closest Stage 2 track box (by IoU) so
downstream stages can key pose data by the same track_id as everything else.
"""
from __future__ import annotations

from src.pose.pose_estimator import PoseDetection


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_poses_to_tracks(
    pose_detections: list[PoseDetection],
    tracks_this_frame: list[dict],
    iou_threshold: float = 0.3,
) -> list[dict]:
    """Greedy best-IoU match between pose detections and Stage 2 tracks.

    Returns a list of dicts: {track_id, bbox, confidence, keypoints} — one
    per pose detection that found a track match above iou_threshold. Pose
    detections with no good match (e.g. someone Stage 2 missed this frame,
    or a stray pose false-positive) are dropped — they'd have no track_id
    to key downstream features by anyway.
    """
    matched: list[dict] = []
    used_track_ids: set[int] = set()

    # Most confident pose detections get first pick of their best track match.
    for pose in sorted(pose_detections, key=lambda p: p.confidence, reverse=True):
        best_iou = 0.0
        best_track = None
        for t in tracks_this_frame:
            if t["track_id"] in used_track_ids:
                continue
            iou = _iou(pose.bbox, tuple(t["bbox"]))
            if iou > best_iou:
                best_iou = iou
                best_track = t

        if best_track is not None and best_iou >= iou_threshold:
            used_track_ids.add(best_track["track_id"])
            matched.append({
                "track_id": best_track["track_id"],
                "bbox": list(pose.bbox),
                "confidence": pose.confidence,
                "keypoints": pose.keypoints,
            })

    return matched