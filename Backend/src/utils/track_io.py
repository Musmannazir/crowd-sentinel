"""Load Stage 2 (tracking) JSON output for use by later pipeline stages."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_tracks_by_frame(tracks_json_path: str | Path) -> dict[int, list[dict]]:
    """Return {frame_idx: [{track_id, bbox, confidence}, ...]} from Stage 2 JSON.

    Grouping by frame_idx up front means every downstream stage (optical
    flow, pose, appearance) can just do tracks_by_frame.get(frame_idx, [])
    instead of re-scanning the full list per frame.
    """
    with open(tracks_json_path, "r") as f:
        raw = json.load(f)

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for entry in raw:
        by_frame[entry["frame_idx"]].append(entry)
    return dict(by_frame)