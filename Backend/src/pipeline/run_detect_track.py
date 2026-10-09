"""Entry point for Stage 1 + Stage 2: Detection + Tracking.

Usage:
    python -m src.pipeline.run_detect_track --video data/sample_videos/sample1.mp4

Reads a video, runs YOLOv8-small + ByteTrack frame-by-frame, writes an
annotated output video to outputs/results/, and saves raw per-frame
track data as JSON to outputs/results/ for downstream stages to consume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.tracking.tracker import PersonTracker
from src.utils.config import PROJECT_ROOT, get_logger, load_config
from src.utils.video_io import VideoReader, VideoWriter
from src.utils.visualize import draw_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking on a video.")
    parser.add_argument(
        "--video", type=str, required=True,
        help="Path to input video (e.g. data/sample_videos/sample1.mp4)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (defaults to configs/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    logger = get_logger("detect_track", log_dir=PROJECT_ROOT / config["paths"]["logs_dir"])

    video_path = Path(args.video)
    out_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = PersonTracker.from_config(config)
    logger.info(f"Loaded detector+tracker (model={config['detection']['model']}, "
                f"tracker={config['tracking']['tracker']}, device={config['detection']['device']})")

    all_tracks_json = []

    with VideoReader(video_path) as reader:
        logger.info(
            f"Opened {video_path.name}: {reader.frame_count} frames, "
            f"{reader.width}x{reader.height} @ {reader.fps:.1f} fps"
        )
        out_video_path = out_dir / f"{video_path.stem}_tracked.mp4"
        with VideoWriter(out_video_path, reader.fps, reader.width, reader.height) as writer:
            for frame_idx, frame in tqdm(reader.frames(), total=reader.frame_count, desc="Tracking"):
                tracks = tracker.track(frame, frame_idx=frame_idx)
                annotated = draw_tracks(frame, tracks)
                writer.write(annotated)

                for t in tracks:
                    all_tracks_json.append({
                        "frame_idx": t.frame_idx,
                        "track_id": t.track_id,
                        "bbox": list(t.bbox),
                        "confidence": t.confidence,
                    })

    out_json_path = out_dir / f"{video_path.stem}_tracks.json"
    with open(out_json_path, "w") as f:
        json.dump(all_tracks_json, f, indent=2)

    unique_ids = {t["track_id"] for t in all_tracks_json}
    logger.info(f"Done. {len(unique_ids)} unique tracks across {len(all_tracks_json)} detections.")
    logger.info(f"Annotated video: {out_video_path}")
    logger.info(f"Track data JSON: {out_json_path}")


if __name__ == "__main__":
    main()
