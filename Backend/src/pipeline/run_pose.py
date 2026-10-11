"""Stage 4 — Pose Estimation, run against an already-tracked video.

Requires Stage 2 to have been run first on the same video — reads
outputs/results/<video_stem>_tracks.json for per-frame person boxes, so
each pose's keypoints can be linked to the same track_id used by every
other stage (via IoU matching, see src/pose/pose_matching.py).

Usage:
    python -m src.pipeline.run_pose --video data/sample_videos/sample1.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.pose.pose_estimator import PoseEstimator
from src.pose.pose_matching import match_poses_to_tracks
from src.utils.config import PROJECT_ROOT, get_logger, load_config
from src.utils.track_io import load_tracks_by_frame
from src.utils.video_io import VideoReader, VideoWriter
from src.utils.visualize import draw_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4 pose estimation on a tracked video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    logger = get_logger("pose", log_dir=PROJECT_ROOT / config["paths"]["logs_dir"])

    video_path = Path(args.video)
    out_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    tracks_json_path = out_dir / f"{video_path.stem}_tracks.json"

    if not tracks_json_path.exists():
        raise FileNotFoundError(
            f"No Stage 2 output found at {tracks_json_path}. "
            f"Run `python -m src.pipeline.run_detect_track --video {args.video}` first."
        )

    tracks_by_frame = load_tracks_by_frame(tracks_json_path)
    logger.info(f"Loaded tracks for {len(tracks_by_frame)} frames from {tracks_json_path.name}")

    estimator = PoseEstimator.from_config(config)
    pose_cfg = config["pose"]
    match_iou_threshold = pose_cfg.get("match_iou_threshold", 0.3)
    logger.info(f"Loaded YOLO11-Pose (model={pose_cfg['model']}, device={pose_cfg.get('device', 'cpu')}, "
                f"match_iou_threshold={match_iou_threshold})")

    pose_records: list[dict] = []
    tracks_total = 0
    tracks_matched = 0
    frames_zero_pose = 0

    with VideoReader(video_path) as reader:
        logger.info(f"Opened {video_path.name}: {reader.frame_count} frames, "
                    f"{reader.width}x{reader.height} @ {reader.fps:.1f} fps")

        out_video_path = out_dir / f"{video_path.stem}_pose.mp4"
        with VideoWriter(out_video_path, reader.fps, reader.width, reader.height) as writer:
            for frame_idx, frame in tqdm(reader.frames(), total=reader.frame_count, desc="Pose"):
                pose_detections = estimator.estimate(frame, frame_idx=frame_idx)
                if not pose_detections:
                    frames_zero_pose += 1

                tracks_this_frame = tracks_by_frame.get(frame_idx, [])
                tracks_total += len(tracks_this_frame)

                matched = match_poses_to_tracks(
                    pose_detections, tracks_this_frame, iou_threshold=match_iou_threshold
                )
                tracks_matched += len(matched)

                for m in matched:
                    pose_records.append({"frame_idx": frame_idx, **m})

                annotated = draw_poses(frame, matched)
                writer.write(annotated)

    out_json_path = out_dir / f"{video_path.stem}_poses.json"
    with open(out_json_path, "w") as f:
        json.dump(pose_records, f, indent=2)

    match_rate = (tracks_matched / tracks_total * 100) if tracks_total > 0 else 0.0
    avg_kp_conf = (
        float(np.mean([c for r in pose_records for (_, _, c) in r["keypoints"]]))
        if pose_records else 0.0
    )

    logger.info("=" * 60)
    logger.info("STAGE 4 POSE ESTIMATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Stage 2 tracks (all frames):   {tracks_total}")
    logger.info(f"Tracks matched to a pose:             {tracks_matched} ({match_rate:.1f}%)")
    logger.info(f"Frames with ZERO poses detected:      {frames_zero_pose}")
    logger.info(f"Avg keypoint confidence:              {avg_kp_conf:.3f}")
    logger.info("=" * 60)
    logger.info(f"Pose data JSON:    {out_json_path}")
    logger.info(f"Annotated video:   {out_video_path}")
    logger.info("Match rate < ~70% usually means match_iou_threshold is too strict in "
                "configs/config.yaml, or pose conf_threshold is filtering valid detections.")


if __name__ == "__main__":
    main()