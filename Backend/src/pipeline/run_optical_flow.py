"""Stage 3 — Optical Flow, run against an already-tracked video.

Requires Stage 2 to have been run first on the same video — this reads
outputs/results/<video_stem>_tracks.json for per-frame person boxes.

For each consecutive frame pair: compute dense RAFT-small flow, then
average the flow inside each tracked person's bbox to get one motion
vector (speed + direction) per track per frame. That per-track feature —
not the dense flow map — is what Stage 6/7 will consume later.

Usage:
    python -m src.pipeline.run_optical_flow --video data/sample_videos/sample1.mp4

    # Quick speed/quality check on the first 60 frame-pairs before committing
    # to a full run (RAFT on CPU is the slowest stage so far):
    python -m src.pipeline.run_optical_flow --video data/sample_videos/sample1.mp4 --max_frames 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.optical_flow.flow_estimator import OpticalFlowEstimator
from src.optical_flow.motion_features import track_motion_from_flow
from src.utils.config import PROJECT_ROOT, get_logger, load_config
from src.utils.track_io import load_tracks_by_frame
from src.utils.video_io import VideoReader, VideoWriter
from src.utils.visualize import draw_motion_arrows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 optical flow on a tracked video.")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Process only the first N frame-pairs — use this first to check "
             "speed/quality before committing to a full-video run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    logger = get_logger("optical_flow", log_dir=PROJECT_ROOT / config["paths"]["logs_dir"])

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

    estimator = OpticalFlowEstimator.from_config(config)
    of_cfg = config["optical_flow"]
    logger.info(f"Loaded RAFT-small (device={of_cfg.get('device', 'cpu')}, "
                f"resize_short_side={of_cfg.get('resize_short_side', 240)})")

    motion_records: list[dict] = []
    per_pair_times: list[float] = []

    with VideoReader(video_path) as reader:
        total_pairs = reader.frame_count - 1
        if args.max_frames is not None:
            total_pairs = min(total_pairs, args.max_frames)

        logger.info(f"Processing {total_pairs} frame pairs"
                    f"{' (LIMITED test run)' if args.max_frames is not None else ''}")

        out_video_path = out_dir / f"{video_path.stem}_motion.mp4"
        with VideoWriter(out_video_path, reader.fps, reader.width, reader.height) as writer:
            pbar = tqdm(total=total_pairs, desc="Optical flow")
            prev_idx, prev_frame = None, None
            pairs_done = 0

            for idx, frame in reader.frames():
                if prev_frame is not None and pairs_done < total_pairs:
                    t0 = time.time()
                    flow = estimator.estimate_flow(prev_frame, frame)
                    per_pair_times.append(time.time() - t0)

                    frame_combined = []
                    for t in tracks_by_frame.get(prev_idx, []):
                        motion = track_motion_from_flow(flow, tuple(t["bbox"]))
                        record = {
                            "frame_idx": prev_idx,
                            "track_id": t["track_id"],
                            "bbox": t["bbox"],
                            **motion,
                        }
                        motion_records.append(record)
                        frame_combined.append(record)

                    annotated = draw_motion_arrows(prev_frame, frame_combined)
                    writer.write(annotated)

                    pairs_done += 1
                    pbar.update(1)

                prev_idx, prev_frame = idx, frame
                if pairs_done >= total_pairs:
                    break
            pbar.close()

        full_video_frame_count = reader.frame_count

    out_json_path = out_dir / f"{video_path.stem}_motion.json"
    with open(out_json_path, "w") as f:
        json.dump(motion_records, f, indent=2)

    times = np.array(per_pair_times) if per_pair_times else np.array([0.0])
    speeds = np.array([m["speed"] for m in motion_records]) if motion_records else np.array([0.0])

    logger.info("=" * 60)
    logger.info("STAGE 3 OPTICAL FLOW SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Frame pairs processed:          {pairs_done}")
    logger.info(f"Avg time / frame pair:           {times.mean():.2f}s "
                f"(~{(1 / times.mean() if times.mean() > 0 else 0):.2f} pairs/sec)")
    logger.info(f"Est. time for FULL video:        "
                f"{times.mean() * (full_video_frame_count - 1) / 60:.1f} min at this speed")
    logger.info(f"Per-track motion records:        {len(motion_records)}")
    logger.info(f"Speed (px/frame) mean/min/max:   "
                f"{speeds.mean():.2f} / {speeds.min():.2f} / {speeds.max():.2f}")
    logger.info("=" * 60)
    logger.info(f"Motion data JSON:     {out_json_path}")
    logger.info(f"Annotated video:      {out_video_path}")
    if args.max_frames is not None:
        logger.info("NOTE: limited test run. Check the 'Est. time for FULL video' line above "
                     "before deciding whether to re-run without --max_frames.")


if __name__ == "__main__":
    main()