
```markdown
# Crowd Outlier Detection System

A privacy-preserving, real-time AI pipeline for detecting behavioral and
appearance outliers in crowds **without facial recognition or identity
tracking**. Outliers are scored relative to *local crowd context* rather
than fixed global rules — the project's core differentiator from commercial
systems (Avigilon ACC, BriefCam, SenseTime/Megvii).

## Pipeline (7 stages)

| # | Stage | Component | Status |
|---|-------|-----------|--------|
| 1 | Detection | YOLOv11-small (Ultralytics) | ✅ Built + tested |
| 2 | Tracking | ByteTrack | ✅ Built + tested |
| 3 | Optical Flow | RAFT-small (torchvision) | ✅ Built + tested |
| 4 | Pose Estimation | YOLO11-Pose (Ultralytics) | ⏳ Next |
| 5 | Appearance Embeddings | CLIP ViT-B/32 (OpenAI) | ❌ Not started |
| 6 | Clustering | DBSCAN (scikit-learn) | ❌ Not started |
| 7 | Anomaly Scoring | Isolation Forest + LSTM Autoencoder, calibrated via Platt scaling | ❌ Not started |

## Repo layout

```
Backend/
├── configs/              # YAML configs (paths, model params, thresholds)
├── data/
│   ├── raw/               # original input videos (gitignored)
│   ├── sample_videos/      # 1-2 dev videos used for current build-out
│   └── processed/         # extracted frames / cached intermediate outputs
├── src/
│   ├── detection/         # Stage 1: YOLOv11-small person detection
│   ├── tracking/           # Stage 2: ByteTrack multi-object tracking
│   ├── optical_flow/       # Stage 3: RAFT-small motion estimation
│   ├── pose/               # Stage 4: YOLO11-Pose keypoints
│   ├── appearance/         # Stage 5: CLIP embeddings
│   ├── clustering/         # Stage 6: DBSCAN local-context grouping
│   ├── anomaly/            # Stage 7: Isolation Forest + LSTM Autoencoder
│   ├── calibration/        # Platt scaling
│   ├── pipeline/           # Runnable entry points per stage, and (later) full orchestration
│   └── utils/              # shared helpers (video IO, visualization, logging, config)
├── notebooks/             # exploration / Colab training notebooks
├── tests/                 # unit tests per stage
└── outputs/
    ├── logs/
    └── results/            # annotated videos + JSON outputs per stage
```

## Environment

- **Local laptop**: dev work, CPU (Stage 1-3 verified working CPU-only)
- **Google Colab**: GPU, for anything CPU makes too slow (optical flow at scale, LSTM training in Stage 7)

## Setup

```bash
cd Backend
python -m venv venv
venv\Scripts\Activate.ps1       # Windows PowerShell
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

## Usage

Each stage has its own standalone runnable script — run them in order, each
one depends on the previous stage's JSON output:

```bash
# Stage 1 only — detection, no tracking (for isolated quality checks)
python -m src.pipeline.run_detect_only --video data/sample_videos/sample1.mp4

# Stage 1 + 2 — detection + tracking combined (produces *_tracks.json, needed by Stage 3+)
python -m src.pipeline.run_detect_track --video data/sample_videos/sample1.mp4

# Stage 3 — optical flow (reads *_tracks.json, produces *_motion.json)
python -m src.pipeline.run_optical_flow --video data/sample_videos/sample1.mp4
# --max_frames N to test on a subset first and estimate full-run time
```

Outputs (annotated videos + JSON) land in `outputs/results/`; logs in `outputs/logs/`.

## Current status

Stage 1 (Detection), Stage 2 (Tracking) and Stage 3 (Optical Flow) are built
and verified end-to-end on a real sample video (516 frames): detection runs
clean, tracking produces stable IDs, and per-track motion vectors (speed +
direction) are extracted and sanity-checked.

Next: **Stage 4 (Pose Estimation)**, which — like Stage 3 — consumes Stage 2's
tracks independently, so it can be built without waiting on Stage 3's output.

Currently building and testing against 1-2 local sample videos before scaling
to benchmark datasets (UCSD Anomaly Detection, CUHK Avenue, ShanghaiTech Campus).
```
