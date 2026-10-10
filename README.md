# Crowd Outlier Detection System

A privacy-preserving, real-time AI pipeline for detecting behavioral and
appearance outliers in crowds **without facial recognition or identity
tracking**. Outliers are scored relative to *local crowd context* rather
than fixed global rules — the project's core differentiator from commercial
systems (Avigilon ACC, BriefCam, SenseTime/Megvii).

## Pipeline (7 stages)

| # | Stage | Component |
|---|-------|-----------|
| 1 | Detection | YOLOv8-small (Ultralytics) |
| 2 | Tracking | ByteTrack |
| 3 | Optical Flow | RAFT-lite (PyTorch) |
| 4 | Pose Estimation | YOLO-Pose (Ultralytics) |
| 5 | Appearance Embeddings | CLIP ViT-B/32 (OpenAI) |
| 6 | Clustering | DBSCAN (scikit-learn) |
| 7 | Anomaly Scoring | Isolation Forest + LSTM Autoencoder, calibrated via Platt scaling |

## Repo layout

```
crowd-outlier-detection/
├── configs/              # YAML configs (paths, model params, thresholds)
├── data/
│   ├── raw/               # original input videos (gitignored)
│   ├── sample_videos/      # 1-2 dev videos used for current build-out
│   └── processed/         # extracted frames / cached intermediate outputs
├── src/
│   ├── detection/         # Stage 1: YOLOv8-small person detection
│   ├── tracking/           # Stage 2: ByteTrack multi-object tracking
│   ├── optical_flow/       # Stage 3: RAFT-lite motion estimation
│   ├── pose/               # Stage 4: YOLO-Pose keypoints
│   ├── appearance/         # Stage 5: CLIP embeddings
│   ├── clustering/         # Stage 6: DBSCAN local-context grouping
│   ├── anomaly/            # Stage 7: Isolation Forest + LSTM Autoencoder
│   ├── calibration/        # Platt scaling
│   ├── pipeline/           # Orchestration — runs stages end-to-end
│   └── utils/              # shared helpers (video IO, visualization, logging)
├── notebooks/             # exploration / Colab training notebooks
├── tests/                 # unit tests per stage
└── outputs/
    ├── logs/
    └── results/            # annotated videos, JSON outputs, metrics
```

## Environment

- **Local laptop**: inference/dev work, CPU (or local GPU if available)
- **Google Colab**: GPU-backed training (LSTM Autoencoder, fine-tuning)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Current status

Implementing **Stage 1 (Detection) → Stage 2 (Tracking)** first, since every
downstream stage consumes tracked person boxes. Building and testing against
1-2 local sample videos before scaling to benchmark datasets (UCSD Anomaly
Detection, CUHK Avenue, ShanghaiTech Campus).
