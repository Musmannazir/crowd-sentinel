# Crowd Outlier Detection System

A privacy-preserving, real-time AI pipeline for detecting behavioral and appearance outliers in crowds **without facial recognition or identity tracking**. Outliers are scored relative to their local crowd context rather than fixed global rules.

The project's core differentiator is its focus on context-aware anomaly detection rather than relying solely on predefined rules or isolated individual behavior.

## Pipeline (7 Stages)

| # | Stage | Component | Status |
|---|---|---|---|
| 1 | Detection | YOLO11-small (Ultralytics) | ✅ Built and tested |
| 2 | Tracking | ByteTrack | ✅ Built and tested |
| 3 | Optical Flow | RAFT-small (torchvision) | ✅ Built and tested |
| 4 | Pose Estimation | YOLO11-Pose (Ultralytics) | ⏳ In progress |
| 5 | Appearance Embeddings | CLIP ViT-B/32 (OpenAI) | ❌ Not started |
| 6 | Clustering | DBSCAN (scikit-learn) | ❌ Not started |
| 7 | Anomaly Scoring | Isolation Forest + LSTM Autoencoder, with Platt scaling for calibration | ❌ Not started |

## Repository Structure

```text
Crowd_Sentinal/
├── Backend/
│   ├── configs/                 # YAML configuration files
│   ├── data/
│   │   ├── raw/                 # Original input videos (gitignored)
│   │   ├── sample_videos/       # Sample videos for development
│   │   └── processed/           # Extracted frames and cached outputs
│   ├── src/
│   │   ├── detection/           # Stage 1: YOLO11-small
│   │   ├── tracking/            # Stage 2: ByteTrack
│   │   ├── optical_flow/        # Stage 3: RAFT-small
│   │   ├── pose/                # Stage 4: YOLO11-Pose
│   │   ├── appearance/          # Stage 5: CLIP embeddings
│   │   ├── clustering/          # Stage 6: DBSCAN
│   │   ├── anomaly/              # Stage 7: Anomaly detection models
│   │   ├── calibration/         # Anomaly score calibration
│   │   ├── pipeline/            # Stage runners and orchestration
│   │   └── utils/                # Shared utilities
│   ├── notebooks/               # Experiments and training notebooks
│   ├── tests/                   # Unit and integration tests
│   ├── outputs/
│   │   ├── logs/                # Pipeline logs
│   │   └── results/             # Annotated videos and JSON outputs
│   └── requirements.txt
├── Frontend/                    # Scope not finalized yet (dashboard vs. demo UI)
└── README.md
```

## Environment

- **Local development:** Windows laptop; Stages 1–3 have been verified in CPU-only mode.
- **Google Colab:** GPU acceleration for computationally intensive optical flow processing and LSTM training.

## Setup

Run the following commands in Windows PowerShell from the repository root:

```powershell
cd Backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For macOS/Linux, activate the virtual environment using:

```bash
source venv/bin/activate
```

## Usage

Each implemented stage has a standalone entry point. Run the stages in order where dependencies require it.

### Stage 1: Person Detection

Run detection independently for isolated detection-quality checks.

```bash
python -m src.pipeline.run_detect_only --video data/sample_videos/sample1.mp4
```

### Stages 1–2: Detection and Tracking

Run person detection and ByteTrack together. This generates the track data required by Stage 3 and subsequent stages.

```bash
python -m src.pipeline.run_detect_track --video data/sample_videos/sample1.mp4
```

### Stage 3: Optical Flow

Run RAFT-small-based motion estimation using the tracking output. This generates per-track motion features, including speed and direction.

```bash
python -m src.pipeline.run_optical_flow --video data/sample_videos/sample1.mp4
```

Use `--max_frames N` to process a limited number of frames during testing and estimate processing time.

### Stage 4: Pose Estimation

Run YOLO11-Pose against the tracking output, matching each pose to its Stage 2 track_id via IoU.

```bash
python -m src.pipeline.run_pose --video data/sample_videos/sample1.mp4
```

**Note:** Run commands from the `Backend/` directory. Ensure the sample video exists at the specified path and that the required dependencies are installed.

Outputs, including annotated videos and JSON files, are saved in `outputs/results/`, while logs are saved in `outputs/logs/`.

## Current Development Status

### Completed

- **Stage 1 — Detection:** Person detection using YOLO11-small.
- **Stage 2 — Tracking:** Multi-object tracking using ByteTrack, with persistent track IDs.
- **Stage 3 — Optical Flow:** RAFT-small-based motion estimation and per-track motion feature extraction.

Stages 1–3 have been tested end-to-end on a real sample video containing 516 frames. Detection results were inspected, track IDs were checked for stability, and extracted motion vectors (speed and direction) were sanity-checked.

### In Progress

**Stage 4 — Pose Estimation**

YOLO11-Pose extraction and IoU-based matching to Stage 2 track IDs is implemented. Not yet verified end-to-end on a sample video due to an unrelated local environment issue (Windows Application Control blocking a torch DLL on the dev machine).

### Planned Evaluation

After the remaining stages are implemented, the system will be evaluated on benchmark datasets, including:

- UCSD Anomaly Detection Dataset
- CUHK Avenue Dataset
- ShanghaiTech Campus Dataset

Development and initial testing are currently focused on one or two local sample videos before scaling to larger benchmark datasets.

## Project Goal

The goal is to develop a modular crowd-anomaly detection pipeline that combines motion, pose, appearance, and local crowd context to identify unusual patterns for further review, while avoiding facial recognition.

The system is intended as a decision-support tool. An anomaly score indicates unusual behavior relative to the observed context; it does not, by itself, establish suspicious intent or wrongdoing.
