"""Smoke tests for Stage 1 (PersonDetector).

These don't assert specific detections (that depends on real footage) —
they just confirm the model loads and returns well-formed output on a
synthetic blank frame, so CI/local runs catch broken installs early.
"""
import numpy as np

from src.detection.detector import Detection, PersonDetector


def test_detector_loads_and_runs():
    detector = PersonDetector(model_path="yolov8s.pt", device="cpu")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(blank_frame, frame_idx=0)

    assert isinstance(detections, list)
    for d in detections:
        assert isinstance(d, Detection)
        assert len(d.bbox) == 4
        assert 0.0 <= d.confidence <= 1.0
