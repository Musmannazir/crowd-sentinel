"""Stage 3 — Optical Flow.

Estimates dense per-pixel motion between consecutive frames using RAFT-small
(torchvision's pretrained implementation). This module only produces the
dense flow field; per-track motion extraction lives in motion_features.py,
kept separate so flow estimation can be reused/tested independently of
how it's later summarized per person.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


class OpticalFlowEstimator:
    """Wraps torchvision's RAFT-small for frame-pair dense optical flow."""

    def __init__(self, device: str = "cpu", resize_short_side: int = 240):
        self.device = device
        self.resize_short_side = resize_short_side

        self.weights = Raft_Small_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_small(weights=self.weights, progress=False).to(device).eval()

    def _resize_dims(self, h: int, w: int) -> tuple[int, int]:
        """Pick (new_h, new_w): short side == resize_short_side, both
        dimensions a multiple of 8 (RAFT requires divisible-by-8 input)."""
        if h <= w:
            new_h = self.resize_short_side
            new_w = int(round(w * (new_h / h)))
        else:
            new_w = self.resize_short_side
            new_h = int(round(h * (new_w / w)))
        new_h = max(8, (new_h // 8) * 8)
        new_w = max(8, (new_w // 8) * 8)
        return new_h, new_w

    @torch.no_grad()
    def estimate_flow(self, frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
        """Return dense flow as (H, W, 2) in frame1's ORIGINAL pixel scale.

        flow[y, x] = (dx, dy): how far the content at pixel (x, y) in
        frame1 moved by frame2, in pixels. Internally RAFT runs at a
        downscaled resolution (resize_short_side) for CPU speed, then the
        flow vectors are rescaled back up to the original resolution.
        """
        orig_h, orig_w = frame1_bgr.shape[:2]
        new_h, new_w = self._resize_dims(orig_h, orig_w)

        # BGR (OpenCV) -> RGB, HWC -> CHW, add batch dim
        img1 = torch.from_numpy(frame1_bgr[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
        img2 = torch.from_numpy(frame2_bgr[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0)

        img1 = TF.resize(img1, size=[new_h, new_w], antialias=False)
        img2 = TF.resize(img2, size=[new_h, new_w], antialias=False)

        img1, img2 = self.transforms(img1, img2)

        list_of_flows = self.model(img1.to(self.device), img2.to(self.device))
        flow = list_of_flows[-1][0].cpu()  # final refinement iteration, drop batch dim: (2, new_h, new_w)

        # Scale flow MAGNITUDE to match original resolution, then upsample the field
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        flow_dx = (flow[0] * scale_x).unsqueeze(0).unsqueeze(0)
        flow_dy = (flow[1] * scale_y).unsqueeze(0).unsqueeze(0)

        flow_dx_full = torch.nn.functional.interpolate(
            flow_dx, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        flow_dy_full = torch.nn.functional.interpolate(
            flow_dy, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        return np.stack([flow_dx_full, flow_dy_full], axis=-1)  # (H, W, 2)

    @classmethod
    def from_config(cls, config: dict) -> "OpticalFlowEstimator":
        of_cfg = config["optical_flow"]
        return cls(
            device=of_cfg.get("device", "cpu"),
            resize_short_side=of_cfg.get("resize_short_side", 240),
        )