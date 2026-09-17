#!/usr/bin/env python3
"""GPU (PyTorch / CUDA) SAE recentness + NMS ridge wire mask.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from EventBased.ridge_obs import _keep_largest_elongated

_NEG = -10**18


class GPURidgeEngine:

    def __init__(self, ev_h: int, ev_w: int, binarize: str = "nms"):
        if str(binarize).lower() != "nms":
            raise RuntimeError("GPU ridge path supports --ridge-binarize nms only")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda not available")

        self.h, self.w = int(ev_h), int(ev_w)
        self.dev = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)
        # ROI shapes vary per batch, so use heuristic algorithm selection.
        torch.backends.cudnn.benchmark = False
        # Strict fp32: the NMS test compares nearly-equal neighbouring ridge
        # responses, and TF32's 10-bit mantissa shifts the band edges.
        torch.backends.cuda.matmul.allow_tf32 = False

        self.T_on = torch.full((self.h * self.w,), _NEG,
                               dtype=torch.int64, device=self.dev)
        self.T_off = torch.full((self.h * self.w,), _NEG,
                                dtype=torch.int64, device=self.dev)

        self._R: Optional[torch.Tensor] = None   # (1,1,h_roi,w_roi) float32
        # bboxes (y0,y1,x0,x1, exclusive ends): events since the last
        # recentness pass, and the last wire band.
        self._evt_bbox:  Optional[tuple[int, int, int, int]] = None
        self._mask_bbox: Optional[tuple[int, int, int, int]] = None
        self._roi = (0, self.h, 0, self.w)

        # Base pixel-coordinate grid for the NMS grid_sample step.
        ys, xs = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32),
            torch.arange(self.w, dtype=torch.float32),
            indexing="ij",
        )
        self._base_x = xs.to(self.dev)
        self._base_y = ys.to(self.dev)

        self._band_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._ellipse_cache: dict[int, torch.Tensor] = {}
        self._sobel = self._make_sobel()

        self._warmup()

    # ── kernel construction (identical weights to the cv2 calls) ───────────

    def _make_sobel(self) -> torch.Tensor:
        """(3,1,3,3) conv weight: channels = Ixx, Iyy, Ixy (cv2.Sobel ksize=3)."""
        k121 = np.array([1.0, 2.0, 1.0], np.float32)
        k1m21 = np.array([1.0, -2.0, 1.0], np.float32)
        km101 = np.array([-1.0, 0.0, 1.0], np.float32)
        ixx = np.outer(k121, k1m21)     # ky ⊗ kx for dx=2, dy=0
        iyy = np.outer(k1m21, k121)     # dx=0, dy=2
        ixy = np.outer(km101, km101)    # dx=1, dy=1
        w = np.stack([ixx, iyy, ixy])[:, None]   # (3,1,3,3)
        return torch.from_numpy(w).to(self.dev)

    @staticmethod
    def _gauss_taps(sigma: float) -> tuple[np.ndarray, int]:
        """Gaussian taps with cv2.GaussianBlur's auto kernel size (float input:
        ksize = round(sigma*4*2 + 1) | 1). Returns (taps, half_width)."""
        ksize = int(round(sigma * 8 + 1)) | 1
        return cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel(), ksize // 2

    def _band_matrix(self, n_out: int, sigma: float) -> torch.Tensor:
        """(n_out + 2p, n_out) banded matrix B with B[j+i, j] = k[i].

        `padded_row_vector @ B` is the 1-D Gaussian correlation of that row,
        so the separable blur becomes two matmuls rather than a wide conv.
        """
        key = (n_out, int(round(sigma * 1000)))
        m = self._band_cache.get(key)
        if m is None:
            k, p = self._gauss_taps(sigma)
            mat = np.zeros((n_out + 2 * p, n_out), np.float32)
            cols = np.arange(n_out)
            for i in range(len(k)):
                mat[cols + i, cols] = k[i]
            m = torch.from_numpy(mat).to(self.dev)
            self._band_cache[key] = m
        return m

    def _ellipse(self, rad: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        k = self._ellipse_cache.get((rad, dtype))
        if k is None:
            e = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)).astype(np.float32)
            k = torch.from_numpy(e).to(self.dev).view(
                1, 1, 2 * rad + 1, 2 * rad + 1).to(dtype)
            self._ellipse_cache[(rad, dtype)] = k
        return k

    # ── SAE maintenance ────────────────────────────────────────────────────

    def update_sae(self, xs: np.ndarray, ys: np.ndarray,
                   ts: np.ndarray, on: np.ndarray) -> None:
        """Scatter one filtered event batch into the GPU SAE planes.

        scatter-max makes duplicate pixels deterministic: the most recent
        timestamp wins, matching the CPU SAE's sequential overwrite.
        """
        idx_np = ys.astype(np.int64) * self.w + xs.astype(np.int64)
        if len(xs) > 0:
            ey0, ey1 = int(ys.min()), int(ys.max()) + 1
            ex0, ex1 = int(xs.min()), int(xs.max()) + 1
            b = self._evt_bbox
            self._evt_bbox = (ey0, ey1, ex0, ex1) if b is None else (
                min(b[0], ey0), max(b[1], ey1), min(b[2], ex0), max(b[3], ex1))
        n_on = int(np.count_nonzero(on))
        packed = np.empty((2, len(idx_np)), dtype=np.int64)
        packed[0, :n_on] = idx_np[on];  packed[0, n_on:] = idx_np[~on]
        packed[1, :n_on] = ts[on];      packed[1, n_on:] = ts[~on]
        dev_p = torch.from_numpy(packed).to(self.dev, non_blocking=True)
        if n_on > 0:
            self.T_on.scatter_reduce_(
                0, dev_p[0, :n_on], dev_p[1, :n_on], reduce="amax")
        if n_on < len(idx_np):
            self.T_off.scatter_reduce_(
                0, dev_p[0, n_on:], dev_p[1, n_on:], reduce="amax")

    # ── Recentness ─────────────────────────────────────────────────────────

    def _update_roi(self, wire_d_px: float) -> None:
        """ROI = (accumulated event bbox ∪ last band bbox) + kernel margin.

        The margin covers everything that can influence a mask pixel: the
        blur footprint (+ Sobel + NMS sample) plus the band-dilate radius.
        """
        boxes = [b for b in (self._evt_bbox, self._mask_bbox) if b is not None]
        self._evt_bbox = None   # consumed
        if not boxes:
            self._roi = (0, self.h, 0, self.w)
            return
        y0 = min(b[0] for b in boxes); y1 = max(b[1] for b in boxes)
        x0 = min(b[2] for b in boxes); x1 = max(b[3] for b in boxes)

        sigma = max(1.5, 0.5 * float(wire_d_px))
        blur_half = (int(round(sigma * 8 + 1)) | 1) // 2
        rad = max(1, int(round(0.5 * float(wire_d_px))))
        margin = blur_half + 2 + rad

        y0 = max(0, y0 - margin); y1 = min(self.h, y1 + margin)
        x0 = max(0, x0 - margin); x1 = min(self.w, x1 + margin)
        # Round dims up to multiples of 128: bounds the set of ROI shapes and
        # guarantees dims exceed the reflect-pad width.
        y1 = min(self.h, y0 + ((y1 - y0 + 127) // 128) * 128)
        x1 = min(self.w, x0 + ((x1 - x0 + 127) // 128) * 128)
        if (y1 - y0) * (x1 - x0) >= 0.85 * self.h * self.w:
            self._roi = (0, self.h, 0, self.w)
        else:
            self._roi = (y0, y1, x0, x1)

    def compute_recentness(self, t_now_us: int, tau_us: float,
                           wire_d_px: float = 50.0) -> None:
        """R = max(exp(-Δt_on/τ), exp(-Δt_off/τ)) over the activity ROI."""
        self._update_roi(wire_d_px)
        y0, y1, x0, x1 = self._roi
        T_on = self.T_on.view(self.h, self.w)[y0:y1, x0:x1]
        T_off = self.T_off.view(self.h, self.w)[y0:y1, x0:x1]
        inv = -1.0 / float(tau_us)
        dt_on = (int(t_now_us) - T_on).clamp_min(0).float()
        dt_off = (int(t_now_us) - T_off).clamp_min(0).float()
        R = torch.maximum(torch.exp(dt_on * inv), torch.exp(dt_off * inv))
        self._R = R.view(1, 1, y1 - y0, x1 - x0)
        # Sync so the pipeline's recentness_ms / wiremask_ms split is accurate.
        torch.cuda.synchronize()

    def apply_gate(self, gate_u8: np.ndarray) -> None:
        """Multiply R by a full-res {0,1} corridor mask (drawn on CPU)."""
        y0, y1, x0, x1 = self._roi
        g = torch.from_numpy(
            np.ascontiguousarray(gate_u8[y0:y1, x0:x1])).to(self.dev)
        self._R = self._R * g.view(1, 1, y1 - y0, x1 - x0).float()

    def download_R(self) -> np.ndarray:
        y0, y1, x0, x1 = self._roi
        R = np.zeros((self.h, self.w), dtype=np.float32)
        R[y0:y1, x0:x1] = self._R[0, 0].cpu().numpy()
        return R

    # ── Ridge NMS (port of ridge_obs._ridge_nms) ───────────────────────────

    def _blur(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        _, p = self._gauss_taps(sigma)
        h, w = img.shape[-2], img.shape[-1]
        xp = F.pad(img, (p, p, 0, 0), mode="reflect")[0, 0]     # (h, w+2p)
        gx = xp @ self._band_matrix(w, sigma)                   # (h, w)
        gp = F.pad(gx.view(1, 1, h, w), (0, 0, p, p), mode="reflect")[0, 0]
        g = self._band_matrix(h, sigma).T @ gp                  # (h, w)
        return g.view(1, 1, h, w)

    def _ridge_nms(self, sigma: float, floor: float) -> torch.Tensor:
        g = self._blur(self._R, sigma)
        s2 = float(sigma) * float(sigma)
        D = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), self._sobel) * s2
        Ixx, Iyy, Ixy = D[:, 0:1], D[:, 1:2], D[:, 2:3]

        half = (Ixx + Iyy) * 0.5
        disc = torch.sqrt(torch.clamp(((Ixx - Iyy) * 0.5) ** 2 + Ixy * Ixy, min=0.0))
        lam2 = half - disc
        v = torch.clamp(-lam2, min=0.0)
        # Clamped max avoids a host sync: on an empty frame the NMS test
        # below is all-False anyway.
        v = v / v.max().clamp_min(1e-30)

        nx = Ixy.clone()
        ny = lam2 - Ixx
        nrm = torch.sqrt(nx * nx + ny * ny) + 1e-12
        nx = (nx / nrm)[0, 0]
        ny = (ny / nrm)[0, 0]

        # Both ±normal samples in one batched grid_sample, in ROI-local coords.
        h_c, w_c = v.shape[-2], v.shape[-1]
        bx = self._base_x[:h_c, :w_c]
        by = self._base_y[:h_c, :w_c]
        gx = 2.0 * (bx + nx) / (w_c - 1) - 1.0
        gy = 2.0 * (by + ny) / (h_c - 1) - 1.0
        gxm = 2.0 * (bx - nx) / (w_c - 1) - 1.0
        gym = 2.0 * (by - ny) / (h_c - 1) - 1.0
        grid = torch.stack([torch.stack([gx, gy], dim=-1),
                            torch.stack([gxm, gym], dim=-1)], dim=0)  # (2,H,W,2)
        vpm = F.grid_sample(v.expand(2, -1, -1, -1), grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return (v >= vpm[0:1]) & (v >= vpm[1:2]) & (v > float(floor))

    # ── Morphology (exact cv2 border semantics) ────────────────────────────

    def _dilate(self, rc: torch.Tensor, rad: int) -> torch.Tensor:
        # cv2.dilate border = -inf ⇒ zero padding
        k = self._ellipse(rad)
        cnt = F.conv2d(rc.float(), k, padding=rad)
        return cnt > 0.5

    def _erode(self, rc: torch.Tensor, rad: int) -> torch.Tensor:
        # cv2.erode border = +inf ⇒ pad with ones
        k = self._ellipse(rad)
        cnt = F.conv2d(F.pad(rc.float(), (rad,) * 4, value=1.0), k)
        return cnt > float(k.sum()) - 0.5

    # ── Public: full ridge_wire_mask equivalent (NMS path) ─────────────────

    def wire_mask(self, wire_d_px: float, args, speed_px: float = 0.0) -> np.ndarray:
        """Port of ridge_obs.ridge_wire_mask (binarize='nms'), full frame.

        compute_recentness (and optionally apply_gate) must have run for the
        current batch. Returns a full-res bool mask on the CPU.
        """
        empty = np.zeros((self.h, self.w), dtype=bool)

        sigma_lo    = float(getattr(args, "ridge_sigma_lo",   1.5))
        nms_floor   = float(getattr(args, "ridge_nms_floor",  0.10))
        band_dilate = float(getattr(args, "ridge_band_dilate", -1.0))
        min_length  = float(getattr(args, "ridge_min_length", 40.0))
        min_aspect  = float(getattr(args, "ridge_min_aspect",  2.0))
        top_k       = int(getattr(args, "ridge_top_k", 5))

        speed_lo = float(getattr(args, "ridge_speed_lo", 10.0))
        speed_hi = float(getattr(args, "ridge_speed_hi", 35.0))
        if speed_px >= speed_hi:
            alpha = 1.0
        elif speed_px <= speed_lo:
            alpha = 0.0
        else:
            alpha = (speed_px - speed_lo) / (speed_hi - speed_lo)

        t0 = time.perf_counter()
        sigma_nms = max(sigma_lo, 0.5 * float(wire_d_px))
        rc = self._ridge_nms(sigma_nms, nms_floor)

        if alpha > 0.0:
            close_fast_px = float(getattr(args, "ridge_close_fast_px", 7.0))
            close_r = max(1, int(round(close_fast_px * alpha)))
            rc = self._erode(self._dilate(rc, close_r), close_r)

        if band_dilate <= 0.0:
            rad = max(1, int(round(0.5 * float(wire_d_px))))
        else:
            rad = int(round(band_dilate))
        if rad >= 1:
            rc = self._dilate(rc, rad)

        # Stage split for the timing table: [0] ROI masking, [1] mask
        # extraction (download + candidate selection + assembly).
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rc_np = rc[0, 0].to(torch.uint8).cpu().numpy().astype(bool)
        kept = _keep_largest_elongated(
            rc_np, min_length=min_length, min_aspect=min_aspect, top_k=top_k)

        y0, y1, x0, x1 = self._roi
        rows = kept.any(axis=1)
        if rows.any():
            cols = kept.any(axis=0)
            ry = np.flatnonzero(rows); rx = np.flatnonzero(cols)
            self._mask_bbox = (y0 + int(ry[0]), y0 + int(ry[-1]) + 1,
                               x0 + int(rx[0]), x0 + int(rx[-1]) + 1)
            # keep the last non-empty bbox when the band momentarily vanishes
        if (y0, y1, x0, x1) == (0, self.h, 0, self.w):
            full = kept
        else:
            full = empty
            full[y0:y1, x0:x1] = kept
        self.last_split_ms = ((t1 - t0) * 1000, (time.perf_counter() - t1) * 1000)
        return full

    # ── Warm-up ────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """One throw-away pass so CUDA init / cudnn autotune isn't paid in-loop.

        Draws a synthetic wire into the SAE so the full path executes; an
        empty frame would early-return before most kernels are built. Uses
        the default wire_d=50.
        """
        class _A:  # defaults only
            pass
        line = np.zeros((self.h, self.w), np.uint8)
        cv2.line(line, (self.w // 8, self.h // 2),
                 (self.w * 7 // 8, self.h // 3), 1, 25)
        ys, xs = np.nonzero(line)
        self.update_sae(xs.astype(np.int32), ys.astype(np.int32),
                        np.full(len(xs), 1000, np.int64),
                        np.ones(len(xs), dtype=bool))
        self.compute_recentness(1500, 10000)
        self.wire_mask(50.0, _A())
        # Exercise the speed-adaptive close radii (1..7): the first use of
        # each conv kernel size pays cudnn workspace allocation.
        rc = self._ridge_nms(25.0, 0.10)
        for r in range(1, 8):
            self._erode(self._dilate(rc, r), r)
        # leave no trace of the synthetic wire
        self.T_on.fill_(_NEG)
        self.T_off.fill_(_NEG)
        self._R = None
        self._evt_bbox = None
        self._mask_bbox = None
        self._roi = (0, self.h, 0, self.w)
        torch.cuda.synchronize()
