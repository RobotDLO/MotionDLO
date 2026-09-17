"""
Visualization utilities for frame-based cable polylines.

Usage
-----
    img = FrameVisualization.draw_polylines(bg, polylines)
    vis = FrameVisualization.compose(mask, polylines, frame_path=frame_path)
    cv2.imwrite("out.png", vis)
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np


_PALETTE = [
    (0, 255, 80),    # green
    (0, 200, 255),   # cyan
    (255, 120, 0),   # orange
    (200, 0, 255),   # purple
    (255, 50, 50),   # red
    (50, 255, 200),  # teal
]

# Adaptive thickness: drawn line = cable_radius * THICKNESS_SCALE, capped at MAX_THICK
THICKNESS_SCALE      = 3.5  # fraction of cable radius → looks proportional, not full-fill
MAX_THICK            = 6    # px cap so very thick cables don't become blobs
POLYLINE_ONLY_THICK  = 10   # wider stroke for polyline-only views so it reads well on images


def _dist_transform(mask: np.ndarray) -> np.ndarray:
    """Distance transform — value at each pixel = distance to nearest background."""
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


def _adaptive_thick(dist: np.ndarray, p: np.ndarray, fallback: int) -> int:
    h, w = dist.shape
    x = int(np.clip(p[0], 0, w - 1))
    y = int(np.clip(p[1], 0, h - 1))
    r = float(dist[y, x])
    if r <= 0:
        return fallback
    return max(2, min(int(round(r * THICKNESS_SCALE)), MAX_THICK))


class FrameVisualization:
    """
    Static-method toolkit for visualizing frame-based polyline results.
    """

    def draw_polylines(
        bg:         np.ndarray,
        polylines:  List[np.ndarray],
        frame_idx:  Optional[int] = None,
        label:      str = "",
        node_r:     int = 7,
        line_thick: int = 5,
        mask:       Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw polylines on a background image.

        Parameters
        ----------
        bg : (H, W) or (H, W, 3) numpy array — background image.
        polylines : list of (N, 2) float32 arrays [x, y].
        frame_idx : optional frame number shown in the header.
        label : optional text label shown in the header.
        node_r : radius of interior node dots.
        line_thick : fallback stroke width (used when mask is not provided).
        mask : optional (H, W) binary mask — enables adaptive line thickness
               proportional to the local cable radius (THICKNESS_SCALE × radius,
               capped at MAX_THICK).

        Returns
        -------
        (H, W, 3) BGR image.
        """
        img = bg.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        dist = _dist_transform(mask) if mask is not None else None

        for ci, Y in enumerate(polylines):
            pts = Y.astype(np.int32)
            col = _PALETTE[ci % len(_PALETTE)]
            M   = len(pts)

            # Polyline body
            for i in range(M - 1):
                if dist is not None:
                    t1 = _adaptive_thick(dist, pts[i],     line_thick)
                    t2 = _adaptive_thick(dist, pts[i + 1], line_thick)
                    thick = (t1 + t2) // 2
                else:
                    thick = line_thick
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), col, thick, cv2.LINE_AA)

            # Interior nodes — one circle per node
            for i in range(1, M - 1):
                cv2.circle(img, tuple(pts[i]), node_r, (0, 255, 255), -1, cv2.LINE_AA)

            # Endpoint markers
            for ep in [pts[0], pts[-1]]:
                cv2.circle(img, tuple(ep), node_r + 3, (0, 215, 255), 2,  cv2.LINE_AA)
                cv2.circle(img, tuple(ep), node_r,     (0, 255, 255),   -1, cv2.LINE_AA)

            # Arc-length label at midpoint
            mid = pts[M // 2]
            arc = float(np.sum(np.linalg.norm(np.diff(Y, axis=0), axis=1)))
            cv2.putText(img, f"C{ci}  arc={arc:.0f}",
                        (mid[0] + 5, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

        # Header
        hdr = (f"[{label}]  " if label else "") + \
              (f"Frame {frame_idx:04d}" if frame_idx is not None else "")
        if hdr:
            cv2.putText(img, hdr, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
        cv2.putText(img, f"cables:{len(polylines)}", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)
        return img

    def compose(
        mask:       np.ndarray,
        polylines:  List[np.ndarray],
        frame_path: Optional[str] = None,
        frame_idx:  Optional[int] = None,
    ) -> np.ndarray:
        """
        Side-by-side visualization: mask view (left) | image view (right).

        Line thickness adapts automatically to the local cable width.

        Parameters
        ----------
        mask : (H, W) binary mask (0/255 or bool).
        polylines : list of (N, 2) float32 arrays.
        frame_path : optional path to the original RGB/BGR frame.
        frame_idx : optional frame number for the header.

        Returns
        -------
        (H, W*2 + 3, 3) BGR image.
        """
        mask_u8 = ((mask > 0).astype(np.uint8) * 255)
        left    = FrameVisualization.draw_polylines(
            cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR),
            polylines, frame_idx, "MASK", mask=mask_u8)

        if frame_path and os.path.exists(frame_path):
            bg = cv2.imread(frame_path)
            if bg is None:
                bg = left.copy()
            elif bg.shape[:2] != mask_u8.shape[:2]:
                bg = cv2.resize(bg, (mask_u8.shape[1], mask_u8.shape[0]))
        else:
            bg = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        right = FrameVisualization.draw_polylines(
            bg, polylines, frame_idx, "IMAGE", mask=mask_u8)

        sep = np.full((left.shape[0], 3, 3), 180, dtype=np.uint8)
        return np.hstack([left, sep, right])

    def draw_polyline_only(
        bg:         np.ndarray,
        polylines:  List[np.ndarray],
        node_r:     int = 8,
        line_thick: int = POLYLINE_ONLY_THICK,
    ) -> np.ndarray:
        """
        Draw the full polyline tracking (line + interior nodes + endpoint markers)
        with no text labels, arc-length annotations, or headers.

        Parameters
        ----------
        bg : (H, W) or (H, W, 3) background image.
        polylines : list of (N, 2) float32 arrays [x, y].
        node_r : radius of interior node dots.
        line_thick : stroke width (defaults to POLYLINE_ONLY_THICK for strong visibility).

        Returns
        -------
        (H, W, 3) BGR image.
        """
        img = bg.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for ci, Y in enumerate(polylines):
            pts = Y.astype(np.int32)
            col = _PALETTE[ci % len(_PALETTE)]
            M   = len(pts)

            # Polyline body
            for i in range(M - 1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), col, line_thick, cv2.LINE_AA)

            # Interior nodes — one circle per node
            for i in range(1, M - 1):
                cv2.circle(img, tuple(pts[i]), node_r, (0, 255, 255), -1, cv2.LINE_AA)

            # Endpoint markers
            for ep in [pts[0], pts[-1]]:
                cv2.circle(img, tuple(ep), node_r + 3, (0, 215, 255), 2,  cv2.LINE_AA)
                cv2.circle(img, tuple(ep), node_r,     (0, 255, 255), -1, cv2.LINE_AA)
        return img