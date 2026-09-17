#!/usr/bin/env python3
"""
Batch GT Polyline Generator
============================
Loops over all 9 sequence folders inside a single cable root directory
and writes ground-truth polyline .npy files.

For each sequence N (1..9):
  - Reads binary cable masks from  <root>/frame_masks_N/
  - Writes (20, 2) float32 .npy    <root>/polylines_N/frame_XXXXXX.npy

Usage
-----
    python generate_polylines.py --root /path/to/Blue_Cable
    python generate_polylines.py --root /path/to/Green_Cable
    python generate_polylines.py --root /path/to/Orange_Cable

    # Change number of nodes (default 20):
    python generate_polylines.py --root /path/to/Blue_Cable --num-nodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

NUM_NODES     = 20
NUM_SEQUENCES = 9
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ── Skeleton extraction pipeline ──────────────────────────────────────────────

def extract_skeleton_polyline(cable_mask: np.ndarray,
                               num_nodes: int = NUM_NODES) -> np.ndarray:
    """
    Fit a smooth, equidistantly-sampled polyline along the cable medial axis.

    Returns
    -------
    np.ndarray  shape (num_nodes, 2) float32  — (x, y) pixel coords,
                ordered top-to-bottom (node 0 = topmost pixel).
                Returns shape (0, 2) if extraction fails.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cable_mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.empty((0, 2), dtype=np.float32)

    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    skel    = skeletonize(labels == largest).astype(np.uint8)
    if skel.sum() < 15:
        return np.empty((0, 2), dtype=np.float32)

    chain = _longest_skeleton_chain(skel)
    if len(chain) < 2:
        return np.empty((0, 2), dtype=np.float32)

    smoothed  = _savgol_smooth(chain)
    resampled = _resample_chord(smoothed, num_nodes)
    if resampled is None:
        return np.empty((0, 2), dtype=np.float32)

    # Order top-to-bottom: node 0 = smallest y (topmost pixel)
    if resampled[0, 1] > resampled[-1, 1]:
        resampled = resampled[::-1]

    return resampled


def _longest_skeleton_chain(skel: np.ndarray) -> np.ndarray:
    """Two-pass BFS to find the skeleton longest path (tree diameter)."""
    ys, xs = np.where(skel > 0)
    pts    = np.column_stack([xs, ys])
    if len(pts) < 2:
        return np.empty((0, 2), dtype=np.float32)

    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(pts)}
    adj = [[] for _ in range(len(pts))]
    for i, (x, y) in enumerate(pts):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = idx.get((int(x) + dx, int(y) + dy))
                if j is not None:
                    adj[i].append(j)

    def bfs_path(src: int) -> list[int]:
        parent = {src: -1}
        q      = deque([src])
        last   = src
        while q:
            u    = q.popleft()
            last = u
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        path, node = [], last
        while node != -1:
            path.append(node)
            node = parent[node]
        return path[::-1]

    deg       = np.array([len(a) for a in adj])
    endpoints = np.where(deg == 1)[0]
    src       = int(endpoints[0]) if len(endpoints) > 0 else 0
    far       = bfs_path(src)[-1]
    chain     = bfs_path(far)
    return pts[chain].astype(np.float32)


def _savgol_smooth(pts: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    N = len(pts)
    w = min(window, N) if N % 2 == 1 else min(window, N - 1 if N > 1 else 1)
    if w % 2 == 0:
        w += 1
    if w < poly + 2 or w > N:
        return pts.copy()
    x = savgol_filter(pts[:, 0], window_length=w, polyorder=poly)
    y = savgol_filter(pts[:, 1], window_length=w, polyorder=poly)
    return np.column_stack([x, y]).astype(np.float32)


def _resample_chord(pts: np.ndarray, M: int) -> np.ndarray | None:
    """Chord-length cubic B-spline resample to M equidistant nodes."""
    if pts is None or len(pts) < 2:
        return None
    mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts  = pts[mask]
    if len(pts) < 2:
        return None
    if len(pts) < 4:
        idx = np.round(np.linspace(0, len(pts) - 1, M)).astype(int)
        return pts[idx].astype(np.float32)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    u /= u[-1] + 1e-12
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=0,
                         k=min(3, len(pts) - 1))
        xu, yu = splev(np.linspace(0, 1, M), tck)
        return np.column_stack([xu, yu]).astype(np.float32)
    except Exception:
        idx = np.round(np.linspace(0, len(pts) - 1, M)).astype(int)
        return pts[idx].astype(np.float32)


# ── Batch processor ───────────────────────────────────────────────────────────

def process_sequence(masks_dir: str, out_dir: str, num_nodes: int) -> dict:
    """
    Process all masks in masks_dir, write .npy to out_dir.
    Returns a summary dict with counts.
    """
    os.makedirs(out_dir, exist_ok=True)

    mask_files = sorted(
        f for f in os.listdir(masks_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )
    if not mask_files:
        return {"total": 0, "empty": 0, "failed": 0}

    empty  = 0
    failed = 0

    for fname in mask_files:
        mask = cv2.imread(os.path.join(masks_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            failed += 1
            continue

        nodes = extract_skeleton_polyline(mask, num_nodes=num_nodes)

        if len(nodes) == 0:
            empty += 1

        npy_name = os.path.splitext(fname)[0] + ".npy"
        np.save(os.path.join(out_dir, npy_name), nodes)

    return {"total": len(mask_files), "empty": empty, "failed": failed}


def run(root: str, num_nodes: int) -> None:
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory not found: {root}")
        sys.exit(1)

    print(f"[INFO] Root     : {root}")
    print(f"[INFO] Sequences: 1 to {NUM_SEQUENCES}")
    print(f"[INFO] Nodes    : {num_nodes}")
    print()

    total_empty  = 0
    total_failed = 0

    for n in range(1, NUM_SEQUENCES + 1):
        masks_dir = os.path.join(root, f"frame_masks_{n}")
        out_dir   = os.path.join(root, f"polylines_{n}")

        if not os.path.isdir(masks_dir):
            print(f"  [{n}/9] SKIP — not found: {masks_dir}")
            continue

        summary = process_sequence(masks_dir, out_dir, num_nodes)

        status = "OK"
        if summary["empty"] > 0:
            status = f"WARN: {summary['empty']} empty skeletons"
        if summary["failed"] > 0:
            status += f"  {summary['failed']} read failures"

        print(f"  [{n}/9]  {masks_dir}")
        print(f"         → {out_dir}")
        print(f"         {summary['total']} frames  |  {status}")
        print()

        total_empty  += summary["empty"]
        total_failed += summary["failed"]

    print("[Done]")
    if total_empty > 0:
        print(f"  Total empty skeletons : {total_empty}  "
              f"(mask too fragmented or too small — .npy saved as (0,2))")
    if total_failed > 0:
        print(f"  Total read failures   : {total_failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Batch GT polyline generator — one cable root at a time")
    ap.add_argument(
        "--root", required=True,
        help="Cable root directory, e.g. /path/to/Blue_Cable")
    ap.add_argument(
        "--num-nodes", type=int, default=NUM_NODES,
        help=f"Number of polyline nodes per frame (default: {NUM_NODES})")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(root=args.root, num_nodes=args.num_nodes)