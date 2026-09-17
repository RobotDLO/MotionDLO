"""
GT Nodes + PWL Generator from Pre-existing Masks
=================================================
Takes already-created binary cable masks (and optionally tape masks) and
produces:
  - gt_nodes/   — (NUM_NODES, 2) .npy files + timestamps.txt
  - gt_pwl/     — visualisation images showing the fitted polyline

Use this when you have already created masks manually or via another tool
and do not need to re-run HSV thresholding.

Usage
-----
    # Cable mask only (no tape):
    python mask_to_gt.py \
        --cable-masks-dir /path/to/cable_masks \
        --frames-dir      /path/to/original_frames \
        --out-dir         /path/to/output \
        --num-nodes       10

    # With manifest for timestamps:
    python mask_to_gt.py \
        --cable-masks-dir /path/to/cable_masks \
        --frames-dir      /path/to/original_frames \
        --out-dir         /path/to/output \
        --manifest        /path/to/manifest.json \
        --num-nodes       10
"""

import argparse
import json
import os
import re
import sys
from collections import deque

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_CABLE_MASKS_DIR = "path/to/ground_truth/cable_masks"
DEFAULT_TAPE_MASKS_DIR  = "path/to/ground_truth/tape_masks"   # set to None if no tape
DEFAULT_FRAMES_DIR      = "path/to/frames"
DEFAULT_OUT_DIR         = "path/to/ground_truth"
DEFAULT_MANIFEST        = "path/to/frames/manifest.json"
NUM_NODES               = 20

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# ── Skeleton extraction pipeline ─────────────────────────────────────────────
def extract_skeleton_polyline(cable_mask, num_nodes=NUM_NODES, min_skel_px=15):
    """
    Fit a smooth, equidistantly-sampled polyline along the cable's medial axis.
    Returns (num_nodes, 2) float32 in (x, y), or empty array if unusable.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cable_mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return np.empty((0, 2), dtype=np.float32)

    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    skel    = skeletonize(labels == largest).astype(np.uint8)
    if skel.sum() < min_skel_px:
        return np.empty((0, 2), dtype=np.float32)

    chain = _longest_skeleton_chain(skel)
    if len(chain) < 2:
        return np.empty((0, 2), dtype=np.float32)

    smoothed  = _savgol_smooth(chain)
    resampled = _resample_chord(smoothed, num_nodes)
    if resampled is None:
        return np.empty((0, 2), dtype=np.float32)
    # Order top→bottom: node 1 = smallest y (top of image), node N = largest y.
    if resampled[0, 1] > resampled[-1, 1]:
        resampled = resampled[::-1]
    return resampled


def _longest_skeleton_chain(skel):
    """Two-pass BFS to find the skeleton's longest path (tree diameter)."""
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
    deg = np.array([len(a) for a in adj])

    def bfs_path(src):
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

    endpoints = np.where(deg == 1)[0]
    src   = int(endpoints[0]) if len(endpoints) > 0 else 0
    far   = bfs_path(src)[-1]
    chain = bfs_path(far)
    return pts[chain].astype(np.float32)


def _savgol_smooth(pts, window=11, poly=3):
    N = len(pts)
    w = min(window, N) if N % 2 == 1 else min(window, N - 1 if N > 1 else 1)
    if w % 2 == 0:
        w += 1
    if w < poly + 2 or w > N:
        return pts.copy()
    x = savgol_filter(pts[:, 0], window_length=w, polyorder=poly)
    y = savgol_filter(pts[:, 1], window_length=w, polyorder=poly)
    return np.column_stack([x, y]).astype(np.float32)


def _resample_chord(pts, M):
    """Chord-length spline resample to M equidistant nodes."""
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


# ── Visualisation ─────────────────────────────────────────────────────────────
def draw_pwl_curve(image, nodes):
    """Draw GT polyline on image. Returns annotated image."""
    out = image.copy()
    if len(nodes) < 1:
        return out

    pts_int = [(int(round(x)), int(round(y))) for x, y in nodes]

    if len(pts_int) > 1:
        for a, b in zip(pts_int[:-1], pts_int[1:]):
            cv2.line(out, a, b, (0, 255, 255), 2)

    for i, pt in enumerate(pts_int):
        cv2.circle(out, pt, 7, (0, 140, 255), -1)
        cv2.circle(out, pt, 7, (255, 255, 255), 1)
        cv2.putText(out, str(i + 1), (pt[0] + 9, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.putText(out, f"GT PWL  ({len(pts_int)} nodes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ── Manifest ──────────────────────────────────────────────────────────────────
def find_manifest(frames_dir, explicit_path=None):
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    for candidate in [
        os.path.join(frames_dir, "manifest.json"),
        os.path.join(frames_dir, "..", "manifest.json"),
    ]:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        raw = json.load(f)
    return {fname: float(ts_us) * 1e-6 for fname, ts_us in raw.items()}


def _trailing_digits(name):
    """Last numeric run in the stem, with leading zeros stripped."""
    stem = os.path.splitext(os.path.basename(name))[0]
    runs = re.findall(r"\d+", stem)
    if not runs:
        return None
    return runs[-1].lstrip("0") or "0"


def _build_manifest_digit_index(manifest):
    """Index manifest entries by the trailing digit run of their key.

    """
    if manifest is None:
        return None
    idx = {}
    for k in manifest:
        d = _trailing_digits(k)
        if d is None:
            continue
        if d in idx and idx[d] != k:
            return None  # ambiguous — refuse to guess
        idx[d] = k
    return idx


# ── Main batch processor ──────────────────────────────────────────────────────
def process(cable_masks_dir, out_dir, frames_dir=None,
            tape_masks_dir=None, manifest_path=None, num_nodes=NUM_NODES):
    """
    Read binary cable masks (+ optional tape masks) and produce GT nodes.

    Parameters
    ----------
    cable_masks_dir : str   folder containing binary cable mask images
    out_dir         : str   output folder (gt_nodes/ and gt_pwl/ created here)
    frames_dir      : str   original RGB frames for PWL visualisation overlay
                            (if None, visualisation is drawn on a black background)
    tape_masks_dir  : str   optional folder of binary tape mask images
    manifest_path   : str   optional path to manifest.json (frame → µs)
    num_nodes       : int   number of GT polyline nodes
    """
    gt_nodes_dir = os.path.join(out_dir, "gt_nodes")
    gt_pwl_dir   = os.path.join(out_dir, "gt_pwl")
    os.makedirs(gt_nodes_dir, exist_ok=True)
    os.makedirs(gt_pwl_dir,   exist_ok=True)

    # load manifest if available
    manifest_path_found = find_manifest(frames_dir or "", manifest_path)
    manifest = None
    if manifest_path_found:
        manifest = load_manifest(manifest_path_found)
        print(f"[INFO] Manifest loaded: {manifest_path_found}")
        print(f"[INFO]   {len(manifest)} timestamps")
    else:
        print(f"[INFO] No manifest found — timestamps.txt will NOT be written")

    # collect mask filenames sorted by name
    mask_files = sorted(
        f for f in os.listdir(cable_masks_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )
    if not mask_files:
        print(f"[ERROR] No mask images found in {cable_masks_dir}")
        sys.exit(1)

    has_tape   = tape_masks_dir is not None and os.path.isdir(tape_masks_dir)
    has_frames = frames_dir is not None and os.path.isdir(frames_dir)

    if has_tape:
        print(f"[INFO] Tape masks: {tape_masks_dir}")
    else:
        print(f"[INFO] No tape masks — using cable mask only")

    print(f"[INFO] {len(mask_files)} mask files  →  num_nodes={num_nodes}")
    

    ts_file       = None
    missing_in_ts = []
    failed        = []
    empty_skel    = 0

    manifest_digit_idx = _build_manifest_digit_index(manifest)
    if manifest is not None and manifest_digit_idx is None:
        print("[WARN] Manifest keys have ambiguous numeric suffixes — "
              "digit-only fallback disabled, lookups will require exact stem match")

    if manifest is not None:
        ts_file = open(os.path.join(gt_nodes_dir, "timestamps.txt"), "w")
        ts_file.write("# filename  timestamp_sec\n")

    for i, fname in enumerate(mask_files):
        cable_mask = cv2.imread(
            os.path.join(cable_masks_dir, fname), cv2.IMREAD_GRAYSCALE)
        if cable_mask is None:
            failed.append(fname)
            continue

        if has_tape:
            tape_mask = cv2.imread(
                os.path.join(tape_masks_dir, fname), cv2.IMREAD_GRAYSCALE)
            if tape_mask is not None:
                full_mask = cv2.bitwise_or(cable_mask, tape_mask)
            else:
                full_mask = cable_mask
        else:
            full_mask = cable_mask

        gt_nodes = extract_skeleton_polyline(full_mask, num_nodes=num_nodes)

        if len(gt_nodes) == 0:
            empty_skel += 1

        npy_name = os.path.splitext(fname)[0] + ".npy"
        np.save(os.path.join(gt_nodes_dir, npy_name), gt_nodes)

        # write timestamp
        # mask filenames may differ from manifest keys in:
        #   - extension     (mask .png vs manifest .jpg)
        #   - prefix        (mask "000001" vs manifest "frame_000001")
        # Try exact, then stem+alt-ext, then trailing-digit fallback.
        if ts_file is not None:
            stem   = os.path.splitext(fname)[0]
            ts_val = manifest.get(fname)
            if ts_val is None:
                for alt_ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                    ts_val = manifest.get(stem + alt_ext)
                    if ts_val is not None:
                        break
            if ts_val is None and manifest_digit_idx is not None:
                d = _trailing_digits(fname)
                if d is not None:
                    matched_key = manifest_digit_idx.get(d)
                    if matched_key is not None:
                        ts_val = manifest.get(matched_key)
            if ts_val is not None:
                ts_file.write(f"{npy_name}  {ts_val:.6f}\n")
            else:
                missing_in_ts.append(fname)

        # visualisation — check which file actually exists before reading
        # avoids OpenCV warnings from trying non-existent extensions
        if has_frames:
            stem       = os.path.splitext(fname)[0]
            frame_path = None
            for alt_ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                candidate = os.path.join(frames_dir, stem + alt_ext)
                if os.path.exists(candidate):
                    frame_path = candidate
                    break
            bg = cv2.imread(frame_path) if frame_path else None
            if bg is None:
                bg = np.zeros((*cable_mask.shape, 3), dtype=np.uint8)
        else:
            bg = np.zeros((*cable_mask.shape, 3), dtype=np.uint8)

        pwl_img = draw_pwl_curve(bg, gt_nodes.tolist() if len(gt_nodes) else [])
        cv2.imwrite(os.path.join(gt_pwl_dir, fname), pwl_img)

        if (i + 1) % 50 == 0 or (i + 1) == len(mask_files):
            print(f"  {i+1:4d}/{len(mask_files)}  {fname}  "
                  f"gt_nodes={len(gt_nodes)}")

    if ts_file is not None:
        ts_file.close()

    print("\n[Done]")
    print(f"  GT nodes → {gt_nodes_dir}")
    print(f"  GT PWL   → {gt_pwl_dir}")
    if empty_skel > 0:
        print(f"  WARNING: {empty_skel} frames produced empty skeletons "
              f"(mask too fragmented or too small)")
    if missing_in_ts:
        sample = ", ".join(missing_in_ts[:5])
        more   = "" if len(missing_in_ts) <= 5 else f", ... (+{len(missing_in_ts)-5} more)"
        print(f"  WARNING: {len(missing_in_ts)} mask(s) absent from manifest: {sample}{more}")
        print(f"           → these rows are NOT written to timestamps.txt. "
              f"Check that mask names share a stem/numeric suffix with manifest keys.")
    if failed:
        print(f"  Failed  : {failed}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate GT nodes + PWL visualisation from pre-existing masks")
    ap.add_argument("--cable-masks-dir", default=DEFAULT_CABLE_MASKS_DIR,
                    help="Folder of binary cable mask images")
    ap.add_argument("--tape-masks-dir",  default=DEFAULT_TAPE_MASKS_DIR,
                    help="Folder of binary tape mask images (optional)")
    ap.add_argument("--frames-dir",      default=DEFAULT_FRAMES_DIR,
                    help="Original RGB frames for visualisation overlay (optional)")
    ap.add_argument("--out-dir",         default=DEFAULT_OUT_DIR,
                    help="Output folder (gt_nodes/ and gt_pwl/ created here)")
    ap.add_argument("--manifest",        default=DEFAULT_MANIFEST,
                    help="Path to manifest.json (frame→µs) for timestamps.txt")
    ap.add_argument("--num-nodes",       type=int, default=NUM_NODES,
                    help=f"Number of GT polyline nodes (default {NUM_NODES})")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        cable_masks_dir = args.cable_masks_dir,
        out_dir         = args.out_dir,
        frames_dir      = args.frames_dir,
        tape_masks_dir  = args.tape_masks_dir,
        manifest_path   = args.manifest,
        num_nodes       = args.num_nodes,
    )