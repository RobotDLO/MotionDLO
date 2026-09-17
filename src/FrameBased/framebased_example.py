#!/usr/bin/env python3
"""
Frame-based cable polyline fitting 

Pipeline
--------
1. SAM3Segmenter  — generate binary masks from RGB frames
2. FramePolyline  — fit polylines to each mask
3. FrameVisualization — render and save side-by-side visualizations
"""

import glob
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

FRAMES_DIR = "/path/to/Dataset/Single_Cable/Green_Cable/Speed_50/1/frames"
OUTPUT_DIR = "/path/to/Results/MotionDLO_Test_Accuracy/Green_Cable/Speed_50/1/Framebased"

# ── Imports ────────────────────────────────────────────────────────────────────

from Segmentation  import SAM3Segmenter
from Polyline      import FramePolyline
from Visualization import FrameVisualization

NUM_NODES   = 20
MIN_LENGTH  = 150        # px — minimum cable arc to keep

SAVE_MASKS  = True
SAVE_CSV    = True
SAVE_VIS    = True
SAVE_NPZ    = True        # one consolidated polylines.npz for the whole run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _natural_key(p: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(p))
    return int(nums[-1]) if nums else -1

def _collect_frames(folder: str):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths, key=_natural_key)

def _load_manifest(frames_dir: str) -> dict:
    """manifest.json sits next to frames/; maps '<name>.jpg' -> timestamp_us."""
    p = Path(frames_dir).parent / "manifest.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    frames = _collect_frames(FRAMES_DIR)
    if not frames:
        raise SystemExit(f"No images found in: {FRAMES_DIR}")
    print(f"[INFO] {len(frames)} frames  |  mode=simple")

    out = os.path.join(OUTPUT_DIR, "simple")
    mask_dir = os.path.join(out, "masks");        os.makedirs(mask_dir, exist_ok=True)
    csv_dir  = os.path.join(out, "csv");          os.makedirs(csv_dir,  exist_ok=True)
    vis_dir  = os.path.join(out, "vis");          os.makedirs(vis_dir,  exist_ok=True)
    poly_dir = os.path.join(out, "polyline_only"); os.makedirs(poly_dir, exist_ok=True)

    manifest = _load_manifest(FRAMES_DIR)
    if SAVE_NPZ and not manifest:
        print(f"[WARN] manifest.json not found next to {FRAMES_DIR} — timestamps will be -1")

    # ── Accumulators for the consolidated polylines.npz ───────────────────────
    npz_polylines  : list = []   # per-detection: (NUM_NODES, 2) float32
    npz_det_fidx   : list = []   # per-detection: int — index into frame registry below
    npz_frame_names: list = []   # per-frame:     str  — frame stem
    npz_frame_ts   : list = []   # per-frame:     int  — timestamp_us from manifest (-1 if unknown)
    npz_cable_cnts : list = []   # per-frame:     int  — cables in this frame

    # ── Initialise modules ────────────────────────────────────────────────────
    segmenter = SAM3Segmenter()   # device auto-detected; model loaded lazily

    total_seg_time = 0.0
    total_fit_time = 0.0
    wall_start     = time.perf_counter()

    for fi, frame_path in enumerate(frames, 1):
        fname = Path(frame_path).stem
        print(f"\n[{fi:04d}/{len(frames)}] {fname}")

        bgr = cv2.imread(frame_path)
        if bgr is None:
            print(f"  [skip] cannot read {frame_path}")
            continue

        # ── Segmentation ───────────────────────────────────────────────────
        mask, seg_time = segmenter.segment_timed(bgr)
        total_seg_time += seg_time
        print(f"  seg={seg_time:.3f}s  fg_px={int((mask > 0).sum())}")

        if SAVE_MASKS:
            cv2.imwrite(os.path.join(mask_dir, f"mask_{fname}.png"), mask)

        # ── Polyline extraction ────────────────────────────────────────────
        t0 = time.perf_counter()

        # Independent per-frame fit 
        polylines = FramePolyline.extract(mask, num_nodes=NUM_NODES,
                                          min_length=MIN_LENGTH)
        fit_time  = time.perf_counter() - t0
        total_fit_time += fit_time
        print(f"  fit={fit_time:.3f}s  cables={len(polylines)}")

        for ci, Y in enumerate(polylines):
            print(f"    C{ci}: arc={FramePolyline.arc_length(Y):.0f}px")

        vis_polylines = polylines 

        # ── Save CSVs ──────────────────────────────────────────────────────
        if SAVE_CSV:
            for ci, Y in enumerate(vis_polylines):
                np.savetxt(
                    os.path.join(csv_dir, f"{fname}_c{ci:02d}.csv"),
                    Y, delimiter=",", header="x,y", comments="", fmt="%.3f")

        # ── Accumulate for the consolidated polylines.npz ─────────────────
        if SAVE_NPZ:
            frame_idx_in_registry = len(npz_frame_names)
            npz_frame_names.append(fname)
            npz_frame_ts.append(int(manifest.get(Path(frame_path).name, -1)))
            npz_cable_cnts.append(len(vis_polylines))

            for Y in vis_polylines:
                Yf = np.asarray(Y, dtype=np.float32)
                if Yf.shape[0] != NUM_NODES:
                    # Polyline came back at a different resolution — keep raw shape
                    # by skipping the stack-friendly path is risky; resampling lives
                    # in FramePolyline.extract, so this is a defensive fallback.
                    Yf = Yf[:NUM_NODES] if Yf.shape[0] > NUM_NODES else np.pad(
                        Yf, ((0, NUM_NODES - Yf.shape[0]), (0, 0)),
                        constant_values=np.nan)
                npz_polylines.append(Yf)
                npz_det_fidx.append(frame_idx_in_registry)

        # ── Visualization ──────────────────────────────────────────────────
        if SAVE_VIS:
            vis = FrameVisualization.compose(mask, vis_polylines,
                                             frame_path=frame_path, frame_idx=fi)
            cv2.imwrite(os.path.join(vis_dir, f"{fname}_vis.png"), vis)

            poly_only = FrameVisualization.draw_polyline_only(bgr, vis_polylines)
            cv2.imwrite(os.path.join(poly_dir, f"{fname}_polyline.png"), poly_only)

    # ── Write consolidated polylines.npz ──────────────────────────────────────
    if SAVE_NPZ and npz_frame_names:
        if npz_polylines:
            polys_arr = np.stack(npz_polylines, axis=0)
        else:
            polys_arr = np.empty((0, NUM_NODES, 2), dtype=np.float32)

        payload = {
            "polylines"    : polys_arr,
            "det_frame_idx": np.asarray(npz_det_fidx,   dtype=np.int64),
            "frame_names"  : np.asarray(npz_frame_names),
            "timestamps_us": np.asarray(npz_frame_ts,   dtype=np.int64),
            "cable_counts" : np.asarray(npz_cable_cnts, dtype=np.int64),
        }

        npz_path = os.path.join(out, "polylines.npz")
        np.savez(npz_path, **payload)
        print(f"  Wrote {len(npz_frame_names)} frames "
              f"({len(npz_polylines)} cable detections) → {npz_path}")

   
    wall = time.perf_counter() - wall_start
    n    = max(len(frames), 1)
    print(f"\n{'='*60}")
    print(f"  DONE  {len(frames)} frames  |  mode=simple")
    print(f"  Segmentation  total={total_seg_time:.1f}s  avg={total_seg_time/n:.3f}s/frame")
    print(f"  Polyline fit  total={total_fit_time:.1f}s  avg={total_fit_time/n:.3f}s/frame")
    print(f"  Wall time     {wall:.1f}s")
    print(f"  Outputs → {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()