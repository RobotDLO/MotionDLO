#!/usr/bin/env python3
"""
Unpack tracker_polylines.npz into per-frame .npy files + timestamps.txt,
matching the format compute_pwl_error.py expects on --tracker-dir.

The .npz contains aligned arrays:
  timestamps_us : (N,)         int64,    microseconds (event/shared clock)
  polylines     : (N, M, 2)    float32,  event-space pixel coords (NaN if untracked)
  state         : (N,)         str,      "BOOT" | "STATIC" | "MOTION"
  branch        : (N,)         str,      "H" (hold) | "A" (SAM3) | "B" (event CPD)

This script extracts each row into:
  out_dir/frame_NNNNNN.npy
  out_dir/timestamps.txt          # filename  timestamp_sec

Filtering options:
  --drop-nan       drop rows whose polyline is entirely NaN
  --motion-only    drop rows whose state != "MOTION"
                   (use for evaluating only the moving phase)
  --branch-b-only  drop rows whose branch != "B"
                   (use for ablations: only count CPD-updated frames)
"""

import argparse
import json
import os
import cv2
import numpy as np


def _load_manifest_sorted(path):
    """Return (ts_us_sorted, fnames_sorted) parallel arrays for nearest-lookup."""
    with open(path) as f:
        raw = json.load(f)
    items = sorted(raw.items(), key=lambda kv: float(kv[1]))
    ts_arr = np.array([float(v) for _, v in items], dtype=np.int64)
    fn_arr = [k for k, _ in items]
    return ts_arr, fn_arr


def _nearest_frame_fname(target_us, manifest_ts_us, manifest_fnames):
    idx = int(np.searchsorted(manifest_ts_us, target_us))
    candidates = []
    if idx < len(manifest_ts_us):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates,
               key=lambda k: abs(int(manifest_ts_us[k]) - int(target_us)))
    return manifest_fnames[best]


def _draw_tracker_pwl(image, nodes, state, branch, frame_idx):
    """Yellow polyline + orange numbered circles + state/branch overlay."""
    out = image.copy()
    pts_int = []
    for x, y in nodes:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pts_int.append((int(round(x)), int(round(y))))

    if len(pts_int) > 1:
        for a, b in zip(pts_int[:-1], pts_int[1:]):
            cv2.line(out, a, b, (0, 255, 255), 2)
    for i, pt in enumerate(pts_int):
        cv2.circle(out, pt, 7, (0, 140, 255), -1)
        cv2.circle(out, pt, 7, (255, 255, 255), 1)
        cv2.putText(out, str(i + 1), (pt[0] + 9, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    head = f"Tracker PWL  ({len(pts_int)} nodes)  frame {frame_idx}"
    sub  = f"state={state}  branch={branch}"
    cv2.putText(out, head, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, sub,  (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 255), 2, cv2.LINE_AA)
    if not pts_int:
        cv2.putText(out, "NO TRACK", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz",           required=True,
                    help="Path to tracker_polylines.npz")
    ap.add_argument("--out-dir",       required=True,
                    help="Output folder for per-frame .npy + timestamps.txt")
    ap.add_argument("--frames-dir",    default=None,
                    help="Folder of RGB frames for visualisation overlay "
                         "(requires --manifest). PNGs are written next to .npy.")
    ap.add_argument("--manifest",      default=None,
                    help="manifest.json mapping frame filename to timestamp us. "
                         "Used to pick the nearest frame per tracker row.")
    ap.add_argument("--drop-nan",      action="store_true",
                    help="Skip rows where the polyline is entirely NaN "
                         "(tracker had no estimate that iteration)")
    ap.add_argument("--motion-only",   action="store_true",
                    help="Keep only rows with state == 'MOTION'")
    ap.add_argument("--branch-b-only", action="store_true",
                    help="Keep only rows with branch == 'B' (event CPD updated)")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    keys = set(data.files)
    polylines = data["polylines"]

    def _decode(arr):
        return [a.decode() if isinstance(a, bytes) else str(a) for a in arr]

    if {"state", "branch"} <= keys:
        # Legacy tracker_polylines.npz: one polyline per iteration, with
        # event-time ts + tracker state/branch tags.
        schema = "tracker"
        ts_us      = data["timestamps_us"]
        states_s   = _decode(data["state"])
        branches_s = _decode(data["branch"])
    elif "det_frame_idx" in keys:
        # Framebased polylines.npz: per-detection rows, frame registry on the side.
        # Expand per-frame timestamps to per-detection so the loop below stays uniform.
        schema       = "framebased"
        det_fidx     = data["det_frame_idx"]
        frame_ts_us  = data["timestamps_us"]              # length = N_frames
        ts_us        = frame_ts_us[det_fidx]              # length = N_det
        if "track_ids" in keys:                           # tracking mode
            tids = data["track_ids"]
            cfs  = data["confidences"]
            states_s   = [f"id={int(t)}" for t in tids]
            branches_s = [f"conf={float(c):.2f}" for c in cfs]
        else:                                             # simple mode
            states_s   = ["FRAMEBASED"] * len(polylines)
            branches_s = ["-"]          * len(polylines)
        if args.motion_only or args.branch_b_only:
            print("[WARN] --motion-only / --branch-b-only are tracker-only flags; "
                  "ignored for framebased polylines.npz")
            args.motion_only  = False
            args.branch_b_only = False
    else:
        raise RuntimeError(
            f"Unrecognized .npz schema in {args.npz}. "
            f"Expected either tracker_polylines.npz (keys: state, branch, ...) "
            f"or framebased polylines.npz (keys: det_frame_idx, ...). "
            f"Found keys: {sorted(keys)}"
        )

    n_total = len(ts_us)
    if not (len(polylines) == len(states_s) == len(branches_s) == n_total):
        raise RuntimeError(
            f"Array length mismatch in {args.npz}: "
            f"ts={n_total}, poly={len(polylines)}, "
            f"state={len(states_s)}, branch={len(branches_s)}"
        )

    print(f"[INFO] Loaded {n_total} entries from {args.npz}  (schema={schema})")
    print(f"[INFO] Polyline shape per row: "
          f"{polylines.shape[1]} nodes × {polylines.shape[2]} coords")
    print(f"[INFO] Time range: "
          f"{ts_us.min()*1e-6:.3f}s → {ts_us.max()*1e-6:.3f}s "
          f"({(ts_us.max()-ts_us.min())*1e-6:.3f}s span)")

    from collections import Counter
    print(f"[INFO] States  : {dict(Counter(states_s))}")
    print(f"[INFO] Branches: {dict(Counter(branches_s))}")

    # Optional visualisation setup
    draw_vis = bool(args.frames_dir and args.manifest)
    manifest_ts_us = manifest_fnames = None
    fallback_shape = (480, 640, 3)
    if draw_vis:
        if not os.path.isdir(args.frames_dir):
            raise RuntimeError(f"--frames-dir does not exist: {args.frames_dir}")
        if not os.path.exists(args.manifest):
            raise RuntimeError(f"--manifest does not exist: {args.manifest}")
        manifest_ts_us, manifest_fnames = _load_manifest_sorted(args.manifest)
        print(f"[INFO] Visualisation ON  ({len(manifest_fnames)} frames in manifest)")
    elif args.frames_dir or args.manifest:
        print("[WARN] --frames-dir and --manifest must both be set for "
              "visualisation; skipping PNG output")

    os.makedirs(args.out_dir, exist_ok=True)
    ts_path = os.path.join(args.out_dir, "timestamps.txt")

    kept = 0
    skipped_state  = 0
    skipped_branch = 0
    skipped_nan    = 0
    missing_frames = 0

    with open(ts_path, "w") as ts_file:
        ts_file.write("# filename  timestamp_sec\n")
        for i in range(n_total):
            if args.motion_only and states_s[i] != "MOTION":
                skipped_state += 1
                continue
            if args.branch_b_only and branches_s[i] != "B":
                skipped_branch += 1
                continue

            poly = polylines[i]
            if args.drop_nan and np.isnan(poly).all():
                skipped_nan += 1
                continue

            fname = f"frame_{i:06d}.npy"
            np.save(os.path.join(args.out_dir, fname),
                    poly.astype(np.float32))
            ts_file.write(f"{fname}  {ts_us[i] * 1e-6:.6f}\n")
            kept += 1

            if draw_vis:
                frame_fn = _nearest_frame_fname(
                    int(ts_us[i]), manifest_ts_us, manifest_fnames)
                bg = cv2.imread(os.path.join(args.frames_dir, frame_fn))
                if bg is None:
                    missing_frames += 1
                    bg = np.zeros(fallback_shape, dtype=np.uint8)
                else:
                    fallback_shape = bg.shape
                vis = _draw_tracker_pwl(bg, poly, states_s[i], branches_s[i], i)
                png_name = f"frame_{i:06d}.png"
                cv2.imwrite(os.path.join(args.out_dir, png_name), vis)

    print(f"\n[Done] Wrote {kept} frames to {args.out_dir}")
    if skipped_state:
        print(f"  Skipped {skipped_state} non-MOTION rows")
    if skipped_branch:
        print(f"  Skipped {skipped_branch} non-branch-B rows")
    if skipped_nan:
        print(f"  Skipped {skipped_nan} all-NaN rows")
    if missing_frames:
        print(f"  WARNING: {missing_frames} matched frame files could not be read")
    if kept == 0:
        print(f"  WARNING: nothing survived the filters — check your flags")


if __name__ == "__main__":
    main()