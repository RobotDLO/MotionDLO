#!/usr/bin/env python3
"""
Compute the per-node tracking error between a tracker's output and the GT.

Implements equations (23) and (24) from the MotionDLO thesis:

    e_t  = (1/M) Σ_m  || y_m - ŷ_m ||_2              (per update step)
    ē    = (1/T) Σ_t  e_t                            (time average)

Two metrics are computed:
  1. Arc-length paired error   — GT resampled to M, paired 1-to-1 by arc position
  2. Point-to-curve error      — each tracker node finds its closest point on GT curve
                                 (more robust when tracker covers different cable extent)

Evaluation window control:
  --first-n-steps N   evaluate only the first N time steps (tracker working correctly)
  --last-n-steps  N   evaluate only the last N time steps (tracker under stress)
  (default: evaluate all steps)

px → mm conversion:
  --cable-masks-dir   auto-compute from cable mask width + physical diameter
  --dlo-diameter-mm   physical cable outer diameter (default 6.5 for DLO2)
  --px-per-mm         explicit override
"""

import argparse
import csv
import os
import cv2
import numpy as np


# ── px → mm auto-computation ──────────────────────────────────────────────────
def compute_px_per_mm(cable_masks_dir, dlo_diameter_mm, max_frames=50,
                      row_step=20, min_width_px=3):
    supported = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(
        f for f in os.listdir(cable_masks_dir)
        if os.path.splitext(f)[1].lower() in supported
    )[:max_frames]

    if not files:
        print(f"[WARN] No mask images in {cable_masks_dir}")
        return None

    widths = []
    for fname in files:
        mask = cv2.imread(os.path.join(cable_masks_dir, fname),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        for row in range(0, mask.shape[0], row_step):
            w = int(np.sum(mask[row] > 0))
            if w >= min_width_px:
                widths.append(w)

    if not widths:
        print(f"[WARN] No valid cable rows — cannot compute px/mm")
        return None

    widths      = np.array(widths)
    median_w_px = float(np.median(widths))
    px_per_mm   = median_w_px / dlo_diameter_mm

    print(f"[INFO] Cable width  : median={median_w_px:.1f} px  "
          f"(sampled {len(widths)} rows from {len(files)} frames)")
    print(f"[INFO] DLO diameter : {dlo_diameter_mm} mm")
    print(f"[INFO] px / mm      : {px_per_mm:.4f}")
    print(f"[INFO] mm / px      : {1/px_per_mm:.4f}")
    return px_per_mm


# ── Arc-length resampling ─────────────────────────────────────────────────────
def resample_polyline_arc_length(pts, M):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return (np.zeros((M, 2)) if len(pts) == 0 else np.tile(pts[0], (M, 1)))

    seg   = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum   = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.tile(pts[0], (M, 1))

    targets = np.linspace(0.0, total, M)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return np.column_stack([xs, ys])


# ── Metric 1: arc-length paired error (equation 23) ──────────────────────────
def arc_length_error(gt_pts, tracker_pts):
    """
    GT is resampled to M points at the same arc-length fractions as the
    tracker. Node m of GT is paired with node m of tracker.
    Both orientations tried; smaller mean is returned.
    """
    tracker_pts = np.asarray(tracker_pts, dtype=float)
    gt_pts      = np.asarray(gt_pts,      dtype=float)

    if len(tracker_pts) < 2 or len(gt_pts) < 2:
        return np.nan

    M     = len(tracker_pts)
    gt_rs = resample_polyline_arc_length(gt_pts, M)

    d_fwd = np.mean(np.linalg.norm(gt_rs       - tracker_pts, axis=1))
    d_rev = np.mean(np.linalg.norm(gt_rs[::-1] - tracker_pts, axis=1))
    return float(min(d_fwd, d_rev))


# ── Metric 2: point-to-curve error ───────────────────────────────────────────
def point_to_curve_error(gt_pts, tracker_pts, n_gt_samples=200):
    """
    For each tracker node, find the closest point on the GT polyline
    (not at the same arc-length fraction — nearest in Euclidean space).
    Average those minimum distances.

    This is robust when tracker covers a different cable extent than GT,
    or when tracking has partially diverged.

    n_gt_samples: number of points to densely sample the GT curve for
                  nearest-neighbour lookup.
    """
    tracker_pts = np.asarray(tracker_pts, dtype=float)
    gt_pts      = np.asarray(gt_pts,      dtype=float)

    if len(tracker_pts) < 2 or len(gt_pts) < 2:
        return np.nan

    # densely sample the GT curve so nearest-neighbour is accurate
    gt_dense = resample_polyline_arc_length(gt_pts, n_gt_samples)

    # for each tracker node find distance to nearest GT sample
    dists = []
    for node in tracker_pts:
        d = np.linalg.norm(gt_dense - node, axis=1).min()
        dists.append(d)

    return float(np.mean(dists))


# ── I/O helpers ───────────────────────────────────────────────────────────────
def load_sequence(folder):
    ts_path = os.path.join(folder, "timestamps.txt")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"No timestamps.txt in {folder}")

    entries = []
    with open(ts_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            fname, ts = parts[0], float(parts[1])
            npy_path = os.path.join(folder, fname)
            if os.path.exists(npy_path):
                entries.append((ts, np.load(npy_path)))

    entries.sort(key=lambda e: e[0])
    if not entries:
        raise RuntimeError(f"No valid .npy entries in {folder}")

    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def nearest_index(sorted_ts, query_t):
    idx = np.searchsorted(sorted_ts, query_t)
    if idx == 0:              return 0
    if idx >= len(sorted_ts): return len(sorted_ts) - 1
    l, r = sorted_ts[idx - 1], sorted_ts[idx]
    return idx - 1 if abs(query_t - l) <= abs(query_t - r) else idx


def print_summary(label, errors_px, px_per_mm):
    if len(errors_px) == 0:
        print(f"  [{label}] no valid steps")
        return
    print(f"\n── {label} ─────────────────────────────────")
    print(f"  steps    : {len(errors_px)}")
    print(f"  mean  ē  : {errors_px.mean():7.3f} px"
          + (f"  =  {errors_px.mean()/px_per_mm:.3f} mm" if px_per_mm else ""))
    print(f"  median   : {np.median(errors_px):7.3f} px"
          + (f"  =  {np.median(errors_px)/px_per_mm:.3f} mm" if px_per_mm else ""))
    print(f"  std      : {errors_px.std():7.3f} px"
          + (f"  =  {errors_px.std()/px_per_mm:.3f} mm" if px_per_mm else ""))


# ── Main benchmark driver ─────────────────────────────────────────────────────
def evaluate(gt_dir, tracker_dir, update_ms=30.0,
             px_per_mm=None, cable_masks_dir=None, dlo_diameter_mm=6.5,
             max_ts_gap=0.1, csv_path=None,
             first_n_steps=None, last_n_steps=None):

    # ── px/mm ─────────────────────────────────────────────────────────────────
    if px_per_mm is None and cable_masks_dir is not None:
        print(f"\n── Auto-computing px/mm ────────────────────────────")
        px_per_mm = compute_px_per_mm(cable_masks_dir, dlo_diameter_mm)

    # ── load data ─────────────────────────────────────────────────────────────
    gt_ts,      gt_nodes      = load_sequence(gt_dir)
    tracker_ts, tracker_nodes = load_sequence(tracker_dir)

    t_start = max(gt_ts[0],  tracker_ts[0])
    t_end   = min(gt_ts[-1], tracker_ts[-1])
    if t_end <= t_start:
        raise RuntimeError("No temporal overlap")

    dt          = update_ms / 1000.0
    query_times = np.arange(t_start, t_end + dt * 0.5, dt)

    print(f"\n[INFO] GT       : {len(gt_ts):4d} entries  "
          f"{gt_ts[0]:.3f}s → {gt_ts[-1]:.3f}s")
    print(f"[INFO] Tracker  : {len(tracker_ts):4d} entries  "
          f"{tracker_ts[0]:.3f}s → {tracker_ts[-1]:.3f}s")
    print(f"[INFO] Overlap  : {t_start:.3f}s → {t_end:.3f}s  "
          f"({t_end - t_start:.3f}s)")
    print(f"[INFO] Total steps (30ms): {len(query_times)}")

    # ── apply window ──────────────────────────────────────────────────────────
    if first_n_steps is not None and last_n_steps is not None:
        print(f"[WARN] Both --first-n-steps and --last-n-steps given. "
              f"Using --first-n-steps only.")
        last_n_steps = None

    if first_n_steps is not None:
        query_times = query_times[:first_n_steps]
        print(f"[INFO] Evaluating FIRST {len(query_times)} steps  "
              f"({query_times[0]:.3f}s → {query_times[-1]:.3f}s)")
    elif last_n_steps is not None:
        query_times = query_times[-last_n_steps:]
        print(f"[INFO] Evaluating LAST {len(query_times)} steps  "
              f"({query_times[0]:.3f}s → {query_times[-1]:.3f}s)")
    else:
        print(f"[INFO] Evaluating ALL {len(query_times)} steps")

    # ── evaluate ──────────────────────────────────────────────────────────────
    per_step = []
    skipped  = 0

    for t in query_times:
        i_gt = nearest_index(gt_ts,      t)
        i_tr = nearest_index(tracker_ts, t)

        if abs(gt_ts[i_gt]      - t) > max_ts_gap: skipped += 1; continue
        if abs(tracker_ts[i_tr] - t) > max_ts_gap: skipped += 1; continue

        gt_n  = gt_nodes[i_gt]
        tr_n  = tracker_nodes[i_tr]

        e_arc = arc_length_error(gt_n, tr_n)
        e_ptc = point_to_curve_error(gt_n, tr_n)

        if np.isnan(e_arc) or np.isnan(e_ptc):
            skipped += 1
            continue

        per_step.append((t, e_arc, e_ptc, len(tr_n)))

    if not per_step:
        raise RuntimeError("No valid steps to evaluate")

    print(f"\n[INFO] Valid steps: {len(per_step)} / {len(query_times)}  "
          f"(skipped {skipped})")

    arc_errors = np.array([p[1] for p in per_step])
    ptc_errors = np.array([p[2] for p in per_step])

    # ── print results ─────────────────────────────────────────────────────────
    print_summary("Arc-length paired error  (eq. 23)", arc_errors, px_per_mm)
    print_summary("Point-to-curve error     (robust)", ptc_errors, px_per_mm)

    if px_per_mm:
        print(f"\n  [scale: {px_per_mm:.4f} px/mm → "
              f"{1/px_per_mm:.4f} mm/px]")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["time_sec", "e_arc_px", "e_ptc_px", "M"]
            if px_per_mm:
                header += ["e_arc_mm", "e_ptc_mm"]
            w.writerow(header)
            for (t, e_arc, e_ptc, M) in per_step:
                row = [f"{t:.6f}", f"{e_arc:.4f}", f"{e_ptc:.4f}", M]
                if px_per_mm:
                    row += [f"{e_arc/px_per_mm:.4f}",
                            f"{e_ptc/px_per_mm:.4f}"]
                w.writerow(row)
        print(f"\n  per-step CSV → {csv_path}")

    return per_step


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-node tracking error (equations 23/24) + point-to-curve")
    ap.add_argument("--gt-dir",           required=True)
    ap.add_argument("--tracker-dir",      required=True)
    ap.add_argument("--update-ms",        type=float, default=30.0)
    ap.add_argument("--px-per-mm",        type=float, default=None)
    ap.add_argument("--cable-masks-dir",  default=None)
    ap.add_argument("--dlo-diameter-mm",  type=float, default=6.5)
    ap.add_argument("--max-ts-gap",       type=float, default=0.1)
    ap.add_argument("--csv",              default=None)
    ap.add_argument("--first-n-steps",    type=int,   default=None,
                    help="Evaluate only first N timesteps (tracker working)")
    ap.add_argument("--last-n-steps",     type=int,   default=None,
                    help="Evaluate only last N timesteps (tracker under stress)")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        gt_dir          = a.gt_dir,
        tracker_dir     = a.tracker_dir,
        update_ms       = a.update_ms,
        px_per_mm       = a.px_per_mm,
        cable_masks_dir = a.cable_masks_dir,
        dlo_diameter_mm = a.dlo_diameter_mm,
        max_ts_gap      = a.max_ts_gap,
        csv_path        = a.csv,
        first_n_steps   = a.first_n_steps,
        last_n_steps    = a.last_n_steps,
    )