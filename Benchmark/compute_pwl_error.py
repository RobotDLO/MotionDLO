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

EVALUATION STRATEGY (changed from the TrackDLO version):
  Errors are computed at every GT timestamp (not on a uniform 30 ms grid).
  Rationale: GT only exists at GT timestamps. A uniform query grid forces
  nearest-GT lookups that, at fast motion, alias real cable motion into the
  error number (e.g. at 0.5 m/s the cable moves ~15 mm in 30 ms — larger than
  expected tracking error). Anchoring queries at GT timestamps removes that
  aliasing by making the GT-side temporal error exactly zero. The tracker
  side is sampled at ~20 ms (one row per event batch), so a tracker frame
  within 25 ms of every GT timestamp is essentially always available.

Evaluation window control:
  --first-n-steps N   evaluate only the first N GT timestamps
  --last-n-steps  N   evaluate only the last N GT timestamps
  (default: evaluate all GT timestamps within the tracker's recording window)

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


def save_overlay(out_dir, idx, t, gt_pts, tracker_pts, e_arc, e_ptc,
                 px_per_mm, img_w, img_h):
    """One PNG per evaluated step: black canvas, green GT, yellow tracker."""
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    def _draw(pts, color, node_r):
        arr = np.asarray(pts, dtype=np.int32)
        if len(arr) >= 2:
            cv2.polylines(canvas, [arr.reshape(-1, 1, 2)],
                          False, color, 2, cv2.LINE_AA)
        for pt in arr:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), node_r, color, -1)

    _draw(gt_pts,      (0, 255, 0),   4)   # green   = GT
    _draw(tracker_pts, (0, 255, 255), 3)   # yellow  = tracker

    hud = [
        f"t={t:.3f}s  M={len(np.asarray(tracker_pts))}",
        f"e_arc={e_arc:.2f}px" + (f"  ({e_arc/px_per_mm:.2f}mm)" if px_per_mm else ""),
        f"e_ptc={e_ptc:.2f}px" + (f"  ({e_ptc/px_per_mm:.2f}mm)" if px_per_mm else ""),
    ]
    for i, line in enumerate(hud):
        cv2.putText(canvas, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT",      (10, img_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   1, cv2.LINE_AA)
    cv2.putText(canvas, "Tracker", (10, img_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    fname = f"overlay_{idx:06d}_t{int(round(t * 1000)):08d}ms.png"
    cv2.imwrite(os.path.join(out_dir, fname), canvas)


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
def evaluate(gt_dir, tracker_dir,
             px_per_mm=None, cable_masks_dir=None, dlo_diameter_mm=6.5,
             max_ts_gap=0.025, csv_path=None,
             first_n_steps=None, last_n_steps=None,
             overlay_dir=None, overlay_w=None, overlay_h=None):

    # ── px/mm ─────────────────────────────────────────────────────────────────
    if px_per_mm is None and cable_masks_dir is not None:
        print(f"\n── Auto-computing px/mm ────────────────────────────")
        px_per_mm = compute_px_per_mm(cable_masks_dir, dlo_diameter_mm)

    # ── load data ─────────────────────────────────────────────────────────────
    gt_ts,      gt_nodes      = load_sequence(gt_dir)
    tracker_ts, tracker_nodes = load_sequence(tracker_dir)

    print(f"\n[INFO] GT       : {len(gt_ts):4d} entries  "
          f"{gt_ts[0]:.3f}s → {gt_ts[-1]:.3f}s")
    print(f"[INFO] Tracker  : {len(tracker_ts):4d} entries  "
          f"{tracker_ts[0]:.3f}s → {tracker_ts[-1]:.3f}s")

    # ── Build query_times = GT timestamps within the tracker's window ────────
    # Each GT timestamp is a moment where the IDS camera actually observed the
    # cable. Only those moments have real ground truth — every other instant
    # would be an interpolated GT, which contaminates the error with cable
    # motion the tracker is not wrong about.
    mask = (gt_ts >= tracker_ts[0]) & (gt_ts <= tracker_ts[-1])
    query_times = gt_ts[mask]

    if len(query_times) == 0:
        raise RuntimeError(
            "No GT timestamps fall inside the tracker's recording window — "
            "check clock alignment between GT and tracker streams."
        )

    n_excluded = int((~mask).sum())
    print(f"[INFO] Overlap  : {query_times[0]:.3f}s → {query_times[-1]:.3f}s")
    print(f"[INFO] Eval points: {len(query_times)} GT timestamps "
          f"(excluded {n_excluded} GT frames outside tracker window)")

    # ── apply window ──────────────────────────────────────────────────────────
    if first_n_steps is not None and last_n_steps is not None:
        print(f"[WARN] Both --first-n-steps and --last-n-steps given. "
              f"Using --first-n-steps only.")
        last_n_steps = None

    if first_n_steps is not None:
        query_times = query_times[:first_n_steps]
        print(f"[INFO] Evaluating FIRST {len(query_times)} GT timestamps  "
              f"({query_times[0]:.3f}s → {query_times[-1]:.3f}s)")
    elif last_n_steps is not None:
        query_times = query_times[-last_n_steps:]
        print(f"[INFO] Evaluating LAST {len(query_times)} GT timestamps  "
              f"({query_times[0]:.3f}s → {query_times[-1]:.3f}s)")
    else:
        print(f"[INFO] Evaluating ALL {len(query_times)} GT timestamps")

    # ── overlay output dir ────────────────────────────────────────────────────
    if overlay_dir is not None:
        # Auto-size canvas to the bbox of GT + tracker polylines if dims not
        # provided. Image-style coords (origin top-left) → just take max + margin.
        if overlay_w is None or overlay_h is None:
            all_pts = []
            for arr in list(gt_nodes) + list(tracker_nodes):
                a = np.asarray(arr, dtype=float)
                if a.ndim == 2 and a.shape[0] >= 1 and a.shape[1] >= 2:
                    all_pts.append(a[:, :2])
            if all_pts:
                stacked = np.concatenate(all_pts, axis=0)
                margin = 20
                auto_w = int(np.ceil(stacked[:, 0].max())) + margin
                auto_h = int(np.ceil(stacked[:, 1].max())) + margin
                overlay_w = overlay_w if overlay_w is not None else auto_w
                overlay_h = overlay_h if overlay_h is not None else auto_h
                print(f"[INFO] Overlay canvas auto-sized from data bbox "
                      f"(margin {margin} px)")
            else:
                overlay_w = overlay_w if overlay_w is not None else 1280
                overlay_h = overlay_h if overlay_h is not None else 720

        os.makedirs(overlay_dir, exist_ok=True)
        print(f"[INFO] Overlays  : writing per-step PNGs → {overlay_dir} "
              f"({overlay_w}×{overlay_h})")

    # ── evaluate ──────────────────────────────────────────────────────────────
    per_step = []
    skipped_tr_gap = 0
    skipped_nan    = 0
    printed_ranges = False

    for t in query_times:
        # GT lookup is exact: t IS a GT timestamp, so np.searchsorted gives
        # the matching index (zero temporal error on the GT side by construction).
        i_gt = int(np.searchsorted(gt_ts, t))
        if i_gt >= len(gt_ts) or gt_ts[i_gt] != t:
            # Defensive: if equality fails due to FP, fall back to nearest.
            i_gt = nearest_index(gt_ts, t)

        i_tr = nearest_index(tracker_ts, t)
        if abs(tracker_ts[i_tr] - t) > max_ts_gap:
            skipped_tr_gap += 1
            continue

        gt_n  = gt_nodes[i_gt]
        tr_n  = tracker_nodes[i_tr]

        if not printed_ranges:
            gt_arr = np.asarray(gt_n, dtype=float)
            tr_arr = np.asarray(tr_n, dtype=float)
            print(f"[INFO] GT      coord range: "
                  f"min={gt_arr.min(axis=0)}  max={gt_arr.max(axis=0)}  "
                  f"N={len(gt_arr)}")
            print(f"[INFO] Tracker coord range: "
                  f"min={tr_arr.min(axis=0)}  max={tr_arr.max(axis=0)}  "
                  f"N={len(tr_arr)}")
            print(f"[INFO] Overlay canvas    : {overlay_w} x {overlay_h}")
            printed_ranges = True

        e_arc = arc_length_error(gt_n, tr_n)
        e_ptc = point_to_curve_error(gt_n, tr_n)

        if np.isnan(e_arc) or np.isnan(e_ptc):
            skipped_nan += 1
            continue

        if overlay_dir is not None:
            save_overlay(
                overlay_dir, len(per_step), float(t),
                gt_n, tr_n, e_arc, e_ptc,
                px_per_mm, overlay_w, overlay_h,
            )

        per_step.append((t, e_arc, e_ptc, len(tr_n),
                         tracker_ts[i_tr] - t))   # signed tracker time error

    if not per_step:
        raise RuntimeError(
            f"No valid steps to evaluate "
            f"(tracker-gap skipped {skipped_tr_gap}, nan skipped {skipped_nan}). "
            f"Try increasing --max-ts-gap or check clock alignment."
        )

    print(f"\n[INFO] Valid steps: {len(per_step)} / {len(query_times)}")
    if skipped_tr_gap:
        print(f"[INFO]   Skipped {skipped_tr_gap} where nearest tracker frame "
              f"was > {max_ts_gap*1000:.0f} ms away")
    if skipped_nan:
        print(f"[INFO]   Skipped {skipped_nan} where GT or tracker polyline "
              f"was empty/degenerate")

    # ── Report tracker-side temporal residual (sanity check) ─────────────────
    tr_dt = np.abs(np.array([p[4] for p in per_step]))
    print(f"[INFO] Tracker-side temporal residual: "
          f"mean={tr_dt.mean()*1000:.2f} ms  max={tr_dt.max()*1000:.2f} ms")

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
            header = ["time_sec", "e_arc_px", "e_ptc_px", "M",
                      "tracker_dt_ms"]
            if px_per_mm:
                header += ["e_arc_mm", "e_ptc_mm"]
            w.writerow(header)
            for (t, e_arc, e_ptc, M, dt) in per_step:
                row = [f"{t:.6f}", f"{e_arc:.4f}", f"{e_ptc:.4f}", M,
                       f"{dt*1000:.3f}"]
                if px_per_mm:
                    row += [f"{e_arc/px_per_mm:.4f}",
                            f"{e_ptc/px_per_mm:.4f}"]
                w.writerow(row)
        print(f"\n  per-step CSV → {csv_path}")

    return per_step


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-node tracking error (eq. 23/24) + point-to-curve. "
                    "Queries are anchored at GT timestamps.")
    ap.add_argument("--gt-dir",           required=True)
    ap.add_argument("--tracker-dir",      required=True)
    ap.add_argument("--px-per-mm",        type=float, default=None)
    ap.add_argument("--cable-masks-dir",  default=None)
    ap.add_argument("--dlo-diameter-mm",  type=float, default=6.5)
    ap.add_argument("--max-ts-gap",       type=float, default=0.025,
                    help="Max allowed |tracker_ts - GT_ts| in seconds "
                         "(default 0.025 s = just over the tracker batch "
                         "period at ~20 ms)")
    ap.add_argument("--csv",              default=None)
    ap.add_argument("--first-n-steps",    type=int,   default=None,
                    help="Evaluate only first N GT timestamps")
    ap.add_argument("--last-n-steps",     type=int,   default=None,
                    help="Evaluate only last N GT timestamps")
    ap.add_argument("--overlay-dir",      default=None,
                    help="If set, write one PNG per evaluated step into this "
                         "folder, with GT (green) and tracker (yellow) polylines "
                         "drawn on a black canvas. Disabled if omitted.")
    ap.add_argument("--overlay-w",        type=int,   default=None,
                    help="Overlay canvas width in pixels. "
                         "Default: auto-sized from GT+tracker bbox.")
    ap.add_argument("--overlay-h",        type=int,   default=None,
                    help="Overlay canvas height in pixels. "
                         "Default: auto-sized from GT+tracker bbox.")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(
        gt_dir          = a.gt_dir,
        tracker_dir     = a.tracker_dir,
        px_per_mm       = a.px_per_mm,
        cable_masks_dir = a.cable_masks_dir,
        dlo_diameter_mm = a.dlo_diameter_mm,
        max_ts_gap      = a.max_ts_gap,
        csv_path        = a.csv,
        first_n_steps   = a.first_n_steps,
        last_n_steps    = a.last_n_steps,
        overlay_dir     = a.overlay_dir,
        overlay_w       = a.overlay_w,
        overlay_h       = a.overlay_h,
    )