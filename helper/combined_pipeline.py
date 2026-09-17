#!/usr/bin/env python3
"""
Combined Pipeline
=================
SAM3 runs first (segments the first RGB frame) to obtain the cable polyline
and wire diameter.  Then the event-based tracking loop starts, using the SAM3
output for CPD initialisation and adaptive morphology.

Usage
-----
    python combined_pipeline.py --show
    python combined_pipeline.py --show --disable-stc
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from skimage.measure import label as sk_label, regionprops

# ── project imports ───────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import EventBased.Filter as ev_filter
from EventBased.sae import sae
from EventBased.Visualization import Visualization
from EventBased.event_cpd import GeodesicCurveTracker, Params, mask_to_skeleton_obs, mask_to_pointcloud
from metavision_core.event_io import EventsIterator
import config

log = logging.getLogger(__name__)


def _keep_elongated(
    wire_bool: np.ndarray,
    min_aspect: float = 3.5,
    min_length: float = 80.0,
) -> np.ndarray:
   
    labeled = sk_label(wire_bool)
    out = np.zeros_like(wire_bool)
    for p in regionprops(labeled):
        if p.minor_axis_length > 0:
            aspect = p.major_axis_length / p.minor_axis_length
            if aspect >= min_aspect and p.major_axis_length >= min_length:
                out[labeled == p.label] = True
    return out


def adaptive_kernels(wire_diameter_px: float):
    """Derive morphological kernel sizes from wire diameter."""
    close_size = int(round(wire_diameter_px * 0.5)) | 1
    close_size = max(close_size, 5)

    open_size = int(round(wire_diameter_px * 0.15)) | 1
    open_size = max(open_size, 3)

    close_iters = max(1, int(round(wire_diameter_px / 25)))

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    return close_k, close_iters, open_k


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a (M,2) polyline to exactly n points by arc-length interpolation."""
    if len(pts) == n:
        return pts.copy()
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1))
    t_new = np.linspace(0.0, total, n)
    xs = np.interp(t_new, cum, pts[:, 0])
    ys = np.interp(t_new, cum, pts[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SAM3 INIT (runs in a subprocess to isolate CUDA from OpenCV GUI)
# ══════════════════════════════════════════════════════════════════════════════

# --- Worker script executed in subprocess ---
_SAM3_WORKER = r'''
import json, os, sys, time
import cv2, numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

manifest_path, frames_dir = sys.argv[1], sys.argv[2]
event_h, event_w, n_nodes  = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out_path                    = sys.argv[6]
H_path                      = sys.argv[7] if len(sys.argv) > 7 else ""
target_ts_us                = int(sys.argv[8]) if len(sys.argv) > 8 else -1

H = np.load(H_path) if H_path and os.path.isfile(H_path) else None

from FrameBased.Segmentation import SAM3Segmenter
from FrameBased.Polyline import FramePolyline

with open(manifest_path) as f:
    manifest = json.load(f)

entries = sorted(manifest.items(), key=lambda kv: kv[1])
if not entries:
    sys.exit(1)

# Pick frame: first one at or after target_ts_us, or the very first frame
if target_ts_us >= 0:
    import bisect
    ts_array = [e[1] for e in entries]
    idx = bisect.bisect_left(ts_array, target_ts_us)
    if idx >= len(entries):
        idx = len(entries) - 1
    fname, ts_us = entries[idx]
else:
    fname, ts_us = entries[0]

frame_path = os.path.join(frames_dir, fname)
if not os.path.isfile(frame_path):
    sys.exit(1)

print(f"[SAM3-sub] Segmenting {fname} (t={ts_us/1e6:.3f}s)", flush=True)
segmenter = SAM3Segmenter()
t0 = time.perf_counter()
bgr = cv2.imread(frame_path)
if bgr is None:
    sys.exit(1)

mask = segmenter.segment(bgr)
print(f"[SAM3-sub] Segmentation took {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)

if int((mask > 0).sum()) < 50:
    sys.exit(1)

rgb_h, rgb_w = mask.shape[:2]
polylines = FramePolyline.extract(mask, num_nodes=n_nodes, min_length=80)
if not polylines:
    sys.exit(1)

best_rgb = max(polylines, key=lambda Y: FramePolyline.arc_length(Y))

pts = best_rgb.astype(np.float32).reshape(-1, 1, 2)
if H is not None:
    pts_ev = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
else:
    sx, sy = event_w / rgb_w, event_h / rgb_h
    pts_ev = best_rgb.astype(np.float32) * np.array([sx, sy], np.float32)

pts_ev[:, 0] = np.clip(pts_ev[:, 0], 0, event_w - 1)
pts_ev[:, 1] = np.clip(pts_ev[:, 1], 0, event_h - 1)

# resample
diffs = np.diff(pts_ev, axis=0)
seg_lens = np.hypot(diffs[:,0], diffs[:,1])
cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
total = cum[-1]
if total == 0:
    polyline = np.tile(pts_ev[0], (n_nodes, 1))
else:
    t_new = np.linspace(0.0, total, n_nodes)
    xs = np.interp(t_new, cum, pts_ev[:,0])
    ys = np.interp(t_new, cum, pts_ev[:,1])
    polyline = np.column_stack([xs, ys]).astype(np.float32)

# wire diameter
arc_rgb = float(FramePolyline.arc_length(best_rgb))
arc_ev  = float(np.sum(np.linalg.norm(np.diff(pts_ev, axis=0), axis=1)))
arc_scale = arc_ev / max(arc_rgb, 1.0)

wire_diameter_ev = 0.0
dist_t = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
interior = dist_t[dist_t > 0]
if len(interior) > 10:
    rgb_diameter = float(np.percentile(interior, 75)) * 2.0
    wire_diameter_ev = rgb_diameter * arc_scale
    print(f"[SAM3-sub] diameter {rgb_diameter:.1f}px (rgb) -> {wire_diameter_ev:.1f}px (event)", flush=True)

cv2.imwrite(os.path.join(os.path.dirname(frames_dir), "sam3_mask.png"), mask)
np.savez(out_path, polyline=polyline, wire_diameter=wire_diameter_ev)
print("[SAM3-sub] done", flush=True)
'''


def run_sam3_init(
    manifest_path: str,
    frames_dir: str,
    event_h: int,
    event_w: int,
    H: Optional[np.ndarray],
    n_nodes: int,
    target_ts_us: int = -1,
) -> Optional[tuple[np.ndarray, float]]:

    import subprocess, tempfile

    # Write worker script to a temp file next to this script (so imports work)
    worker_path = os.path.join(_SRC, "_sam3_worker_tmp.py")
    with open(worker_path, "w") as f:
        f.write(_SAM3_WORKER)

    out_fd, out_path = tempfile.mkstemp(suffix=".npz")
    os.close(out_fd)

    # Save homography to temp file if present
    h_path = ""
    if H is not None:
        h_fd, h_path = tempfile.mkstemp(suffix=".npy")
        os.close(h_fd)
        np.save(h_path, H)

    cmd = [
        sys.executable, worker_path,
        manifest_path, frames_dir,
        str(event_h), str(event_w), str(n_nodes),
        out_path,
    ]
    if h_path:
        cmd.append(h_path)
    else:
        cmd.append("")  # placeholder so target_ts_us lands in argv[8]
    cmd.append(str(target_ts_us))

    log.info("[SAM3] Launching subprocess …")
    try:
        proc = subprocess.run(cmd, timeout=120, capture_output=True, text=True)
        # Forward subprocess output
        for line in proc.stdout.strip().splitlines():
            log.info(line)
        if proc.returncode != 0:
            if proc.stderr:
                log.error("[SAM3] stderr: %s", proc.stderr[-500:])
            log.warning("[SAM3] Subprocess failed (rc=%d)", proc.returncode)
            return None

        data = np.load(out_path)
        polyline = data["polyline"]
        wire_diameter = float(data["wire_diameter"])
        return polyline, wire_diameter
    except subprocess.TimeoutExpired:
        log.error("[SAM3] Subprocess timed out")
        return None
    except Exception as e:
        log.error("[SAM3] Subprocess error — %s", e)
        return None
    finally:
        for p in (worker_path, out_path, h_path):
            if p and os.path.isfile(p):
                os.remove(p)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Default paths — override via CLI or edit here
RAW_PATH = "/path/to/Dataset/Single_Cable/Green_Cable/Speed_100/1/events.raw"
FRAMES_DIR = "/path/to/Dataset/Single_Cable/Green_Cable/Speed_100/1/frames"
MANIFEST_PATH = "/path/to/Dataset/Single_Cable/Green_Cable/Speed_100/1/manifest.json"
HOMOGRAPHY_PATH = "/path/to/calibration/event_to_rgb_H.npy"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = config.parse_args()

    # ── Paths (from CLI --input-path or defaults) ─────────────────────────────
    raw_path = args.input_path if args.input_path else RAW_PATH
    frames_dir = FRAMES_DIR
    manifest_path = MANIFEST_PATH
    homography_path = HOMOGRAPHY_PATH

    # ── Load homography ───────────────────────────────────────────────────────
    H: Optional[np.ndarray] = None
    if os.path.isfile(homography_path):
        H = np.load(homography_path)
        log.info("Homography loaded from %s", homography_path)

    # ── Open event stream ─────────────────────────────────────────────────────
    it = EventsIterator(
        input_path=raw_path,
        delta_t=args.delta_t_us,
        mode="delta_t",
    )
    h, w = it.get_size()
    log.info("Event sensor: %d × %d", w, h)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — Run SAM3 
    # ══════════════════════════════════════════════════════════════════════════
    wire_d = 50  # default wire diameter

    if os.path.isfile(manifest_path) and os.path.isdir(frames_dir):
        result = run_sam3_init(
            manifest_path=manifest_path,
            frames_dir=frames_dir,
            event_h=h,
            event_w=w,
            H=H,
            n_nodes=args.spline_samples,
        )
        if result is not None:
            _, wire_d_sam3 = result
            if wire_d_sam3 > 0:
                wire_d = wire_d_sam3 
            log.info("SAM3 init done — wire_d=%.1fpx", wire_d)
        else:
            log.warning("SAM3 init failed — falling back to event-only")
    else:
        log.info("No manifest/frames_dir — running event-only")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — Event loop
    # ══════════════════════════════════════════════════════════════════════════

    # ── Event filters (STC) ───────────────────────────────────────────────────
    trail, trail_out, afk, afk_out, stc_filter, stc_out = ev_filter.filter_events(it, w, h, args)

    # ── Polarity-separated time surfaces ──────────────────────────────────────
    T_on = np.full((h, w), -10**18, dtype=np.int64)
    T_off = np.full((h, w), -10**18, dtype=np.int64)

    # ── CPD tracker ───────────────────────────────────────────────────────────
    tracker = GeodesicCurveTracker(Params(
        num_nodes=args.spline_samples,
        beta=20.0,
        lam=0.0001,
        mu=0.1,
        k_vis=0.05,
        max_iter=30,
        tol=1e-4,
        beta_pre_proc=1.0,
        lam_pre_proc=0.0001,
        prune_threshold=300.0,
    ))

    # ── Velocity warm-start state ─────────────────────────────────────────────
    _prev_centroid = None
    _prev_t = None
    _velocity = np.zeros(2)  # (vx, vy) in px/µs

    # ── Precompute morphological kernels  ──
    close_k, close_iters, open_k = adaptive_kernels(wire_d)

    # ── Hibernation state ─────────────────────────────────────────────────────
    hibernate_timeout_us = int(args.hibernate_timeout_ms * 1000)
    hibernate_event_frac = args.hibernate_event_frac
    total_pixels = h * w
    hibernate_thr = int(total_pixels * hibernate_event_frac)
    _quiet_since_us: Optional[int] = None   # first timestamp of current quiet period
    _hibernating = False
    _last_polyline: Optional[np.ndarray] = None  # kept during hibernation
    has_manifest = os.path.isfile(manifest_path) and os.path.isdir(frames_dir)

    log.info("Event loop started (hibernate after %dms quiet, frac=%.3f → thr=%d px).",
             args.hibernate_timeout_ms, hibernate_event_frac, hibernate_thr)

    if args.show:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cpd", cv2.WINDOW_NORMAL)

    # ── Timing accumulators ─────────────────────────────────────────────────
    _t_filter, _t_sae, _t_recentness = [], [], []
    _t_wiremask, _t_elongated, _t_obs = [], [], []
    _t_cpd, _t_vis, _t_iter = [], [], []

    try:
        for evs in it:
            # Pump GUI every iteration so windows stay responsive
            if args.show and (cv2.waitKey(1) & 0xFF) == 27:
                break

            if len(evs) == 0:
                continue

            t_iter_start = time.perf_counter()

            # ── Filter (STC only) ──────────────────────────────────────────
            t0 = time.perf_counter()
            ev = evs
            if stc_filter is not None:
                stc_filter.process_events(ev, stc_out)
                ev = stc_out.numpy()
            dt_filter_ms = (time.perf_counter() - t0) * 1000
            if len(ev) == 0:
                continue

            xs = ev["x"].astype(np.int32, copy=False)
            ys = ev["y"].astype(np.int32, copy=False)
            ts = ev["t"].astype(np.int64, copy=False)
            ps = ev["p"]
            t_now = int(ts[-1])

            on = ps if ps.dtype == np.bool_ else (ps > 0)

            # ──Update SAE  ───────────────
            t0 = time.perf_counter()
            if np.any(on):
                T_on[ys[on], xs[on]] = ts[on]
            if np.any(~on):
                T_off[ys[~on], xs[~on]] = ts[~on]
            dt_sae_ms = (time.perf_counter() - t0) * 1000

            dt_batch_us = int(ts[-1] - ts[0])
            if dt_batch_us <= 0:
                continue
            event_rate = len(ev) / (dt_batch_us * 1e-6)
            tau_us_dyn = ev_filter.adaptive_tau_us(event_rate, tau_min=800, tau_max=10000)

            # ── Recentness map ─────────────────────────────────────────────
            t0 = time.perf_counter()
            R = sae.compute_recentness(T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn)
            dt_recentness_ms = (time.perf_counter() - t0) * 1000

            # ── Wire mask (Otsu + adaptive morphology) ─────────────────────
            t0 = time.perf_counter()
            R_gated = R
            active = R_gated[R_gated > 0]
            if len(active) == 0:
                # Count as quiet
                if tracker.Y is not None and not _hibernating:
                    if _quiet_since_us is None:
                        _quiet_since_us = t_now
                continue

            # Otsu on active pixels only 
            active_u8 = np.clip(active * 255, 0, 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(active_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            wire_thr = max(float(otsu_val) / 255.0, 0.05)

            # Bounding box of active pixels — run morphology on crop only
            act_ys, act_xs = np.where(R_gated > 0)
            pad = max(close_k.shape[0], open_k.shape[0])  # pad by kernel size
            y0 = max(int(act_ys.min()) - pad, 0)
            y1 = min(int(act_ys.max()) + pad + 1, R_gated.shape[0])
            x0 = max(int(act_xs.min()) - pad, 0)
            x1 = min(int(act_xs.max()) + pad + 1, R_gated.shape[1])

            crop = (R_gated[y0:y1, x0:x1] >= wire_thr).astype(np.uint8)
            crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, close_k, iterations=close_iters)
            crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, open_k, iterations=1)

            wire_mask = np.zeros(R_gated.shape, dtype=bool)
            wire_mask[y0:y1, x0:x1] = crop.astype(bool)
            dt_wiremask_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            wire_mask = _keep_elongated(wire_mask, min_aspect=3.5)
            dt_elongated_ms = (time.perf_counter() - t0) * 1000

            mask_count = int(wire_mask.sum())

            # ──  Hibernation detection ─────────────────────────────────────
            is_quiet = mask_count < hibernate_thr

            if is_quiet and tracker.Y is not None and not _hibernating:
                if _quiet_since_us is None:
                    _quiet_since_us = t_now
                elif (t_now - _quiet_since_us) >= hibernate_timeout_us:
                    # Enter hibernation
                    _hibernating = True
                    _last_polyline = tracker.Y.copy()
                    log.info("[Hibernate] Wire quiet for %.0fms at t=%.3fs — hibernating. "
                             "Keeping last polyline.",
                             (t_now - _quiet_since_us) / 1000.0, t_now / 1e6)

                    # Re-run SAM3 on frame closest to hibernation time
                    if has_manifest:
                        log.info("[Hibernate] Re-running SAM3 …")
                        result = run_sam3_init(
                            manifest_path=manifest_path,
                            frames_dir=frames_dir,
                            event_h=h,
                            event_w=w,
                            H=H,
                            n_nodes=args.spline_samples,
                            target_ts_us=t_now,
                        )
                        if result is not None:
                            _, wire_d_sam3 = result
                            if wire_d_sam3 > 0:
                                wire_d = wire_d_sam3 
                                close_k, close_iters, open_k = adaptive_kernels(wire_d)
                            log.info("[Hibernate] SAM3 done — wire_d=%.1fpx", wire_d)
                        else:
                            log.warning("[Hibernate] SAM3 failed — keeping previous wire_d")
                continue  # skip tracking while quiet (whether hibernating or counting down)

            if not is_quiet:
                if _hibernating:
                    # Wake up: reset tracker for fresh init
                    log.info("[Hibernate] Wire active again at t=%.3fs — waking up, "
                             "re-initialising tracker.", t_now / 1e6)
                    tracker = GeodesicCurveTracker(Params(
                        num_nodes=args.spline_samples,
                        beta=20.0,
                        lam=0.0001,
                        mu=0.1,
                        k_vis=0.05,
                        max_iter=30,
                        tol=1e-4,
                        beta_pre_proc=1.0,
                        lam_pre_proc=0.0001,
                        prune_threshold=300.0,
                    ))
                    _prev_centroid = None
                    _prev_t = None
                    _velocity = np.zeros(2)
                    _hibernating = False
                    _last_polyline = None
                _quiet_since_us = None

            if mask_count < 30 or mask_count > wire_mask.size * 0.3:
                continue

            # ── Observations and init path ─────────────────────────────────
            t0 = time.perf_counter()
            if tracker.Y is None:
                # First init: need skeleton for ordered init_path
                _, init_path = mask_to_skeleton_obs(wire_mask, max_pts=300)
                obs = mask_to_pointcloud(wire_mask, max_pts=500)
                if obs is None:
                    continue
                # Trust-region filtering: keep obs near skeleton
                if init_path is not None and len(obs) > 0:
                    dists = np.min(np.linalg.norm(obs[:, None, :] - init_path[None, :, :], axis=2), axis=1)
                    obs_gated = obs[dists < 30.0]
                    if len(obs_gated) >= 10:
                        obs = obs_gated
            else:
                # Tracker already running:
                init_path = None
                obs = mask_to_pointcloud(wire_mask, max_pts=500)
                if obs is None:
                    continue
            dt_obs_ms = (time.perf_counter() - t0) * 1000

            # ── Velocity warm-start ────────────────────────────────────────
            if tracker.Y is not None and _prev_centroid is not None and _prev_t is not None:
                dt = t_now - _prev_t
                if dt > 0:
                    cur_centroid = tracker.Y.mean(axis=0)
                    _velocity = (cur_centroid - _prev_centroid) / dt
            predicted_shift = _velocity * (t_now - _prev_t) if _prev_t is not None else np.zeros(2)
            if init_path is not None and np.linalg.norm(predicted_shift) > 0:
                init_path = init_path + predicted_shift[None, :]

            # ── CPD tracking step ──────────────────────────────────────────
            t0 = time.perf_counter()
            tracker.track(obs, init_path=init_path)
            dt_cpd_ms = (time.perf_counter() - t0) * 1000

            if tracker.Y is not None:
                _prev_centroid = tracker.Y.mean(axis=0)
                _prev_t = t_now

            # ── Visualise ──────────────────────────────────────────────────
            t0 = time.perf_counter()
            if args.show:
                Visualization.visualize_wire(R, mask=wire_mask, skel=None, win="debug", scale=1.0)

                frame = cv2.cvtColor(
                    np.clip(R * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
                )
                if obs is not None:
                    for pt in obs.astype(np.int32):
                        cv2.circle(frame, (pt[0], pt[1]), 1, (0, 200, 0), -1)
                if init_path is not None and len(init_path) >= 2:
                    cv2.polylines(
                        frame,
                        [init_path.astype(np.int32).reshape(-1, 1, 2)],
                        False, (255, 100, 0), 1, cv2.LINE_AA,
                    )
               
                show_poly = tracker.Y if tracker.Y is not None else _last_polyline
                if show_poly is not None:
                    polyline_cv = show_poly.astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [polyline_cv], False, (0, 255, 255), 2, cv2.LINE_AA)
                    for pt in show_poly.astype(np.int32):
                        cv2.circle(frame, (pt[0], pt[1]), 3, (0, 128, 255), -1)
                cv2.imshow("cpd", frame)
            dt_vis_ms = (time.perf_counter() - t0) * 1000

            # ── Timing report ─────────────────────────────────────────────────
            dt_iter_ms = (time.perf_counter() - t_iter_start) * 1000
            _t_filter.append(dt_filter_ms)
            _t_sae.append(dt_sae_ms)
            _t_recentness.append(dt_recentness_ms)
            _t_wiremask.append(dt_wiremask_ms)
            _t_elongated.append(dt_elongated_ms)
            _t_obs.append(dt_obs_ms)
            _t_cpd.append(dt_cpd_ms)
            _t_vis.append(dt_vis_ms)
            _t_iter.append(dt_iter_ms)

            log.info("[Timing] iter=%.1fms | filter=%.1f sae=%.1f recentness=%.1f "
                     "wiremask=%.1f elongated=%.1f obs=%.1f cpd=%.1f vis=%.1f | "
                     "obs=%d mask=%d",
                     dt_iter_ms, dt_filter_ms, dt_sae_ms, dt_recentness_ms,
                     dt_wiremask_ms, dt_elongated_ms, dt_obs_ms, dt_cpd_ms, dt_vis_ms,
                     len(obs) if obs is not None else 0, mask_count)

    finally:
        cv2.destroyAllWindows()

        # ── Average timing summary ────────────────────────────────────────────
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        n = len(_t_iter)
        if n > 0:
            log.info("═" * 70)
            log.info("[Avg Timing] %d iterations", n)
            log.info("  iter      = %.2fms", _avg(_t_iter))
            log.info("  filter    = %.2fms", _avg(_t_filter))
            log.info("  sae       = %.2fms", _avg(_t_sae))
            log.info("  recentness= %.2fms", _avg(_t_recentness))
            log.info("  wiremask  = %.2fms", _avg(_t_wiremask))
            log.info("  elongated = %.2fms", _avg(_t_elongated))
            log.info("  obs       = %.2fms", _avg(_t_obs))
            log.info("  cpd       = %.2fms", _avg(_t_cpd))
            log.info("  vis       = %.2fms", _avg(_t_vis))
            log.info("═" * 70)

        log.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
