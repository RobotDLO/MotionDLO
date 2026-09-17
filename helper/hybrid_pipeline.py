#!/usr/bin/env python3
"""
Hybrid Cable Tracker
====================
Implements the dual-path architecture 

Modes
-----
  Offline (default): reads events from .raw file + SAM3 from manifest.json + frames/
  Live:              reads events from Prophesee camera + RGB from IDS Peak or webcam

Usage
-----
    # Offline (existing recording):
    python hybrid_pipeline.py --raw /path/to/events.raw \
                              --frames-dir /path/to/frames \
                              --manifest  /path/to/manifest.json \
                              --show

    # Offline, event-only (no SAM3 corrections):
    python hybrid_pipeline.py --raw /path/to/events.raw --show

    # Offline with homography for accurate RGB→event mapping:
    python hybrid_pipeline.py --raw /path/to/events.raw \
                              --frames-dir /path/to/frames \
                              --manifest  /path/to/manifest.json \
                              --homography /path/to/rgb_to_event_H.npy \
                              --show

Configuration
-------------
Edit the DEFAULTS section below, or pass CLI args.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import EventBased.Filter as ev_filter
from EventBased.sae import sae
from EventBased.Visualization import Visualization
from EventBased.event_cpd import GeodesicCurveTracker, Params, mask_to_skeleton_obs
from shared_state import SharedState, CorrectionSignal
from metavision_core.event_io import EventsIterator

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  DEFAULTS  — edit here or override with CLI args
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = dict(
    # ── paths ──────────────────────────────────────────────────────────────
    raw         = "/path/to/Sam3_cables/recordings/run_006/events.raw",
    frames_dir  = "/path/to/Sam3_cables/recordings/run_006/frames",
    manifest    = "/path/to/Sam3_cables/recordings/run_006/manifest.json",
    homography  = "/path/to/calibration/rgb_to_event_H.npy",

    # ── event processing ───────────────────────────────────────────────────
    delta_t_us      = 5000,
    tau_min_us      = 10000,
    tau_max_us      = 50000,
    trail_us        = 8000,
    disable_trail   = False,
    afk_min_freq    = 90.0,
    afk_max_freq    = 110.0,
    afk_filter_length    = 7,
    afk_diff_thresh_us   = 1500,
    disable_afk     = False,
    disable_stc     = False,
    stc_filter_thr  = 4000,
    stc_cut_trail   = True,
    spline_samples  = 60,

    # ── wire mask ──────────────────────────────────────────────────────────
    wire_thr_frac   = 0.65,      # threshold = max(R) * wire_thr_frac
    mask_min_px     = 30,
    mask_max_frac   = 0.30,
    min_aspect      = 1.5,       # cable is elongated (aspect >= 1.5)
    min_length_px   = 20.0,      # cable is long (major axis >= 20 px)
    near_weight     = 0.7,       # spatial-proximity weight once tracking starts

    # ── CPD tracker ────────────────────────────────────────────────────────
    cpd_beta        = 20.0,
    cpd_lam         = 0.0001,
    cpd_mu          = 0.1,
    cpd_max_iter    = 30,
    cpd_tol         = 1e-4,
    cpd_prune_thr   = 300.0,
    roi_radius_px   = 80,        # circle radius around each tracker node for ROI
    roi_dil_px      = 60,        # dilation of ROI before gating R
    init_min_len    = 200,       # min end-to-end span (px) to accept at first init

    # ── ROI loss handling ──────────────────────────────────────────────────
    empty_expand    = 8,         # expand ROI after this many empty frames
    empty_reset     = 24,        # full-frame fallback after this many empty frames

    # ── SAM3 correction ────────────────────────────────────────────────────
    blend_tau_ms            = 2000.0, # exponential decay time constant for correction weight
    correction_max_age_ms   = 10000.0, # hard cutoff: discard corrections older than this
    init_timeout_s          = 3.0,   # wait this long for SAM3/long-cable before event fallback

    # ── timing ─────────────────────────────────────────────────────────────
    skip_s          = 0.0,       # skip this many seconds at start (fast-forward past static period)

    # ── display ────────────────────────────────────────────────────────────
    show            = False,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _best_cable_component(
    wire_bool:   np.ndarray,
    min_aspect:  float = 4.0,
    min_length:  float = 80.0,
    near_pt:     Optional[np.ndarray] = None,
    near_weight: float = 0.7,
) -> np.ndarray:
   
    mask_u8 = wire_bool.view(np.uint8) if wire_bool.dtype == bool else wire_bool.astype(np.uint8)
    n_labels, labeled, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    candidates = []
    for i in range(1, n_labels):          # skip background (label 0)
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        major = float(max(w, h))
        minor = float(min(w, h))
        if minor > 0 and major / minor >= min_aspect and major >= min_length:
            cx, cy = centroids[i]         # (x, y) already
            candidates.append((i, major, cx, cy))

    if not candidates:
        return np.zeros_like(wire_bool)

    max_len = max(c[1] for c in candidates) or 1.0
    best_label, best_score = None, -1.0
    for label, length, cx, cy in candidates:
        norm_len = length / max_len
        if near_pt is not None:
            dist  = np.hypot(cx - near_pt[0], cy - near_pt[1])
            prox  = 1.0 / (1.0 + dist * 0.01)
            score = (1.0 - near_weight) * norm_len + near_weight * prox
        else:
            score = norm_len
        if score > best_score:
            best_score = score
            best_label = label

    out = np.zeros_like(wire_bool)
    if best_label is not None:
        out[labeled == best_label] = True
    return out


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


def _align_polyline_to_events(
    polyline: np.ndarray,
    R: np.ndarray,
    r_threshold: float = 0.15,
    window: int = 8,
) -> np.ndarray:
   
    ev_ys, ev_xs = np.where(R > r_threshold)
    if len(ev_xs) < 20:
        return polyline   # not enough events to align

    ev_cx = float(ev_xs.mean())
    ev_cy = float(ev_ys.mean())
    ev_centroid = np.array([ev_cx, ev_cy])

    M = len(polyline)
    w = min(window, M)

    # Slide a window of `w` nodes along the polyline; find the window whose
    # centroid is closest to the event centroid
    best_offset = np.zeros(2, dtype=np.float32)
    best_dist   = np.inf
    for i in range(M - w + 1):
        seg_centroid = polyline[i:i + w].mean(axis=0)
        d = float(np.linalg.norm(seg_centroid - ev_centroid))
        if d < best_dist:
            best_dist   = d
            best_offset = ev_centroid - seg_centroid

    log.info("[Align] Nearest SAM3 segment dist=%.0fpx → translating by (%.0f, %.0f)px",
             best_dist, best_offset[0], best_offset[1])
    return (polyline + best_offset).astype(np.float32)


def _mask_to_polyline(
    mask_rgb_space: np.ndarray,
    event_h: int,
    event_w: int,
    H: Optional[np.ndarray],
    n_nodes: int,
) -> Optional[tuple]:
 
    if int((mask_rgb_space > 0).sum()) < 50:
        return None

    from FrameBased.Polyline import FramePolyline

    rgb_h, rgb_w = mask_rgb_space.shape[:2]

    # ── Extract polylines in RGB space — full pipeline ─────────────────────
    polylines = FramePolyline.extract(mask_rgb_space, num_nodes=n_nodes, min_length=80)
    if not polylines:
        return None

    best_rgb = max(polylines, key=lambda Y: FramePolyline.arc_length(Y))
    arc_rgb  = FramePolyline.arc_length(best_rgb)

    # ── Warp polyline points RGB → event space ─────────────────────────────
    pts = best_rgb.astype(np.float32).reshape(-1, 1, 2)
    if H is not None:
        pts_ev = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    else:
        sx = event_w / rgb_w
        sy = event_h / rgb_h
        pts_ev = best_rgb.astype(np.float32) * np.array([sx, sy], dtype=np.float32)

    pts_ev[:, 0] = np.clip(pts_ev[:, 0], 0, event_w - 1)
    pts_ev[:, 1] = np.clip(pts_ev[:, 1], 0, event_h - 1)

    arc_ev    = float(np.sum(np.linalg.norm(np.diff(pts_ev, axis=0), axis=1)))
    arc_scale = arc_ev / max(arc_rgb, 1.0)

    return _resample_polyline(pts_ev, n_nodes), arc_scale


# ══════════════════════════════════════════════════════════════════════════════
#  SAM3 CORRECTION THREAD
# ══════════════════════════════════════════════════════════════════════════════

class SAM3Thread(threading.Thread):

    def __init__(
        self,
        manifest_path:    str,
        frames_dir:       str,
        shared_state:     SharedState,
        event_h:          int,
        event_w:          int,
        H:                Optional[np.ndarray],
        n_nodes:          int,
        stop_event:       Optional[threading.Event] = None,
    ):
        super().__init__(daemon=True, name="SAM3Thread")
        self.manifest_path = manifest_path
        self.frames_dir    = frames_dir
        self.shared_state  = shared_state
        self.event_h       = event_h
        self.event_w       = event_w
        self.H             = H
        self.n_nodes       = n_nodes
        self._stop         = stop_event or threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):

        try:
            from FrameBased.Segmentation import SAM3Segmenter
        except ImportError as e:
            log.error("SAM3Thread: cannot import SAM3Segmenter — %s", e)
            log.error("SAM3 corrections disabled.")
            return

        try:
            with open(self.manifest_path) as f:
                manifest: dict = json.load(f)
        except Exception as e:
            log.error("SAM3Thread: cannot read manifest %s — %s", self.manifest_path, e)
            return

        # Sort by event timestamp (ascending) 
        entries = sorted(manifest.items(), key=lambda kv: kv[1])   # [(fname, ts_us), ...]
        ts_array = [e[1] for e in entries]
        if not entries:
            log.warning("SAM3Thread: manifest is empty.")
            return

        log.info("SAM3Thread: %d frames in manifest (ts %.1fs – %.1fs)",
                 len(entries), ts_array[0] / 1e6, ts_array[-1] / 1e6)
        log.info("SAM3Thread: waiting for event loop to detect cable activity…")

        # ── Block until event loop signals meaningful cable activity ──────────
        trigger_us = self.shared_state.wait_for_init_request(timeout=120.0)
        if trigger_us is None or self._stop.is_set():
            log.warning("SAM3Thread: no init trigger received — aborting.")
            return

        log.info("SAM3Thread: triggered at event t=%.3fs — finding closest frame…",
                 trigger_us / 1e6)

        # ── Binary search: find manifest entry closest to trigger_us ──────────
        import bisect
        idx = bisect.bisect_left(ts_array, trigger_us)
        if idx == 0:
            closest_idx = 0
        elif idx >= len(ts_array):
            closest_idx = len(ts_array) - 1
        else:
            # Pick whichever neighbour is closer in time
            closest_idx = idx if abs(ts_array[idx] - trigger_us) <= abs(ts_array[idx-1] - trigger_us) else idx - 1

        fname, ts_us = entries[closest_idx]
        log.info("SAM3Thread: closest frame = %s  Δt=%.0fms",
                 fname, abs(ts_us - trigger_us) / 1000.0)

        frame_path = os.path.join(self.frames_dir, fname)
        if not os.path.isfile(frame_path):
            log.error("SAM3Thread: frame not found — %s", frame_path)
            return

        segmenter = SAM3Segmenter()

        # ── Run SAM3 on the temporally-synchronized frame ─────────────────────
        t0  = time.perf_counter()
        bgr = cv2.imread(frame_path)
        if bgr is None:
            log.error("SAM3Thread: cannot read %s", frame_path)
            return

        try:
            mask = segmenter.segment(bgr)   # (H_rgb, W_rgb) uint8 0/255
        except Exception as e:
            log.error("SAM3Thread: SAM3 failed on %s — %s", fname, e)
            return

        dt_sam3 = (time.perf_counter() - t0) * 1000

        # Save mask + warped for visual inspection
        _mask_dir = os.path.join(os.path.dirname(self.frames_dir), "sam3_masks")
        os.makedirs(_mask_dir, exist_ok=True)
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(_mask_dir, f"{stem}_mask.png"), mask)

        if self.H is not None:
            warped = cv2.warpPerspective(mask, self.H, (self.event_w, self.event_h),
                                         flags=cv2.INTER_NEAREST)
        else:
            warped = cv2.resize(mask, (self.event_w, self.event_h),
                                interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(_mask_dir, f"{stem}_warped.png"), warped)

        if int((mask > 0).sum()) == 0:
            log.warning("SAM3Thread: empty mask for %s — no cable found", fname)
            return

        # ── Convert mask to event-space polyline ──────────────────────────────
        result = _mask_to_polyline(
            mask, self.event_h, self.event_w, self.H, self.n_nodes
        )
        if result is None:
            log.warning("SAM3Thread: could not extract polyline for %s", fname)
            return
        polyline, arc_scale = result

        cable_px   = int((mask > 0).sum())
        total_px   = mask.shape[0] * mask.shape[1]
        confidence = min(cable_px / max(total_px * 0.001, 1), 1.0)


        wire_diameter_ev = 0.0
        dist_t = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        interior = dist_t[dist_t > 0]
        if len(interior) > 10:
            rgb_diameter = float(np.percentile(interior, 75)) * 2.0
            wire_diameter_ev = rgb_diameter * arc_scale
            log.info("SAM3Thread: cable diameter %.1f px (rgb) → %.1f px (event space)  "
                     "arc_scale=%.3f", rgb_diameter, wire_diameter_ev, arc_scale)

        # Use trigger_us as the signal timestamp so age=0 in the event loop
        signal = CorrectionSignal(
            polyline         = polyline,
            confidence       = confidence,
            timestamp_us     = trigger_us,
            wire_diameter_px = wire_diameter_ev,
        )
        self.shared_state.post_correction(signal)
        log.info(
            "SAM3Thread: posted init correction  conf=%.2f  sam3=%.0fms  [%s]",
            confidence, dt_sam3, fname,
        )
        log.info("SAM3Thread: done (single-shot init).")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── Load homography if provided ────────────────────────────────────────────
    H: Optional[np.ndarray] = None
    if cfg["homography"] and os.path.isfile(cfg["homography"]):
        H = np.load(cfg["homography"])
        log.info("Homography loaded from %s", cfg["homography"])
    elif cfg["homography"]:
        log.warning("Homography file not found: %s — falling back to resize", cfg["homography"])

    # ── Open event stream ──────────────────────────────────────────────────────
    it = EventsIterator(
        input_path=cfg["raw"],
        delta_t=cfg["delta_t_us"],
        mode="delta_t",
    )
    ev_h, ev_w = it.get_size()   # returns (height, width) → ev_h=720, ev_w=1280
    log.info("Event sensor: %d × %d", ev_w, ev_h)

    # ── Event filters ──────────────────────────────────────────────────────────
    class _Args:
        disable_trail      = cfg["disable_trail"]
        trail_us           = cfg["trail_us"]
        disable_afk        = cfg["disable_afk"]
        afk_min_freq       = cfg["afk_min_freq"]
        afk_max_freq       = cfg["afk_max_freq"]
        afk_filter_length  = cfg["afk_filter_length"]
        afk_diff_thresh_us = cfg["afk_diff_thresh_us"]
        disable_stc        = cfg["disable_stc"]
        stc_filter_thr     = cfg["stc_filter_thr"]
        stc_cut_trail      = cfg["stc_cut_trail"]

    trail, trail_out, afk, afk_out, stc_filter, stc_out = ev_filter.filter_events(it, ev_w, ev_h, _Args())

    # ── CPD tracker ────────────────────────────────────────────────────────────
    tracker = GeodesicCurveTracker(Params(
        num_nodes       = cfg["spline_samples"],
        beta            = cfg["cpd_beta"],
        lam             = cfg["cpd_lam"],
        mu              = cfg["cpd_mu"],
        k_vis           = 0.05,
        max_iter        = cfg["cpd_max_iter"],
        tol             = cfg["cpd_tol"],
        beta_pre_proc   = 1.0,
        lam_pre_proc    = 0.0001,
        prune_threshold = cfg["cpd_prune_thr"],
    ))

    # ── Shared state (corrections from SAM3Thread) ─────────────────────────────
    shared = SharedState(max_pending=5)
    stop_event = threading.Event()

    # ── SAM3 correction thread ─────────────────────────────────────────────────
    sam3_thread: Optional[SAM3Thread] = None
    if cfg["manifest"] and cfg["frames_dir"]:
        if os.path.isfile(cfg["manifest"]) and os.path.isdir(cfg["frames_dir"]):
            sam3_thread = SAM3Thread(
                manifest_path = cfg["manifest"],
                frames_dir    = cfg["frames_dir"],
                shared_state  = shared,
                event_h       = ev_h,
                event_w       = ev_w,
                H             = H,
                n_nodes       = cfg["spline_samples"],
                stop_event    = stop_event,
            )
            sam3_thread.start()
            log.info("SAM3Thread started (manifest: %s)", cfg["manifest"])
        else:
            log.warning("manifest or frames_dir missing — running event-only")
    else:
        log.info("No manifest/frames_dir — running event-only (no SAM3 corrections)")

    # ── ROI + state ────────────────────────────────────────────────────────────
    T_on  = np.full((ev_h, ev_w), -(10**18), dtype=np.int64)
    T_off = np.full((ev_h, ev_w), -(10**18), dtype=np.int64)

    _roi_r_combined     = cfg["roi_radius_px"] + cfg["roi_dil_px"] // 2
    _roi_exp_k          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (201, 201))
    roi_mask            = None
    last_cable_centroid = None
    empty_streak        = 0

    EMPTY_EXPAND     = cfg["empty_expand"]
    EMPTY_RESET      = cfg["empty_reset"]
    INIT_MIN_LEN     = cfg["init_min_len"]
    BLEND_TAU_MS     = cfg["blend_tau_ms"]       # exponential decay time constant
    MAX_AGE_MS       = cfg["correction_max_age_ms"]
    INIT_TIMEOUT_US  = int(cfg["init_timeout_s"] * 1e6)  # wait for SAM3 before event fallback
    _init_deadline_us: Optional[int] = None      # set on first event batch
    _skip_us         = int(cfg.get("skip_s", 0.0) * 1_000_000)
    _skip_done       = (_skip_us == 0)           # True means no skip needed
    _sam3_init_path:     Optional[np.ndarray] = None  # persists until tracker is initialised
    _sam3_anchor:        Optional[np.ndarray] = None  # SAM3 path kept as CPD reinit anchor
    _anchor_frames_left: int = 0                       # frames to keep anchor active
    SAM3_ANCHOR_FRAMES   = 30                          # anchor CPD reinit for this many frames
    _sam3_triggered      = False                       

    if cfg["show"]:
        cv2.namedWindow("cpd",   cv2.WINDOW_NORMAL)
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    log.info("Event loop started.")

    try:
        for evs in it:
            # Pump GUI event loop every iteration so window stays responsive
            if cfg["show"] and (cv2.waitKey(1) & 0xFF) == 27:
                break

            if len(evs) == 0:
                continue

            # ─ Filter ─────────────────────────────────────────────────────
            ev = evs
            if trail is not None:
                trail.process_events(ev, trail_out)
                ev = trail_out.numpy()
            if afk is not None:
                afk.process_events(ev, afk_out)
                ev = afk_out.numpy()
            if stc_filter is not None:
                stc_filter.process_events(ev, stc_out)
                ev = stc_out.numpy()
            if len(ev) == 0:
                continue

            xs = ev["x"].astype(np.int32, copy=False)
            ys = ev["y"].astype(np.int32, copy=False)
            ts = ev["t"].astype(np.int64, copy=False)
            ps = ev["p"]
            t_now = int(ts[-1])
            t_iter_start = time.perf_counter()

            #  fast-forward past static period ───────────────────────
            if not _skip_done:
                if t_now < _skip_us:
                    continue   # discard batches before skip point
                # First batch past skip point — reset time surfaces clean
                T_on  = np.full((ev_h, ev_w), -(10**18), dtype=np.int64)
                T_off = np.full((ev_h, ev_w), -(10**18), dtype=np.int64)
                _skip_done = True
                log.info("[skip] Skipped to t=%.2fs — time surfaces reset", t_now / 1e6)

            on = ps if ps.dtype == np.bool_ else (ps > 0)

            # ── 2. Update time surfaces ───────────────────────────────────────
            if np.any(on):
                T_on[ys[on], xs[on]] = ts[on]
            if np.any(~on):
                T_off[ys[~on], xs[~on]] = ts[~on]

            dt_batch_us = int(ts[-1] - ts[0])
            if dt_batch_us <= 0:
                continue
            event_rate = len(ev) / (dt_batch_us * 1e-6)
            tau_us_dyn = ev_filter.adaptive_tau_us(
                event_rate,
                tau_min=cfg["tau_min_us"],
                tau_max=cfg["tau_max_us"],
            )

            # ── Recentness map ─────────────────────────────────────────────
            t_recentness = time.perf_counter()
            R = sae.compute_recentness(
                T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn
            )
            dt_recentness_ms = (time.perf_counter() - t_recentness) * 1000

            # ── Apply SAM3 corrections  ─────────
            for correction in shared.pop_all_corrections():
                age_ms = (t_now - correction.timestamp_us) / 1000.0
                w = correction.confidence * math.exp(-max(age_ms, 0.0) / BLEND_TAU_MS)
                w = min(w, 1.0)
                if age_ms > MAX_AGE_MS or w < 0.05:
                    log.debug("[Correction] Discarded  age=%.0fms  conf=%.2f  w=%.2f",
                              age_ms, correction.confidence, w)
                    continue
                sam3_poly = correction.polyline
                sam3_cx, sam3_cy = sam3_poly[:, 0].mean(), sam3_poly[:, 1].mean()
                log.info("[Correction] SAM3 poly centroid=(%.0f,%.0f) span_x=%.0f span_y=%.0f",
                         sam3_cx, sam3_cy,
                         sam3_poly[:, 0].max() - sam3_poly[:, 0].min(),
                         sam3_poly[:, 1].max() - sam3_poly[:, 1].min())
                if tracker.Y is not None:
                    trk_cx, trk_cy = tracker.Y[:, 0].mean(), tracker.Y[:, 1].mean()
                    log.info("[Correction] Tracker centroid=(%.0f,%.0f)", trk_cx, trk_cy)
                    n = len(tracker.Y)
                    if len(sam3_poly) != n:
                        sam3_poly = _resample_polyline(sam3_poly, n)
                    d_fwd = float(np.linalg.norm(sam3_poly - tracker.Y))
                    d_rev = float(np.linalg.norm(sam3_poly[::-1] - tracker.Y))
                    if d_rev < d_fwd:
                        sam3_poly = sam3_poly[::-1]
                    dist_to_sam3 = float(np.mean(np.linalg.norm(sam3_poly - tracker.Y, axis=1)))
                    if tracker.sigma2 > 50.0 or dist_to_sam3 > 100.0:
                        tracker.Y      = sam3_poly.copy()
                        tracker.sigma2 = 4.0
                        _sam3_init_path = None
                        
                        new_roi = np.zeros((ev_h, ev_w), dtype=np.uint8)
                        for pt in sam3_poly.astype(np.int32):
                            cv2.circle(new_roi, (pt[0], pt[1]),
                                       _roi_r_combined, 255, -1)
                        roi_mask = new_roi
                        _sam3_anchor        = sam3_poly.copy()
                        _anchor_frames_left = SAM3_ANCHOR_FRAMES
                        log.info("[Correction] HARD RESET onto SAM3 polyline  dist=%.0fpx  σ²=%.1f",
                                 dist_to_sam3, tracker.sigma2)
                    else:
                        tracker.Y = ((1.0 - w) * tracker.Y + w * sam3_poly).astype(np.float32)
                        if tracker.sigma2 > 0:
                            tracker.sigma2 = tracker.sigma2 * (1.0 - 0.5 * w)
                        log.info("[Correction] Applied  age=%.0fms  conf=%.2f  w=%.2f  dist=%.0fpx",
                                 age_ms, correction.confidence, w, dist_to_sam3)
                else:
                    # translate SAM3 polyline to align with events ──
                    sam3_poly = _align_polyline_to_events(sam3_poly, R)
                    # Update centroid after alignment
                    sam3_cx = float(sam3_poly[:, 0].mean())
                    sam3_cy = float(sam3_poly[:, 1].mean())

                    # ── set CPD prune_threshold from cable diameter ───
                    if correction.wire_diameter_px > 0:
                        gate = max(correction.wire_diameter_px * 1.5, 8.0)
                        tracker.p.prune_threshold = gate
                        log.info("[Correction] prune_threshold set to %.1fpx "
                                 "(diameter=%.1fpx × 1.5)",
                                 gate, correction.wire_diameter_px)

                    _sam3_init_path = sam3_poly
                    log.info("[Correction] SAM3 init stored  age=%.0fms  conf=%.2f  "
                             "aligned_centroid=(%.0f,%.0f)",
                             age_ms, correction.confidence, sam3_cx, sam3_cy)

            # ─ROI gating ─────────────────────────────────────────────────
            t_roi = time.perf_counter()
            if roi_mask is not None:
                R_gated     = R * (roi_mask > 0).astype(np.float32)
            else:
                R_gated = R
            dt_roi_ms = (time.perf_counter() - t_roi) * 1000

            # ── 6. Wire mask ──────────────────────────────────────────────────
            t_wiremask = time.perf_counter()
            active = R_gated[R_gated > 0]
            if len(active) == 0:
                continue

            wire_thr  = max(float(R_gated.max()) * cfg["wire_thr_frac"], 0.05)
            wire_mask = (R_gated >= wire_thr).astype(np.uint8)
            kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            wire_mask = wire_mask.astype(bool)
            dt_wiremask_ms = (time.perf_counter() - t_wiremask) * 1000

            # ──Select best cable component ────────────────────────────────
            if tracker.Y is not None:
                near_pt     = tracker.Y.mean(axis=0)
                near_weight = cfg["near_weight"]
            elif last_cable_centroid is not None:
                near_pt     = last_cable_centroid
                near_weight = cfg["near_weight"]
            else:
                near_pt     = None
                near_weight = 0.0

            raw_count = int(wire_mask.sum())
            t_bestcable = time.perf_counter()
            wire_mask = _best_cable_component(
                wire_mask,
                min_aspect  = cfg["min_aspect"],
                min_length  = cfg["min_length_px"],
                near_pt     = near_pt,
                near_weight = near_weight,
            )
            dt_bestcable_ms = (time.perf_counter() - t_bestcable) * 1000

            mask_count = int(wire_mask.sum())
            if raw_count > 0 and mask_count == 0:
                log.debug("[best_cable] raw=%d → 0 (no elongated component found)", raw_count)

            # ── Trigger SAM3 init on first valid cable detection ──────────────
            if (not _sam3_triggered
                    and sam3_thread is not None
                    and mask_count >= cfg["mask_min_px"]
                    and tracker.Y is None):
                shared.request_init(t_now)
                _sam3_triggered = True
                log.info("[SAM3-trigger] Cable detected at t=%.3fs — requesting SAM3 init",
                         t_now / 1e6)

            # If wire mask is empty but SAM3 gave an init path, use it directly
            if mask_count < cfg["mask_min_px"] or mask_count > wire_mask.size * cfg["mask_max_frac"]:
                log.debug("[mask] rejected count=%d min=%d max=%d",
                          mask_count, cfg["mask_min_px"], int(wire_mask.size * cfg["mask_max_frac"]))
                if _sam3_init_path is not None and tracker.Y is None:
                    # Init CPD directly from SAM3 polyline 
                    tracker.initialize(_sam3_init_path)
                    if tracker.Y is not None:

                        seg = np.linalg.norm(np.diff(tracker.Y, axis=0), axis=1)
                        spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                        tracker.sigma2 = max((spacing * 0.4) ** 2, 16.0)
                        # Keep SAM3 path as anchor for CPD reinit for next N frames
                        _sam3_anchor        = _sam3_init_path.copy()
                        _anchor_frames_left = SAM3_ANCHOR_FRAMES
                        _sam3_init_path     = None
                        last_cable_centroid = tracker.Y.mean(axis=0)
                        new_roi = np.zeros((ev_h, ev_w), dtype=np.uint8)
                        for pt in tracker.Y.astype(np.int32):
                            cv2.circle(new_roi, (pt[0], pt[1]), _roi_r_combined, 255, -1)
                        roi_mask = new_roi
                        log.info("[Init] CPD initialised from SAM3 path  σ²=%.1f", tracker.sigma2)
                continue

            # ──Skeleton observations ──────────────────────────────────────
            t_skel = time.perf_counter()
            obs, init_path = mask_to_skeleton_obs(wire_mask, max_pts=300)
            dt_skel_ms = (time.perf_counter() - t_skel) * 1000

            if _sam3_anchor is not None and _anchor_frames_left > 0:
                init_path = _sam3_anchor
            elif init_path is None and _sam3_init_path is not None:
                init_path = _sam3_init_path
            if obs is None:
                empty_streak += 1
                if roi_mask is not None:
                    if empty_streak == EMPTY_EXPAND:
                        roi_mask = cv2.dilate(roi_mask, _roi_exp_k)
                        log.info("[ROI] Expanded (empty for %d frames)", EMPTY_EXPAND)
                    elif empty_streak >= EMPTY_RESET:
                        roi_mask     = None
                        empty_streak = 0
                        log.info("[ROI] Full reset — cable lost for %d frames", EMPTY_RESET)
                continue

            empty_streak = 0

      
            if tracker.Y is None:
                if init_path is not None:
                    # Have an init path — check it's long enough
                    span = (np.linalg.norm(init_path[-1] - init_path[0])
                            if len(init_path) > 1 else 0)
                    if span < INIT_MIN_LEN:
                        # Too short (likely gripper edge) — keep waiting
                        if _init_deadline_us is None:
                            _init_deadline_us = t_now + INIT_TIMEOUT_US
                        continue
                    # Span OK 
                else:
                    # No init_path yet
                    if _init_deadline_us is None:
                        _init_deadline_us = t_now + INIT_TIMEOUT_US
                    if t_now < _init_deadline_us:
                        continue   # still waiting for SAM3 correction 
                    # Timeout reached 

            # ──  CPD tracking step ─────────────────────────────────────────
            t_cpd_start = time.perf_counter()
            tracker.track(obs, init_path=init_path)
            dt_cpd_ms = (time.perf_counter() - t_cpd_start) * 1000
            if tracker.Y is not None:
                _sam3_init_path = None   # consumed
            # Decrement anchor counter; clear when expired
            if _anchor_frames_left > 0:
                _anchor_frames_left -= 1
                if _anchor_frames_left == 0:
                    _sam3_anchor = None
                    log.info("[Anchor] SAM3 anchor expired — CPD reinit now uses wire mask")

            # ── Update ROI from tracker ───────────────────────────────────
            if tracker.Y is not None:
                last_cable_centroid = tracker.Y.mean(axis=0)
                new_roi = np.zeros((ev_h, ev_w), dtype=np.uint8)
                for pt in tracker.Y.astype(np.int32):
                    cv2.circle(new_roi, (pt[0], pt[1]), _roi_r_combined, 255, -1)
                roi_mask = new_roi

            # ── Visualise ─────────────────────────────────────────────────
            t_vis = time.perf_counter()
            if cfg["show"]:
                Visualization.visualize_wire(
                    R, mask=wire_mask, skel=None, win="debug", scale=1.0
                )
                frame = cv2.cvtColor(
                    np.clip(R * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
                )
                if obs is not None:
                    for pt in obs.astype(np.int32):
                        cv2.circle(frame, (pt[0], pt[1]), 1, (0, 200, 0), -1)
                if tracker.Y is None and init_path is not None and len(init_path) >= 2:
                    cv2.polylines(
                        frame,
                        [init_path.astype(np.int32).reshape(-1, 1, 2)],
                        False, (255, 100, 0), 1, cv2.LINE_AA,
                    )
                if tracker.Y is not None:
                    cv2.polylines(
                        frame,
                        [tracker.Y.astype(np.int32).reshape(-1, 1, 2)],
                        False, (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    for pt in tracker.Y.astype(np.int32):
                        cv2.circle(frame, (pt[0], pt[1]), 3, (0, 128, 255), -1)
                cv2.imshow("cpd", frame)

                # Save first event frame for calibration
                _snap_path = os.path.join(os.path.dirname(cfg["raw"]), "event_snap.png")
                if not os.path.exists(_snap_path):
                    cv2.imwrite(_snap_path, frame)
                    log.info("Saved event snap → %s", _snap_path)

                pass  # ESC handled at top of loop

            # ── Timing report ─────────────────────────────────────────────────
            dt_vis_ms  = (time.perf_counter() - t_vis) * 1000
            dt_iter_ms = (time.perf_counter() - t_iter_start) * 1000
            log.info("[Timing] iter=%.1fms | recentness=%.1f roi=%.1f wiremask=%.1f "
                     "bestcable=%.1f skel=%.1f cpd=%.1f vis=%.1f | obs=%d",
                     dt_iter_ms, dt_recentness_ms, dt_roi_ms, dt_wiremask_ms,
                     dt_bestcable_ms, dt_skel_ms, dt_cpd_ms, dt_vis_ms,
                     len(obs) if obs is not None else 0)

    finally:
        stop_event.set()
        if sam3_thread is not None:
            sam3_thread.join(timeout=5.0)
        cv2.destroyAllWindows()
        log.info("Pipeline stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Hybrid event+SAM3 cable tracker")
    p.add_argument("--raw",         default=DEFAULTS["raw"])
    p.add_argument("--frames-dir",  default=DEFAULTS["frames_dir"])
    p.add_argument("--manifest",    default=DEFAULTS["manifest"])
    p.add_argument("--homography",  default=DEFAULTS["homography"])
    p.add_argument("--show",        action="store_true", default=DEFAULTS["show"])
    p.add_argument("--delta-t-us",  type=int,   default=DEFAULTS["delta_t_us"])
    p.add_argument("--spline-samples", type=int, default=DEFAULTS["spline_samples"])
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(
        raw            = args.raw,
        frames_dir     = args.frames_dir,
        manifest       = args.manifest,
        homography     = args.homography,
        show           = args.show,
        delta_t_us     = args.delta_t_us,
        spline_samples = args.spline_samples,
    )
    return cfg


if __name__ == "__main__":
    run(_parse_args())
