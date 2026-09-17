#!/usr/bin/env python3
"""
MotionDLO Pipeline
==================
Architecture
------------
Two branches feed a single CPD tracker; branch selection is driven by the DLO
motion state v_DLO:

  A. Frame branch   RGB → SAM3 → polyline                  
  B. Event branch   events → SAE → mask → skeleton         

State machine 
------------------------------

  BOOT     v_DLO=0, t=0:    Wait for SAM3 init. Events drop. SAE keeps updating.
                            On SAM3 result: tracker.initialize() → STATIC.
  MOTION   v_DLO≠0:         Branch B updates tracker every batch.
                            Velocity warm-start active.
  STATIC   v_DLO=0, t≠0:    Branch B paused (hibernation = keep last event
                            polyline). One SAM3 fires on entry. On result:
                            hard-replace tracker.Y in event space (via H),
                            zero velocity. SAM3 retries up to K times on
                            empty mask. Stay here until motion resumes.

ρ_t is the count of post-STC events that landed on the *current* wire_mask
within a window of size rho_window_us. Hysteresis: state only switches when
ρ_t crosses ρ_thr in the consistent direction for N consecutive windows.

Modes (--mode)
--------------
  hybrid      Full state machine (paper).
  event-only  Always MOTION; no SAM3 thread. Event-only ablation.
  frame-only  Always STATIC; SAM3 fires every --frame-interval-ms; CPD bypassed.
              SAM3-only (frame-based) baseline.

Live mode
---------
Stubbed.  Run with --input-path set to a .raw file (offline).  To wire up the
IDS Peak path, replace `LiveRGBSource` below with the SDK-backed implementation.

Usage
-----
    python motiondlo_pipeline.py --mode hybrid \\
        --input-path events.raw --frames-dir frames/ \\
        --manifest manifest.json --homography rgb_to_event_H.npy --show

    See config.py for the full set of flags (outputs, CPD params, ρ_t tuning).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import label as sk_label, regionprops

# ── project imports ───────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import EventBased.Filter as ev_filter
from EventBased.sae import sae
from EventBased.Visualization import Visualization
from EventBased.event_cpd import (
    GeodesicCurveTracker,
    Params,
    mask_to_skeleton_obs,
    mask_to_pointcloud,
)
from metavision_core.event_io import EventsIterator
from EventBased.ridge_obs import ridge_wire_mask
import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (kernels, polyline ops)
# ══════════════════════════════════════════════════════════════════════════════
def _keep_elongated(
    wire_bool: np.ndarray,
    min_aspect: float = 3.5,
    min_length: float = 80.0,
) -> np.ndarray:

    wm = wire_bool.astype(np.uint8)
    contours, _ = cv2.findContours(wm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(wire_bool)
    out = np.zeros_like(wm)
    for c in contours:
        if len(c) >= 5:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
        else:
            x, y, w, h = cv2.boundingRect(c)
        major, minor = (w, h) if w >= h else (h, w)
        if minor > 0 and major >= min_length and (major / minor) >= min_aspect:
            cv2.drawContours(out, [c], -1, 1, cv2.FILLED)
    return out.astype(bool)

def adaptive_kernels(wire_diameter_px: float):
    """Derive morphological kernel sizes from the wire diameter."""
    # close: events fire on the wire edges, not the interior, so the kernel
    # needs ~1.5x the wire width to bridge the two rails
    close_size = max(int(round(wire_diameter_px * 1.5))  | 1, 5)
    # open: remove noise smaller than ~30% of wire width
    open_size  = max(int(round(wire_diameter_px * 0.3))  | 1, 3)
    # at least 2 iterations, more for thicker wires
    close_iters = max(2, int(round(wire_diameter_px / 25)))
    # Min area: wire is elongated, expect at least 10× diameter in length
    min_area    = int(wire_diameter_px * wire_diameter_px * 3)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size,  open_size))
    return close_k, close_iters, open_k, min_area

def _maxpool_sae(T: np.ndarray, s: int) -> np.ndarray:
    """s×s max-pool of an SAE timestamp plane (truncates partial edge blocks).

    exp(-(t_now - T)/tau) is monotone in T, so this is exact: max-pooling
    timestamps gives the same result as max-pooling full-res recentness.
    Unlike strided subsampling, no event is dropped.
    """
    hs, ws = T.shape[0] // s, T.shape[1] // s
    return np.maximum.reduce(
        [T[i::s, j::s][:hs, :ws] for i in range(s) for j in range(s)]
    )


#checked-aligns
def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resample a (M,2) polyline to exactly n points."""
    if len(pts) == n:
        return pts.astype(np.float32, copy=True)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.tile(pts[0], (n, 1)).astype(np.float32)
    t_new = np.linspace(0.0, total, n)
    xs = np.interp(t_new, cum, pts[:, 0])
    ys = np.interp(t_new, cum, pts[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)

#need to check
def _mask_rgb_to_event_polyline(
    mask_rgb: np.ndarray,
    H_rgb_to_ev: Optional[np.ndarray],
    event_h: int,
    event_w: int,
    n_nodes: int,
) -> Optional[tuple[np.ndarray, float]]:
    """Extract a polyline from an RGB SAM3 mask, warp it to event space via
    H_rgb_to_ev, and measure the wire diameter (event px, arc-length scaled).

    Returns (polyline_ev_n2, wire_diameter_ev_px) or None on failure.
    """
    if int((mask_rgb > 0).sum()) < 50:
        return None

    from FrameBased.Polyline import FramePolyline

    rgb_h, rgb_w = mask_rgb.shape[:2]

    polylines = FramePolyline.extract(mask_rgb, num_nodes=n_nodes, min_length=80)
    if not polylines:
        return None
    best_rgb = max(polylines, key=lambda Y: FramePolyline.arc_length(Y))
    arc_rgb  = float(FramePolyline.arc_length(best_rgb))

    # warp the points, not the mask (no rasterisation artefacts)
    pts = best_rgb.astype(np.float32).reshape(-1, 1, 2)
    if H_rgb_to_ev is not None:
        pts_ev = cv2.perspectiveTransform(pts, H_rgb_to_ev).reshape(-1, 2)
    else:
        sx, sy = event_w / rgb_w, event_h / rgb_h
        pts_ev = best_rgb.astype(np.float32) * np.array([sx, sy], dtype=np.float32)

    pts_ev[:, 0] = np.clip(pts_ev[:, 0], 0, event_w - 1)
    pts_ev[:, 1] = np.clip(pts_ev[:, 1], 0, event_h - 1)

    arc_ev    = float(np.sum(np.linalg.norm(np.diff(pts_ev, axis=0), axis=1)))
    arc_scale = arc_ev / max(arc_rgb, 1.0)

    # wire diameter in RGB space (P75 of distance transform) * arc_scale
    wire_d_ev = 0.0
    dist_t = cv2.distanceTransform((mask_rgb > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = dist_t[dist_t > 0]
    if len(interior) > 10:
        rgb_d = float(np.percentile(interior, 75)) * 2.0
        wire_d_ev = rgb_d * arc_scale

    return _resample_polyline(pts_ev, n_nodes), wire_d_ev


# ══════════════════════════════════════════════════════════════════════════════
#  Coordination primitives
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAM3Result:
    polyline_ev:      np.ndarray   # (n, 2) float32 in event space
    wire_diameter_px: float
    trigger_us:       int
    confidence:       float


class SAM3Channel:
    """Re-armable request/result channel.

    Each MOTION → STATIC transition posts a fresh request. The worker
    consumes it, retries on failure, and posts a result unless cancelled.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._pending_trigger: Optional[int] = None
        self._results: list[SAM3Result]      = []

    # ── Main thread side ───────────────────────────────────────────────────
    def request(self, trigger_us: int):
        with self._cond:
            self._pending_trigger = int(trigger_us)
            self._cond.notify_all()

    def pop_all_results(self) -> list[SAM3Result]:
        with self._cond:
            out, self._results = self._results, []
            return out

    def wait_for_results(self, timeout: Optional[float] = None) -> bool:
        """Block until at least one result is queued, without consuming it."""
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._cond:
            while not self._results:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if deadline is not None and remaining == 0.0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

    # ── Worker thread side ─────────────────────────────────────────────────
    def wait_for_request(self, timeout: float) -> Optional[int]:
        with self._cond:
            if self._pending_trigger is None:
                self._cond.wait(timeout=timeout)
            t, self._pending_trigger = self._pending_trigger, None
            return t

    def post_result(self, result: SAM3Result):
        with self._cond:
            self._results.append(result)
            self._cond.notify_all()


# ══════════════════════════════════════════════════════════════════════════════
#  SAM3 worker thread: one shot per static period, retry up to K times
# ══════════════════════════════════════════════════════════════════════════════

class SAM3Worker(threading.Thread):
    """Waits for SAM3Channel.request(), then per request:

      1. picks the manifest frame closest to the trigger ts (offline) or the
         latest live frame (live, stub)
      2. runs SAM3, up to retry_k+1 attempts on empty mask / extraction
         failure, stepping to the next manifest frame each retry
      3. mask -> RGB polyline -> event-space polyline via H
      4. drops the result if cancel_event is set (motion resumed mid-inference)
    """

    def __init__(
        self,
        channel:       SAM3Channel,
        manifest_path: Optional[str],
        frames_dir:    Optional[str],
        event_h:       int,
        event_w:       int,
        H_rgb_to_ev:   Optional[np.ndarray],
        n_nodes:       int,
        retry_k:       int,
        cancel_event:  threading.Event,
        stop_event:    threading.Event,
        live_source:   Optional["LiveRGBSource"] = None,
        debug_images:  bool = False,
    ):
        super().__init__(daemon=True, name="SAM3Worker")
        self.debug_images  = debug_images
        self.ch            = channel
        self.manifest_path = manifest_path
        self.frames_dir    = frames_dir
        self.event_h       = event_h
        self.event_w       = event_w
        self.H             = H_rgb_to_ev
        self.n_nodes       = n_nodes
        self.retry_k       = retry_k
        self.cancel_event  = cancel_event
        self._stop_event   = stop_event
        self.live_source   = live_source

        self._segmenter = None
        self._entries: list[tuple[str, int]] = []
        self._ts:      list[int]             = []
        self._prewarm_done = threading.Event()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the worker has loaded + pre-warmed SAM3 (or failed)."""
        return self._prewarm_done.wait(timeout=timeout)

    # ── Initialisation ─────────────────────────────────────────────────────
    def _lazy_load(self) -> bool:
        try:
            from FrameBased.Segmentation import SAM3Segmenter
            self._segmenter = SAM3Segmenter()
        except ImportError as e:
            log.error("SAM3Worker: cannot import SAM3Segmenter — %s", e)
            return False

        if self.manifest_path and os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path) as f:
                    manifest = json.load(f)
                self._entries = sorted(manifest.items(), key=lambda kv: kv[1])
                self._ts = [e[1] for e in self._entries]
                log.info("SAM3Worker: %d manifest frames loaded", len(self._entries))
            except Exception as e:
                log.error("SAM3Worker: cannot read manifest %s — %s", self.manifest_path, e)
                return False
        else:
            log.info("SAM3Worker: no manifest — assuming live mode")
        return True

    # ── Frame selection ────────────────────────────────────────────────────
    def _pick_frame_offline(self, trigger_us: int, attempt: int) -> Optional[tuple[str, int]]:
        """Closest manifest frame to trigger_us; retries step forward by `attempt`."""
        if not self._entries:
            return None
        idx = bisect.bisect_left(self._ts, trigger_us)
        if idx >= len(self._ts):
            base = len(self._ts) - 1
        elif idx == 0:
            base = 0
        else:
            base = idx if abs(self._ts[idx] - trigger_us) \
                          <= abs(self._ts[idx - 1] - trigger_us) \
                       else idx - 1
        target = min(base + attempt, len(self._entries) - 1)
        return self._entries[target]

    def _pick_frame_live(self) -> Optional[np.ndarray]:
        if self.live_source is None:
            raise NotImplementedError(
                "Live RGB source not configured. "
                "Provide a LiveRGBSource (e.g. IDS Peak) or use offline mode."
            )
        return self.live_source.get_latest_frame()

    # ── Pre-warm ───────────────────────────────────────────────────────────
    def _prewarm(self) -> None:
        bgr = None
        if self._entries and self.frames_dir:
            fname, _ = self._entries[0]
            bgr = cv2.imread(os.path.join(self.frames_dir, fname))
        if bgr is None:
            bgr = np.zeros((720, 1280, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        try:
            self._segmenter.segment(bgr)
        except Exception as e:
            log.warning("SAM3Worker: pre-warm failed (%s) — first real call will pay the cost", e)
            return
        log.info("SAM3Worker: pre-warm done in %.0fms", (time.perf_counter() - t0) * 1000)

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self):
        if not self._lazy_load():
            log.error("SAM3Worker: aborting (load failed).")
            self._prewarm_done.set()   # unblock waiters; they'll observe failure elsewhere
            return

        self._prewarm()
        self._prewarm_done.set()

        while not self._stop_event.is_set():
            trigger_us = self.ch.wait_for_request(timeout=1.0)
            if trigger_us is None:
                continue
            if self._stop_event.is_set():
                break
            log.info("SAM3Worker: triggered at t=%.3fs", trigger_us / 1e6)
            self.cancel_event.clear()
            self._handle_request(trigger_us)

    def _handle_request(self, trigger_us: int):
        last_err: Optional[str] = None
        for attempt in range(self.retry_k + 1):
            if self.cancel_event.is_set() or self._stop_event.is_set():
                log.info("SAM3Worker: cancelled before attempt %d", attempt)
                return

            # 1) Acquire frame
            if self._entries:
                pick = self._pick_frame_offline(trigger_us, attempt)
                if pick is None:
                    log.warning("SAM3Worker: no frame available")
                    return
                fname, ts_us = pick
                frame_path = os.path.join(self.frames_dir, fname)
                bgr = cv2.imread(frame_path)
                if bgr is None:
                    last_err = f"cannot read {frame_path}"
                    continue
                src_desc = fname
            else:
                try:
                    bgr = self._pick_frame_live()
                except NotImplementedError as e:
                    log.error("SAM3Worker: %s", e)
                    return
                if bgr is None:
                    last_err = "live source returned None"
                    continue
                src_desc = f"live@{trigger_us}us"

            # 2) Run SAM3
            t0 = time.perf_counter()
            try:
                mask = self._segmenter.segment(bgr)
            except Exception as e:
                last_err = f"SAM3 exception: {e}"
                log.warning("SAM3Worker: %s (attempt %d)", last_err, attempt)
                continue
            dt_sam3 = (time.perf_counter() - t0) * 1000

            if self.cancel_event.is_set() or self._stop_event.is_set():
                log.info("SAM3Worker: cancelled after inference (%.0fms wasted)", dt_sam3)
                return

            if int((mask > 0).sum()) < 50:
                last_err = "empty mask"
                log.info("SAM3Worker: empty mask on %s (attempt %d)", src_desc, attempt)
                continue

            # dump mask + frame + H-warped mask, for checking the homography
            if self.debug_images:
              try:
                debug_dir = "sam3_debug"
                os.makedirs(debug_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(src_desc))[0] or f"t{trigger_us:09d}us"
                tag  = f"{stem}_a{attempt}"
                mask_u8 = mask.astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
                cv2.imwrite(os.path.join(debug_dir, f"{tag}_mask.png"), mask_u8)
                cv2.imwrite(os.path.join(debug_dir, f"{tag}_frame.png"), bgr)
                # red mask over the source, for an at-a-glance check
                overlay = bgr.copy()
                overlay[mask > 0] = (0, 0, 255)
                blended = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
                cv2.imwrite(os.path.join(debug_dir, f"{tag}_overlay.png"), blended)
                # H-warped mask at event resolution
                if self.H is not None:
                    warped = cv2.warpPerspective(
                        mask_u8, self.H, (self.event_w, self.event_h),
                        flags=cv2.INTER_NEAREST,
                    )
                    cv2.imwrite(os.path.join(debug_dir, f"{tag}_mask_warped.png"), warped)
              except Exception as e:
                log.warning("SAM3Worker: debug-dump failed (%s)", e)

            # 3) RGB mask → event-space polyline
            res = _mask_rgb_to_event_polyline(
                mask, self.H, self.event_h, self.event_w, self.n_nodes
            )
            if res is None:
                last_err = "polyline extraction failed"
                log.info("SAM3Worker: %s on %s (attempt %d)", last_err, src_desc, attempt)
                continue
            polyline_ev, wire_d_ev = res

            if self.cancel_event.is_set() or self._stop_event.is_set():
                log.info("SAM3Worker: cancelled before posting")
                return

            cable_px = int((mask > 0).sum())
            total_px = mask.shape[0] * mask.shape[1]
            confidence = min(cable_px / max(total_px * 0.001, 1), 1.0)

            self.ch.post_result(SAM3Result(
                polyline_ev      = polyline_ev,
                wire_diameter_px = wire_d_ev,
                trigger_us       = trigger_us,
                confidence       = confidence,
            ))
            log.info("SAM3Worker: posted  src=%s  d=%.1fpx  conf=%.2f  sam3=%.0fms  attempt=%d",
                     src_desc, wire_d_ev, confidence, dt_sam3, attempt)
            return

        log.warning("SAM3Worker: all %d attempts failed (last err: %s) — giving up",
                    self.retry_k + 1, last_err)


# ══════════════════════════════════════════════════════════════════════════════
#  Live RGB source (stub)
# ══════════════════════════════════════════════════════════════════════════════

class LiveRGBSource:
    """Stub.  Replace with IDS Peak (or other) implementation when needed."""
    def get_latest_frame(self) -> Optional[np.ndarray]:
        raise NotImplementedError(
            "LiveRGBSource is a stub. Implement get_latest_frame() with IDS Peak "
            "(see synchronised_recorder.py) or pass --input-path for offline mode."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Hysteretic motion-state detector
# ══════════════════════════════════════════════════════════════════════════════

class MotionStateDetector:
    """Hysteretic v_DLO detector.

    ρ_t updates once per fixed-width window. The state only switches after
    ρ_t has crossed ρ_thr in the same direction for N consecutive windows.
    """

    def __init__(self, rho_thr: float, n_hysteresis: int, initially_moving: bool = False):
        self.rho_thr   = float(rho_thr)
        self.n         = int(max(1, n_hysteresis))
        self.is_moving = bool(initially_moving)
        self._above    = 0
        self._below    = 0

    def update(self, rho_t: float) -> bool:
        """Update with one window's ρ. Returns True if state changed."""
        if rho_t > self.rho_thr:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0

        changed = False
        if not self.is_moving and self._above >= self.n:
            self.is_moving = True
            changed = True
        elif self.is_moving and self._below >= self.n:
            self.is_moving = False
            changed = True
        return changed


# ══════════════════════════════════════════════════════════════════════════════
#  Output recorders
# ══════════════════════════════════════════════════════════════════════════════

class PolylineRecorder:
    """Per-iteration tracker polylines → single .npz at the end."""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self._ts, self._polys, self._state, self._branch = [], [], [], []

    def append(self, ts_us: int, polyline: Optional[np.ndarray],
               state: str, branch: str):
        if polyline is None:
            poly = np.full((self.n_nodes, 2), np.nan, dtype=np.float32)
        else:
            poly = polyline.astype(np.float32, copy=False)
            if poly.shape[0] != self.n_nodes:
                poly = _resample_polyline(poly, self.n_nodes)
        self._ts.append(int(ts_us))
        self._polys.append(poly)
        self._state.append(state)
        self._branch.append(branch)

    def save(self, path: Optional[str]):
        if not path or not self._ts:
            return
        np.savez(
            path,
            timestamps_us=np.asarray(self._ts, dtype=np.int64),
            polylines    =np.stack(self._polys, axis=0),
            state        =np.asarray(self._state),
            branch       =np.asarray(self._branch),
        )
        log.info("PolylineRecorder: wrote %d entries to %s", len(self._ts), path)


class SAM3PolylineRecorder:
    """Every SAM3 polyline that was actually applied to the tracker."""

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self._ts, self._polys, self._wd = [], [], []

    def append(self, ts_us: int, polyline_ev: np.ndarray, wire_d: float):
        poly = polyline_ev.astype(np.float32, copy=False)
        if poly.shape[0] != self.n_nodes:
            poly = _resample_polyline(poly, self.n_nodes)
        self._ts.append(int(ts_us))
        self._polys.append(poly)
        self._wd.append(float(wire_d))

    def save(self, path: Optional[str]):
        if not path or not self._ts:
            return
        np.savez(
            path,
            timestamps_us=np.asarray(self._ts, dtype=np.int64),
            polylines_ev =np.stack(self._polys, axis=0),
            wire_diameter=np.asarray(self._wd, dtype=np.float32),
        )
        log.info("SAM3PolylineRecorder: wrote %d entries to %s", len(self._ts), path)


class TimingCSVWriter:
    """One timing row per event-loop iteration."""

    COLUMNS = [
        "timestamp_us", "state", "branch", "rho_t", "mask_count",
        "event_count", "num_obs",
        "filter_ms", "sae_ms", "recentness_ms", "wiremask_ms",
        "elongated_ms", "skel_ms", "cpd_ms", "vis_ms", "total_ms",
    ]

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None
        self._w  = None
        if path:
            self._fh = open(path, "w", newline="")
            self._w  = csv.DictWriter(self._fh, fieldnames=self.COLUMNS)
            self._w.writeheader()

    def write(self, **kw):
        if self._w is None:
            return
        self._w.writerow({k: kw.get(k, "") for k in self.COLUMNS})

    def close(self):
        if self._fh is not None:
            self._fh.close()
            log.info("TimingCSVWriter: closed %s", self.path)


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    mode = getattr(args, "mode", "hybrid")
    log.info("MotionDLO pipeline starting  mode=%s", mode)

    # ── Paths ────────────────────────────────────────────────────────────────
    raw_path        = args.input_path
    frames_dir      = getattr(args, "frames_dir", "") or ""
    manifest_path   = getattr(args, "manifest",   "") or ""
    homography_path = getattr(args, "homography", "") or ""

    # ── Homography (RGB → event) ─────────────────────────────────────────────
    H: Optional[np.ndarray] = None
    if homography_path and os.path.isfile(homography_path):
        H = np.load(homography_path)
        log.info("Homography (RGB→event) loaded from %s  shape=%s", homography_path, H.shape)
    elif mode in ("hybrid", "frame-only"):
        log.warning("No homography file — falling back to image-ratio scaling")

    # ── Event source (offline only, live is a stub) ─────────────────────────
    if not raw_path:
        raise NotImplementedError(
            "Live event source not wired in this build. "
            "Pass --input-path /path/to/events.raw"
        )
    it = EventsIterator(input_path=raw_path, delta_t=args.delta_t_us, mode="delta_t")
    ev_h, ev_w = it.get_size()
    log.info("Event sensor: %d × %d  raw=%s", ev_w, ev_h, raw_path)

    # ── save-vis output dirs (next to the input .raw) ───────────────────────
    #   results/  full cpd view
    #   masks/    recentness + HUD only
    save_vis     = bool(getattr(args, "save_vis", False))
    debug_images = bool(getattr(args, "debug_images", False))
    results_dir: Optional[str] = None
    masks_dir:   Optional[str] = None
    if save_vis:
        parent = os.path.dirname(os.path.abspath(raw_path))
        results_dir = os.path.join(parent, "results")
        masks_dir   = os.path.join(parent, "masks")
        for d in (results_dir, masks_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        log.info("save-vis: cpd → %s | masks → %s (cleared)", results_dir, masks_dir)

    # ── STC filter chain ─────────────────────────────────────────────────────
    filter_chain = ev_filter.EventFilterChain(ev_w, ev_h, args)

    # ── CPD tracker ──────────────────────────────────────────────────────────
    tracker = GeodesicCurveTracker(Params(
        num_nodes       = args.spline_samples,
        beta            = getattr(args, "cpd_beta",        20.0),
        lam             = getattr(args, "cpd_lam",         0.0001),
        mu              = getattr(args, "cpd_mu",          0.1),
        k_vis           = getattr(args, "cpd_k_vis",       0.05),
        max_iter        = getattr(args, "cpd_max_iter",    12),
        tol             = getattr(args, "cpd_tol",         1e-4),
        beta_pre_proc   = getattr(args, "cpd_beta_pre",    1.0),
        lam_pre_proc    = getattr(args, "cpd_lam_pre",     0.0001),
        prune_threshold = getattr(args, "cpd_prune_thr",   300.0),
    ))

    # ── SAE buffers + initial morphology kernels ─────────────────────────────
    T_on  = np.full((ev_h, ev_w), -10**18, dtype=np.int64)
    T_off = np.full((ev_h, ev_w), -10**18, dtype=np.int64)

    wire_d_px = float(getattr(args, "default_wire_diameter_px", 50.0))
    close_k, close_iters, open_k, _min_area = adaptive_kernels(wire_d_px)
    log.info("Initial morphology kernels  wire_d=%.1fpx", wire_d_px)

    # ── Velocity warm-start state ────────────────────────────────────────────
    _prev_centroid: Optional[np.ndarray] = None
    _prev_t:        Optional[int]        = None
    _velocity = np.zeros(2, dtype=np.float32)

    # recentness crop bbox, set after the first wire_mask succeeds
    _last_active_bbox: Optional[tuple[int, int, int, int]] = None
    RECENTNESS_MARGIN_PX = 120

    def _reset_velocity():
        nonlocal _prev_centroid, _prev_t
        _prev_centroid = None
        _prev_t        = None
        _velocity.fill(0.0)

    # ── Low-res mask path (ridge only) ───────────────────────────────────────
    # ridge cost scales with wire_d_px and the detector is scale-invariant,
    # so run recentness + ridge + skeleton at 1/N res with wire_d/N.
    # CPD still runs in full-res event coords.
    mask_scale_arg = int(getattr(args, "mask_scale", 0))
    use_ridge_mask = bool(getattr(args, "use_ridge_mask", False))
    _args_lo_cache: dict[int, argparse.Namespace] = {}
    _last_logged_scale = [0]
    gpu_engine = None   # set after the GUI block (CUDA init must follow imshow)

    def _mask_scale() -> int:
        """Current downscale factor. Auto mode targets ~12 px wire width."""
        if not use_ridge_mask or gpu_engine is not None:
            return 1
        sc = mask_scale_arg if mask_scale_arg >= 1 \
             else max(1, int(round(wire_d_px / 12.0)))
        if sc != _last_logged_scale[0]:
            log.info("[mask-scale] 1/%d resolution (wire_d=%.1fpx)", sc, wire_d_px)
            _last_logged_scale[0] = sc
        return sc

    def _scaled_ridge_args(sc: int) -> argparse.Namespace:
        """args copy with px-valued ridge params shrunk to the low-res grid."""
        a = _args_lo_cache.get(sc)
        if a is None:
            a = argparse.Namespace(**vars(args))
            a.ridge_min_length    = float(getattr(args, "ridge_min_length",    40.0)) / sc
            a.ridge_close_fast_px = float(getattr(args, "ridge_close_fast_px",  7.0)) / sc
            bd = float(getattr(args, "ridge_band_dilate", -1.0))
            a.ridge_band_dilate   = bd / sc if bd > 0 else bd
            _args_lo_cache[sc] = a
        return a

    # ── Motion-state detector ────────────────────────────────────────────────
    detector = MotionStateDetector(
        rho_thr      = float(getattr(args, "rho_thr",         50.0)),
        n_hysteresis = int(  getattr(args, "rho_hysteresis_n", 3)),
        initially_moving = (mode == "event-only"),
    )

    # ── GUI windows ────────────────
    if args.show:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
        cv2.namedWindow("cpd",   cv2.WINDOW_NORMAL)
        _dummy = np.zeros((ev_h, ev_w, 3), dtype=np.uint8)
        cv2.imshow("debug", _dummy)
        cv2.imshow("cpd",   _dummy)
        cv2.waitKey(50)   # give Qt a moment to actually paint

    # ── GPU ridge engine (CUDA init must come after the GUI block) ───────────
    if bool(getattr(args, "mask_gpu", False)) and use_ridge_mask:
        try:
            from EventBased.ridge_gpu import GPURidgeEngine
            gpu_engine = GPURidgeEngine(
                ev_h, ev_w,
                binarize=str(getattr(args, "ridge_binarize", "nms")),
            )
            log.info("[mask-gpu] full-res ridge path on %s", gpu_engine.device_name)
        except Exception as e:
            log.warning("[mask-gpu] unavailable (%s) — using CPU path", e)
            gpu_engine = None

    # ── SAM3 channel + worker thread ─────────────────────────────────────────
    sam3_ch      = SAM3Channel()
    stop_event   = threading.Event()
    cancel_event = threading.Event()
    sam3_worker: Optional[SAM3Worker] = None

    if mode in ("hybrid", "frame-only"):
        sam3_worker = SAM3Worker(
            channel       = sam3_ch,
            manifest_path = manifest_path if os.path.isfile(manifest_path) else None,
            frames_dir    = frames_dir,
            event_h       = ev_h,
            event_w       = ev_w,
            H_rgb_to_ev   = H,
            n_nodes       = args.spline_samples,
            retry_k       = int(getattr(args, "sam3_retry_k", 3)),
            cancel_event  = cancel_event,
            stop_event    = stop_event,
            live_source   = None,
            debug_images  = debug_images,
        )
        sam3_worker.start()
        log.info("Waiting for SAM3 pre-warm to finish …")
        if not sam3_worker.wait_until_ready(timeout=60.0):
            log.warning("SAM3 pre-warm did not finish within 60s — continuing anyway")

    # ── States ───────────────────────────────────────────────────────────────
    BOOT, MOTION, STATIC = "BOOT", "MOTION", "STATIC"
    if   mode == "hybrid":     state = BOOT
    elif mode == "event-only": state = MOTION
    else:                      state = STATIC  # frame-only

    sam3_pending = False   # True while its waiting for a SAM3 result

    if mode == "hybrid":
        sam3_ch.request(0)
        sam3_pending = True
        log.info("[State] BOOT — waiting for first SAM3 result …")
        if sam3_worker is not None and not sam3_ch.wait_for_results(timeout=30.0):
            log.warning("First SAM3 result not received within 30s — BOOT will stall")
    elif mode == "frame-only":
        sam3_ch.request(0)
        sam3_pending = True
        if sam3_worker is not None and not sam3_ch.wait_for_results(timeout=30.0):
            log.warning("First SAM3 result not received within 30s")

    last_polyline: Optional[np.ndarray] = None  
    # ── Output writers ───────────────────────────────────────────────────────
    poly_rec  = PolylineRecorder(n_nodes=args.spline_samples)
    sam3_rec  = SAM3PolylineRecorder(n_nodes=args.spline_samples)
    timing    = TimingCSVWriter(getattr(args, "timing_csv", "") or None)

    # ── ρ-window accounting ──────────────────────────────────────────────────
    rho_window_us       = int(getattr(args, "rho_window_us", 20000))
    rho_window_start_us : Optional[int] = None
    rho_event_count     = 0
    rho_t_last          = 0.0

    # ── frame-only mode: periodic SAM3 retrigger ─────────────────────────────
    frame_only_period_us = int(getattr(args, "frame_interval_ms", 1000)) * 1000
    next_frame_only_us   = frame_only_period_us  # first one already requested at t=0

    # ── Timing accumulators ──────────────────────────────────────────────────
    _t_total = []
    _iter    = 0

    _event_mask_count = 0

    log.info("Event loop ready  state=%s  rho_thr=%.1f  N=%d  window=%dµs",
             state, detector.rho_thr, detector.n, rho_window_us)

    display_fps_arg = int(getattr(args, "display_fps", 0) or 0)
    waitkey_ms = max(1, int(round(1000.0 / display_fps_arg))) if display_fps_arg > 0 else 1

    try:
        for evs in it:
            # GUI pump
            if args.show and (cv2.waitKey(waitkey_ms) & 0xFF) == 27:
                break
            if len(evs) == 0:
                continue

            t_iter = time.perf_counter()

            # ─── 1. STC filter ────────────────────────────────────────────────
            t0 = time.perf_counter()
            ev = filter_chain.apply(evs)
            dt_filter_ms = (time.perf_counter() - t0) * 1000
            if len(ev) == 0:
                continue

            xs = ev["x"].astype(np.int32, copy=False)
            ys = ev["y"].astype(np.int32, copy=False)
            ts = ev["t"].astype(np.int64, copy=False)
            ps = ev["p"]
            on = ps if ps.dtype == np.bool_ else (ps > 0)
            t_now = int(ts[-1])

            # ─── 2. SAE update (always, including BOOT and STATIC) ───────────
            t0 = time.perf_counter()
            if np.any(on):
                T_on [ys[on],  xs[on]]  = ts[on]
            if np.any(~on):
                T_off[ys[~on], xs[~on]] = ts[~on]
            if gpu_engine is not None:
                gpu_engine.update_sae(xs, ys, ts, np.asarray(on, dtype=bool))
            dt_sae_ms = (time.perf_counter() - t0) * 1000

            dt_batch_us = int(ts[-1] - ts[0])
            if dt_batch_us <= 0:
                continue
            event_rate = len(ev) / (dt_batch_us * 1e-6)
            tau_us_dyn = ev_filter.adaptive_tau_us(
                            event_rate,
                            tau_min=args.tau_min_us,
                            tau_max=args.tau_max_us,
                            r_lo=args.tau_rate_lo,
                            r_hi=args.tau_rate_hi,
                        )

            # ═══════════════════════════════════════════════════════════════
            #  BOOT: drop events for CPD, just wait for SAM3
            # ═══════════════════════════════════════════════════════════════
            if state == BOOT:
                for sig in sam3_ch.pop_all_results():
                    if sig.polyline_ev is None or len(sig.polyline_ev) < 2:
                        continue
                    if sig.wire_diameter_px > 5.0:
                        wire_d_px = sig.wire_diameter_px
                        close_k, close_iters, open_k, _min_area = adaptive_kernels(wire_d_px)
                        log.info("[SAM3] wire_d=%.1fpx — kernels updated", wire_d_px)

                    tracker.initialize(sig.polyline_ev)
                    if tracker.Y is not None:
                        seg = np.linalg.norm(np.diff(tracker.Y, axis=0), axis=1)
                        spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                        tracker.sigma2 = max((spacing * 0.4) ** 2, 16.0)
                    _reset_velocity()
                    last_polyline = tracker.Y.copy() if tracker.Y is not None else None
                    sam3_rec.append(t_now, sig.polyline_ev, sig.wire_diameter_px)
                    sam3_pending = False
                    state = STATIC
                    log.info("[State] BOOT → STATIC  (CPD initialised, σ²=%.1f)",
                             float(tracker.sigma2) if tracker.Y is not None else -1.0)
                    break

                poly_rec.append(t_now, last_polyline, state, "H")
                timing.write(
                    timestamp_us=t_now, state=state, branch="H",
                    rho_t=0, mask_count=0, event_count=len(ev), num_obs=0,
                    filter_ms=f"{dt_filter_ms:.3f}", sae_ms=f"{dt_sae_ms:.3f}",
                    recentness_ms=0, wiremask_ms=0, elongated_ms=0,
                    skel_ms=0, cpd_ms=0, vis_ms=0,
                    total_ms=f"{(time.perf_counter()-t_iter)*1000:.3f}",
                )
                continue

            # ═══════════════════════════════════════════════════════════════
            #  MOTION + STATIC + frame-only: compute recentness + wire_mask
            # ═══════════════════════════════════════════════════════════════
            sc = _mask_scale()
            t0 = time.perf_counter()
            R: Optional[np.ndarray] = None   # full-res; materialised lazily
            # crop to (last cable bbox ∪ current event bbox) + margin; anything
            # outside has decayed to ~0. first iteration is full-frame.
            if gpu_engine is not None:
                # R stays on the device, consumed by gpu_engine.wire_mask
                Rw = None
                gpu_engine.compute_recentness(t_now, tau_us_dyn, wire_d_px=wire_d_px)
            elif sc > 1:
                # exact at 1/sc² the pixels, see _maxpool_sae
                Rw = sae.compute_recentness(
                    T_on=_maxpool_sae(T_on, sc), T_off=_maxpool_sae(T_off, sc),
                    t_now_us=t_now, tau_us=tau_us_dyn,
                )
            elif _last_active_bbox is None:
                R = sae.compute_recentness(
                    T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn,
                )
            else:
                ly0, ly1, lx0, lx1 = _last_active_bbox
                ey0, ey1 = int(ys.min()), int(ys.max()) + 1
                ex0, ex1 = int(xs.min()), int(xs.max()) + 1
                y0_r = max(0,    min(ly0, ey0) - RECENTNESS_MARGIN_PX)
                y1_r = min(ev_h, max(ly1, ey1) + RECENTNESS_MARGIN_PX)
                x0_r = max(0,    min(lx0, ex0) - RECENTNESS_MARGIN_PX)
                x1_r = min(ev_w, max(lx1, ex1) + RECENTNESS_MARGIN_PX)
                if (y1_r - y0_r) * (x1_r - x0_r) > 0.8 * ev_h * ev_w:
                    R = sae.compute_recentness(
                        T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn,
                    )
                else:
                    R_crop = sae.compute_recentness(
                        T_on=T_on[y0_r:y1_r, x0_r:x1_r],
                        T_off=T_off[y0_r:y1_r, x0_r:x1_r],
                        t_now_us=t_now, tau_us=tau_us_dyn,
                    )
                    R = np.zeros((ev_h, ev_w), dtype=np.float32)
                    R[y0_r:y1_r, x0_r:x1_r] = R_crop
            if sc == 1:
                Rw = R   # working image = full-res
            dt_recentness_ms = (time.perf_counter() - t0) * 1000

            _V = getattr(tracker, "_V", None)
            node_speed = (float(np.linalg.norm(_V, axis=1).mean())
                          if _V is not None else 0.0)

            R_gated = Rw
            if getattr(args, "use_corridor", False) and tracker.Y is not None:
                pred = (tracker.Y + _V) if _V is not None else tracker.Y
                corridor_margin = int(round(
                    (args.corridor_margin_px + args.corridor_speed_gain * node_speed) / sc))
                corridor_margin = max(corridor_margin, 1)
                gate_shape = (ev_h, ev_w) if gpu_engine is not None else Rw.shape
                corridor_mask = np.zeros(gate_shape, np.uint8)
                cv2.polylines(corridor_mask, [(pred / sc).astype(np.int32)], False, 1, 1)
                corridor_mask = cv2.dilate(corridor_mask, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * corridor_margin + 1, 2 * corridor_margin + 1)))
                if gpu_engine is not None:
                    gpu_engine.apply_gate(corridor_mask)
                else:
                    R_gated = Rw * corridor_mask.astype(np.float32)

            t0 = time.perf_counter()
            wm_lo: Optional[np.ndarray] = None   # low-res mask (sc>1), feeds skeleton
            if use_ridge_mask:
                # Ridge filter handles thresholding + elongation internally.
                if gpu_engine is not None:
                    wire_mask = gpu_engine.wire_mask(wire_d_px, args, speed_px=node_speed)
                    # Table-V split: wiremask_ms = ROI masking on GPU,
                    # elongated_ms = download + candidate selection
                    dt_wiremask_ms = gpu_engine.last_split_ms[0]
                    dt_elong_ms    = gpu_engine.last_split_ms[1]
                elif sc > 1:
                    wm_lo = ridge_wire_mask(R_gated, wire_d_px / sc,
                                            _scaled_ridge_args(sc), speed_px=node_speed)
                    wire_mask = cv2.resize(
                        wm_lo.astype(np.uint8), (ev_w, ev_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    wire_mask = ridge_wire_mask(R_gated, wire_d_px, args, speed_px=node_speed)
                if gpu_engine is None:
                    dt_wiremask_ms = (time.perf_counter() - t0) * 1000
                    dt_elong_ms = 0.0
            else:
                wire_mask = np.zeros((ev_h, ev_w), dtype=bool)
                active = R_gated[R_gated > 0]
                if len(active) > 0:
                    R_u8 = np.clip(R_gated * 255, 0, 255).astype(np.uint8)
                    otsu_val, _ = cv2.threshold(
                        R_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    wire_thr = min(max(float(otsu_val) / 255.0, args.wire_thr_floor),
                                   args.wire_thr_cap)
                    wm = (R_gated >= wire_thr).astype(np.uint8)

                    if debug_images:
                        try:
                            os.makedirs("sam3_debug", exist_ok=True)
                            cv2.imwrite(os.path.join(
                                "sam3_debug",
                                f"wm_prethresh_{_iter:06d}_t{t_now:09d}us.png"),
                                wm * 255)
                        except Exception as e:
                            log.warning("prethresh dump failed: %s", e)

                    ys_m, xs_m = np.nonzero(wm)
                    if len(ys_m) > 0:
                        margin = (close_k.shape[0] // 2) * close_iters \
                               + (open_k.shape[0]  // 2) + 2
                        y0 = max(0,    int(ys_m.min()) - margin)
                        y1 = min(ev_h, int(ys_m.max()) + margin + 1)
                        x0 = max(0,    int(xs_m.min()) - margin)
                        x1 = min(ev_w, int(xs_m.max()) + margin + 1)

                        patch = wm[y0:y1, x0:x1]
                        patch = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, close_k, iterations=close_iters)
                        patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN,  open_k,  iterations=1)

                        wm = np.zeros_like(wm)
                        wm[y0:y1, x0:x1] = patch

                    wire_mask = wm.astype(bool)
                dt_wiremask_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                if wire_mask.any():
                    wire_mask = _keep_elongated(
                        wire_mask,
                        min_aspect=args.mask_min_aspect,
                        min_length=args.mask_min_length,
                    )
                dt_elong_ms = (time.perf_counter() - t0) * 1000
            mask_count = int(np.count_nonzero(wire_mask))

            # cache cable bbox for the next recentness crop (unused on GPU)
            if mask_count > 0 and gpu_engine is None:
                if wm_lo is not None:
                    ys_w, xs_w = np.nonzero(wm_lo)
                    _last_active_bbox = (
                        int(ys_w.min()) * sc, min(ev_h, (int(ys_w.max()) + 1) * sc),
                        int(xs_w.min()) * sc, min(ev_w, (int(xs_w.max()) + 1) * sc),
                    )
                else:
                    ys_w, xs_w = np.nonzero(wire_mask)
                    _last_active_bbox = (
                        int(ys_w.min()), int(ys_w.max()) + 1,
                        int(xs_w.min()), int(xs_w.max()) + 1,
                    )

            # full-res R is only needed for display/debug, skip when headless
            if R is None and (args.show or save_vis or debug_images):
                if gpu_engine is not None:
                    R = gpu_engine.download_R()
                else:
                    R = cv2.resize(Rw, (ev_w, ev_h), interpolation=cv2.INTER_LINEAR)

            # numbered + timestamped so they line up against frames/manifest
            if debug_images and mask_count > 200:
                try:
                    os.makedirs("sam3_debug", exist_ok=True)
                    R_u8_vis = np.clip(R * 255.0, 0, 255).astype(np.uint8)
                    R_color  = cv2.applyColorMap(R_u8_vis, cv2.COLORMAP_TURBO)
                    R_color[wire_mask] = (0, 255, 0)
                    tag = f"{_event_mask_count:03d}_t{t_now:09d}us"
                    cv2.imwrite(f"sam3_debug/event_debug_snapshot_{tag}.png", R_color)
                    cv2.imwrite(f"sam3_debug/event_wire_mask_{tag}.png",
                                (wire_mask.astype(np.uint8) * 255))
                    log.info(
                        "Saved event-space debug snapshot %d "
                        "(sam3_debug/event_debug_snapshot_%s.png, mask_count=%d)",
                        _event_mask_count + 1, tag, mask_count,
                    )
                    _event_mask_count += 1
                except Exception as e:
                    log.warning("Could not save event debug snapshot: %s", e)

            # ─── 3. ρ_t accumulation across windows ──────────────────────────
            if rho_window_start_us is None:
                rho_window_start_us = t_now
            if mask_count > 0:
                rho_event_count += int(wire_mask[ys, xs].sum())

            state_changed = False
            if (t_now - rho_window_start_us) >= rho_window_us:
                rho_t_last = float(rho_event_count)
                if mode == "hybrid":
                    state_changed = detector.update(rho_t_last)
                rho_event_count     = 0
                rho_window_start_us = t_now

            # ─── 4. Branch-switching transitions (hybrid only) ───────────────
            if mode == "hybrid" and state_changed:
                if detector.is_moving and state == STATIC:
                    log.info("[State] STATIC → MOTION  ρ_t=%.1f  (cancelling SAM3 if running)",
                             rho_t_last)
                    cancel_event.set()
                    sam3_pending = False
                    _reset_velocity()
                    state = MOTION
                elif (not detector.is_moving) and state == MOTION:
                    log.info("[State] MOTION → STATIC  ρ_t=%.1f  (firing SAM3)", rho_t_last)
                    cancel_event.clear()
                    sam3_ch.request(t_now)
                    sam3_pending = True
                    _reset_velocity()
                    state = STATIC

            # ─── 5. Periodic SAM3 retrigger (frame-only mode) ────────────────
            if mode == "frame-only" and not sam3_pending and t_now >= next_frame_only_us:
                sam3_ch.request(t_now)
                sam3_pending = True
                next_frame_only_us = t_now + frame_only_period_us

            # ─── 6. Consume SAM3 results: hard-replace tracker.Y ─────────────
            branch_tag = "H"   # default: hold

            if sam3_pending and mode in ("hybrid", "frame-only"):
                for sig in sam3_ch.pop_all_results():
                    if sig.polyline_ev is None or len(sig.polyline_ev) < 2:
                        continue
                    new_Y = sig.polyline_ev
                    if new_Y.shape[0] != args.spline_samples:
                        new_Y = _resample_polyline(new_Y, args.spline_samples)

                    # Orientation alignment with previous tracker.Y
                    if tracker.Y is not None:
                        d_fwd = float(np.linalg.norm(new_Y       - tracker.Y))
                        d_rev = float(np.linalg.norm(new_Y[::-1] - tracker.Y))
                        if d_rev < d_fwd:
                            new_Y = new_Y[::-1]

                    # Adaptive kernel refresh if a new wire diameter came in
                    if sig.wire_diameter_px > 5.0:
                        wire_d_px = sig.wire_diameter_px
                        close_k, close_iters, open_k, _min_area = adaptive_kernels(wire_d_px)
                        log.info("[SAM3] wire_d=%.1fpx — kernels updated", wire_d_px)

                    tracker.Y = new_Y.astype(np.float32)
                    seg = np.linalg.norm(np.diff(tracker.Y, axis=0), axis=1)
                    spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                    tracker.sigma2 = max((spacing * 0.4) ** 2, 16.0)
                    _reset_velocity()
                    last_polyline = tracker.Y.copy()
                    sam3_rec.append(t_now, sig.polyline_ev, sig.wire_diameter_px)
                    sam3_pending = False
                    branch_tag = "A"
                    log.info("[SAM3] hard-replaced tracker.Y  σ²=%.1f", float(tracker.sigma2))

            # ─── 7. Branch B: CPD update (MOTION only) ─
            dt_skel_ms = 0.0
            dt_cpd_ms  = 0.0
            num_obs    = 0
            obs:       Optional[np.ndarray] = None
            init_path: Optional[np.ndarray] = None

            if state == MOTION and mode != "frame-only":
                # Event_only mask-count gate (skip CPD only, ρ_t still counted)
                if 30 <= mask_count <= int(wire_mask.size * 0.3):
                    t0 = time.perf_counter()

                    # obs = raw mask pixels (evidence of where the cable IS now)
                    # init_path = ordered tip-to-tip path (used by track() on reinit)
                    if wm_lo is not None:
                        # skeletonise low-res and map back; +0.5*(sc-1) centres
                        # each sample in its sc×sc block
                        __, init_path = mask_to_skeleton_obs(wm_lo, max_pts=300)
                        if init_path is not None:
                            init_path = init_path * float(sc) + 0.5 * (sc - 1)
                    else:
                        __, init_path = mask_to_skeleton_obs(wire_mask, max_pts=300)
                    obs = _resample_polyline(init_path, 250) if init_path is not None and len(init_path) >= 2 else None

                    if obs is not None and init_path is not None and len(obs) > 0:
                        # Trust-region: keep obs within 30 px of the skeleton.
                        dists, _ = cKDTree(init_path).query(obs, k=1)
                        obs_gated = obs[dists < 30.0]
                        if len(obs_gated) >= 10:
                            obs = obs_gated
                    dt_skel_ms = (time.perf_counter() - t0) * 1000

                    if obs is not None and len(obs) > 0:
                        # warm-start: shift init_path by the predicted
                        # displacement. tracker._V does per-node velocity.
                        if (tracker.Y is not None
                                and _prev_centroid is not None
                                and _prev_t is not None):
                            dt_vel = t_now - _prev_t
                            if dt_vel > 0:
                                cur_centroid = tracker.Y.mean(axis=0).astype(np.float32)
                                _velocity[:] = (cur_centroid - _prev_centroid) / dt_vel
                        if init_path is not None and _prev_t is not None:
                            predicted_shift = _velocity * float(t_now - _prev_t)
                            if np.linalg.norm(predicted_shift) > 0:
                                init_path = init_path + predicted_shift[None, :]

                        # ── CPD (handles init + steady-state internally) ──
                        t0 = time.perf_counter()
                        tracker.track(obs, init_path=init_path)
                        dt_cpd_ms = (time.perf_counter() - t0) * 1000

                        if tracker.Y is not None:
                            _prev_centroid = tracker.Y.mean(axis=0).astype(np.float32)
                            _prev_t        = t_now
                            last_polyline  = tracker.Y.copy()
                            num_obs        = len(obs)
                            branch_tag     = "B"

            # ─── 8. Visualisation ────────────────────────────────────────────
            t0 = time.perf_counter()
            if args.show:
                Visualization.visualize_wire(
                    R, mask=wire_mask, skel=None, win="debug", scale=1.0
                )
            if args.show or save_vis:
                frame = cv2.cvtColor(
                    np.clip(R * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
                )
                frame[wire_mask] = (0, 180, 0)
                hud = f"{state}  branch={branch_tag}  rho_t={rho_t_last:.0f}/{detector.rho_thr:.0f}"

                # masks/: base + HUD only, no overlays
                if save_vis and masks_dir is not None:
                    mask_img = frame.copy()
                    cv2.putText(mask_img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(
                        os.path.join(masks_dir, f"mask_{_iter:06d}_t{t_now:010d}us.png"),
                        mask_img,
                    )

                # cpd view: obs dots first, then init_path, then Y
                if obs is not None:
                    for pt in obs.astype(np.int32):
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 200, 0), -1)
                if init_path is not None and len(init_path) >= 2:
                    cv2.polylines(
                        frame,
                        [init_path.astype(np.int32).reshape(-1, 1, 2)],
                        False, (255, 100, 0), 1, cv2.LINE_AA,
                    )
                if last_polyline is not None:
                    cv2.polylines(
                        frame,
                        [last_polyline.astype(np.int32).reshape(-1, 1, 2)],
                        False, (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    for pt in last_polyline.astype(np.int32):
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 128, 255), -1)

                cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
                if args.show:
                    cv2.imshow("cpd", frame)
                if save_vis and results_dir is not None:
                    cv2.imwrite(
                        os.path.join(results_dir, f"cpd_{_iter:06d}_t{t_now:010d}us.png"),
                        frame,
                    )
            dt_vis_ms = (time.perf_counter() - t0) * 1000

            # ─── 9. Record this iteration ────────────────────────────────────
            dt_total_ms = (time.perf_counter() - t_iter) * 1000
            _t_total.append(dt_total_ms)
            poly_rec.append(t_now, last_polyline, state, branch_tag)
            timing.write(
                timestamp_us=t_now, state=state, branch=branch_tag,
                rho_t=f"{rho_t_last:.0f}", mask_count=mask_count,
                event_count=len(ev), num_obs=num_obs,
                filter_ms=f"{dt_filter_ms:.3f}",
                sae_ms=f"{dt_sae_ms:.3f}",
                recentness_ms=f"{dt_recentness_ms:.3f}",
                wiremask_ms=f"{dt_wiremask_ms:.3f}",
                elongated_ms=f"{dt_elong_ms:.3f}",
                skel_ms=f"{dt_skel_ms:.3f}",
                cpd_ms=f"{dt_cpd_ms:.3f}",
                vis_ms=f"{dt_vis_ms:.3f}",
                total_ms=f"{dt_total_ms:.3f}",
            )
            _iter += 1

    finally:
        stop_event.set()
        cancel_event.set()
        sam3_ch.request(-1)   # wake the worker so it can exit
        if sam3_worker is not None:
            sam3_worker.join(timeout=5.0)
        cv2.destroyAllWindows()

        poly_rec.save(getattr(args, "polyline_out",      "") or None)
        sam3_rec.save(getattr(args, "sam3_polyline_out", "") or None)
        timing.close()

        if _t_total:
            arr = np.asarray(_t_total, dtype=np.float32)
            log.info("═" * 72)
            log.info("[Avg Timing] %d iterations | mean=%.2fms  median=%.2fms  p95=%.2fms",
                     len(arr), float(arr.mean()), float(np.median(arr)),
                     float(np.percentile(arr, 95)))
            log.info("═" * 72)
        log.info("Pipeline stopped.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = config.parse_args()
    run(args)