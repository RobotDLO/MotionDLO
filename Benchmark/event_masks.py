"""Extract one wire mask per manifest frame from an events.raw recording.

For every (frame_name -> timestamp_us) entry in manifest.json the script
accumulates events up to that timestamp and writes the resulting binary mask
to <out_dir>/<frame_stem>.png so masks line up 1:1 with the RGB frames.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import EventBased.Filter as ev_filter
from EventBased.sae import sae
from metavision_core.event_io import EventsIterator
import config

from Minimal_example import _keep_elongated, adaptive_kernels


WIRE_DIAMETER_PX = 50  # measured cable thickness, same value as Minimal_example.py


def extract_masks(raw_path: str, out_dir: str, args) -> None:
    manifest_path = os.path.join(os.path.dirname(raw_path), "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    schedule = sorted(manifest.items(), key=lambda kv: kv[1])  # [(name, t_us), ...]
    os.makedirs(out_dir, exist_ok=True)

    it = EventsIterator(input_path=raw_path, delta_t=args.delta_t_us, mode="delta_t")
    h, w = it.get_size()

    T_on = np.full((h, w), -10**18, dtype=np.int64)
    T_off = np.full((h, w), -10**18, dtype=np.int64)

    _, _, _, _, stc_filter, stc_out = ev_filter.filter_events(it, w, h, args)
    close_k, close_iters, open_k, _ = adaptive_kernels(WIRE_DIAMETER_PX)

    next_idx = 0  # pointer into schedule
    last_mask = np.zeros((h, w), dtype=np.uint8)

    def save_mask(name: str, mask: np.ndarray) -> None:
        stem = os.path.splitext(name)[0]
        # BGR image: wire pixels rendered green, background black
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb[mask.astype(bool)] = (0, 255, 0)
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), rgb)

    for evs in it:
        if len(evs) == 0:
            continue

        ev = evs
        if stc_filter is not None:
            stc_filter.process_events(ev, stc_out)
            ev = stc_out.numpy()
        if len(ev) == 0:
            continue

        xs = ev["x"].astype(np.int32, copy=False)
        ys = ev["y"].astype(np.int32, copy=False)
        ts = ev["t"].astype(np.int64, copy=False)
        ps = ev["p"]
        on = ps if ps.dtype == np.bool_ else (ps > 0)
        if np.any(on):
            T_on[ys[on], xs[on]] = ts[on]
        if np.any(~on):
            T_off[ys[~on], xs[~on]] = ts[~on]

        t_now = int(ts[-1])

        # Skip mask computation until we are at/after the next requested frame timestamp
        if next_idx >= len(schedule) or t_now < schedule[next_idx][1]:
            continue

        dt_batch_us = int(ts[-1] - ts[0])
        if dt_batch_us <= 0:
            continue
        event_rate = len(ev) / (dt_batch_us * 1e-6)
        tau_us_dyn = ev_filter.adaptive_tau_us(event_rate, tau_min=800, tau_max=10000)

        R = sae.compute_recentness(T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn)
        R_u8 = np.clip(R * 255, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(R_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        wire_thr = max(float(otsu_val) / 255.0, 0.05)
        wire_mask = (R >= wire_thr).astype(np.uint8)
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, close_k, iterations=close_iters)
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, open_k, iterations=1)
        wire_mask = _keep_elongated(wire_mask.astype(bool), min_aspect=3.5).astype(np.uint8)
        last_mask = wire_mask

        # Emit a mask for every scheduled timestamp we have now passed
        while next_idx < len(schedule) and t_now >= schedule[next_idx][1]:
            save_mask(schedule[next_idx][0], last_mask)
            next_idx += 1

    # Pad any remaining frames (event stream ended early) with the last mask
    while next_idx < len(schedule):
        save_mask(schedule[next_idx][0], last_mask)
        next_idx += 1

    print(f"[event_masks] wrote {len(schedule)} masks to {out_dir}")


RAW_PATH = "/path/to/Dataset/Single Cable/Orange_Cable/Speed_50/1/events.raw"
OUT_DIR  = "/path/to/Results/Event_based/Single_cable/Orange_cable/speed_50/1/frames"


def main():
    pipeline_args = config.parse_args()
    extract_masks(RAW_PATH, OUT_DIR, pipeline_args)


if __name__ == "__main__":
    main()
