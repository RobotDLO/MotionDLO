#!/usr/bin/env python3
"""
Overlay GT polyline and tracker polyline on an event-space debug image,
so you can see by eye whether they're tracking the same cable.

This is the diagnostic to run when compute_pwl_error.py reports
implausibly large errors.

Usage:
    python overlay_gt_vs_tracker.py \\
        --gt-dir      /path/to/gt_nodes_event_space \\
        --tracker-dir /path/to/tracker_event_space \\
        --bg-image    sam3_debug/event_debug_snapshot_*.png \\
        --time-sec    0.138 \\
        --out         overlay.png
"""

import argparse
import os

import cv2
import numpy as np


def load_sequence(folder):
    ts_path = os.path.join(folder, "timestamps.txt")
    entries = []
    with open(ts_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fname, ts = line.split()[0], float(line.split()[1])
            npy_path = os.path.join(folder, fname)
            if os.path.exists(npy_path):
                entries.append((ts, np.load(npy_path)))
    entries.sort(key=lambda e: e[0])
    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def load_mask_paths(folder, fallback_ts_dir=None):
    ts_path = os.path.join(folder, "timestamps.txt")
    if os.path.exists(ts_path):
        entries = []
        with open(ts_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fname, ts = line.split()[0], float(line.split()[1])
                path = os.path.join(folder, fname)
                if os.path.exists(path):
                    entries.append((ts, path))
        entries.sort(key=lambda e: e[0])
        return np.array([e[0] for e in entries]), [e[1] for e in entries]

    if fallback_ts_dir is None:
        raise FileNotFoundError(
            f"{ts_path} not found and no fallback timestamps dir given")
    fb_ts = os.path.join(fallback_ts_dir, "timestamps.txt")
    if not os.path.exists(fb_ts):
        raise FileNotFoundError(
            f"Neither {ts_path} nor {fb_ts} exists")
    ts_by_stem = {}
    with open(fb_ts) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fname, ts = line.split()[0], float(line.split()[1])
            ts_by_stem[os.path.splitext(fname)[0]] = ts
    entries = []
    for f in sorted(os.listdir(folder)):
        path = os.path.join(folder, f)
        if not os.path.isfile(path):
            continue
        stem = os.path.splitext(f)[0]
        if stem in ts_by_stem:
            entries.append((ts_by_stem[stem], path))
    if not entries:
        raise RuntimeError(
            f"No mask files in {folder} match any entry in {fb_ts}")
    entries.sort(key=lambda e: e[0])
    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def nearest_index(sorted_ts, query_t):
    idx = np.searchsorted(sorted_ts, query_t)
    if idx == 0:              return 0
    if idx >= len(sorted_ts): return len(sorted_ts) - 1
    l, r = sorted_ts[idx - 1], sorted_ts[idx]
    return idx - 1 if abs(query_t - l) <= abs(query_t - r) else idx


def draw_polyline(img, pts, color, line_w=5, halo_w=10, dot_r=4):
    pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], False, (0, 0, 0), halo_w, cv2.LINE_AA)
    cv2.polylines(img, [pts_i], False, color,     line_w, cv2.LINE_AA)
    for p in pts.astype(np.int32):
        cv2.circle(img, (int(p[0]), int(p[1])), dot_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), dot_r,     color,     -1, cv2.LINE_AA)


def draw_legend(img, entries, org=(15, 15), pad=10, line_len=40, line_w=3,
                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_th=1):
    text_sizes = [cv2.getTextSize(t, font, font_scale, font_th)[0] for _, t in entries]
    text_w = max(w for w, _ in text_sizes)
    text_h = max(h for _, h in text_sizes)
    row_h  = text_h + 12
    box_w  = pad + line_len + 8 + text_w + pad
    box_h  = pad + row_h * len(entries) + pad - 6
    x0, y0 = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1, cv2.LINE_AA)
    for i, (color, text) in enumerate(entries):
        cy = y0 + pad + row_h // 2 + i * row_h - 4
        lx = x0 + pad
        cv2.line(img, (lx, cy), (lx + line_len, cy), color, line_w, cv2.LINE_AA)
        cv2.putText(img, text, (lx + line_len + 8, cy + text_h // 2 - 1),
                    font, font_scale, (255, 255, 255), font_th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir",      required=True)
    ap.add_argument("--tracker-dir", required=True)
    ap.add_argument("--bg-image",    required=True,
                    help="Event-space background image (e.g. a "
                         "sam3_debug/event_debug_snapshot_*.png)")
    ap.add_argument("--time-sec",    type=float, required=True,
                    help="Query time in seconds. The nearest GT and tracker "
                         "polylines to this time will be plotted.")
    ap.add_argument("--mask-dir",    default=None,
                    help="Optional folder of cable masks (image files + "
                         "timestamps.txt). The nearest mask to --time-sec is "
                         "drawn as a translucent gray underlay so you can see "
                         "where the cable actually is in the frame.")
    ap.add_argument("--mask-alpha",  type=float, default=0.45,
                    help="Opacity of the mask underlay (0..1). Default 0.45.")
    ap.add_argument("--out",         default="overlay.png")
    ap.add_argument("--gt-time-offset", type=float, default=0.0,
                    help="Additive correction applied to GT timestamps: "
                         "gt_t' = gt_t * scale + offset. Must match the "
                         "correction used in compute_pwl_error.py.")
    ap.add_argument("--gt-time-scale",  type=float, default=1.0,
                    help="Multiplicative correction applied to GT timestamps.")
    args = ap.parse_args()

    gt_ts,      gt_polys      = load_sequence(args.gt_dir)
    tracker_ts, tracker_polys = load_sequence(args.tracker_dir)

    if args.gt_time_scale != 1.0 or args.gt_time_offset != 0.0:
        gt_ts = gt_ts * args.gt_time_scale + args.gt_time_offset
        order = np.argsort(gt_ts)
        gt_ts = gt_ts[order]
        gt_polys = [gt_polys[i] for i in order]
        print(f"[INFO] GT timestamp correction: "
              f"gt_t' = gt_t * {args.gt_time_scale:.4f} + {args.gt_time_offset:+.4f}")
        print(f"[INFO] GT range after: {gt_ts[0]:.3f}s → {gt_ts[-1]:.3f}s")

    i_gt = nearest_index(gt_ts,      args.time_sec)
    i_tr = nearest_index(tracker_ts, args.time_sec)

    gt = gt_polys[i_gt]
    tr = tracker_polys[i_tr]

    print(f"[INFO] Query time     : {args.time_sec:.3f} s")
    print(f"[INFO] GT      ts={gt_ts[i_gt]:.3f}s  shape={gt.shape}")
    print(f"[INFO] Tracker ts={tracker_ts[i_tr]:.3f}s  shape={tr.shape}")
    if not np.isnan(gt).all():
        print(f"[INFO] GT      bbox: x=[{gt[:,0].min():.0f}, {gt[:,0].max():.0f}]  "
              f"y=[{gt[:,1].min():.0f}, {gt[:,1].max():.0f}]")
    if not np.isnan(tr).all():
        print(f"[INFO] Tracker bbox: x=[{tr[:,0].min():.0f}, {tr[:,0].max():.0f}]  "
              f"y=[{tr[:,1].min():.0f}, {tr[:,1].max():.0f}]")

    mask = None
    if args.mask_dir:
        mask_ts, mask_paths = load_mask_paths(args.mask_dir,
                                              fallback_ts_dir=args.gt_dir)
        i_mask = nearest_index(mask_ts, args.time_sec)
        mask = cv2.imread(mask_paths[i_mask], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Could not read mask {mask_paths[i_mask]} — skipping")
        else:
            print(f"[INFO] Mask    ts={mask_ts[i_mask]:.3f}s  "
                  f"path={os.path.basename(mask_paths[i_mask])}  shape={mask.shape}")

    img = cv2.imread(args.bg_image)
    if img is None:
        pieces = []
        if not np.isnan(gt).all():
            pieces.append(gt[~np.isnan(gt).any(axis=1)])
        if not np.isnan(tr).all():
            pieces.append(tr[~np.isnan(tr).any(axis=1)])
        if pieces:
            allp = np.concatenate(pieces, axis=0)
            margin = 40
            w = max(1280, int(np.ceil(allp[:, 0].max())) + margin)
            h = max(720,  int(np.ceil(allp[:, 1].max())) + margin)
        else:
            w, h = 1280, 720
        if mask is not None:
            h = max(h, mask.shape[0])
            w = max(w, mask.shape[1])
        img = np.zeros((h, w, 3), dtype=np.uint8)
        print(f"[WARN] Could not read {args.bg_image} — using {w}x{h} black background")

    if mask is not None:
        H, W = img.shape[:2]
        mfit = np.zeros((H, W), dtype=mask.dtype)
        mh, mw = mask.shape[:2]
        mfit[:min(mh, H), :min(mw, W)] = mask[:min(mh, H), :min(mw, W)]
        bin_mask = mfit > 0
        tint = np.full_like(img, (200, 200, 200))   # light gray
        alpha = float(np.clip(args.mask_alpha, 0.0, 1.0))
        blended = cv2.addWeighted(tint, alpha, img, 1.0 - alpha, 0)
        img[bin_mask] = blended[bin_mask]

    scale  = max(1.0, max(img.shape[0], img.shape[1]) / 1280)
    line_w = int(round(6  * scale))
    halo_w = int(round(12 * scale))
    dot_r  = int(round(5  * scale))

    GT_COLOR = (80, 230, 80)     # green   (BGR)
    TR_COLOR = (0, 140, 255)     # orange  (BGR)

    if not np.isnan(gt).all() and len(gt) >= 2:
        draw_polyline(img, gt, GT_COLOR, line_w=line_w, halo_w=halo_w, dot_r=dot_r)
    if not np.isnan(tr).all() and len(tr) >= 2:
        draw_polyline(img, tr, TR_COLOR, line_w=line_w, halo_w=halo_w, dot_r=dot_r)

    draw_legend(img, [(GT_COLOR, "Ground truth"), (TR_COLOR, "Tracker")])

    cv2.imwrite(args.out, img)
    print(f"\n[Done] Wrote {args.out}")


if __name__ == "__main__":
    main()