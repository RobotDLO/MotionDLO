"""
id_switch_counter.py

Counts per-frame instance count and strict MOT-style ID switches across a
sequence of black-background instance-segmentation JPGs, where each
non-black colour encodes one tracker instance ID.

Method
------
1. Discover a canonical palette of N dominant non-background colours across
   the whole sequence (coarse quantisation + greedy merge) so JPEG bleed
   doesn't fragment one cable into many shades.
2. For each frame, snap every pixel to the nearest palette colour ->
   {colour: binary_mask}.
3. For each consecutive frame pair, Hungarian-match masks by IoU
   (independent of colours). For accepted pairs, count as ID switch iff the
   colour changes; unmatched instances at t/t+1 are deaths/births.

Usage
-----
    python id_switch_counter.py --dir /path/to/sequence \\
        --glob "*.jpg" --n-instances 4 --out switches.csv

Optional: --snap-check writes one snapped frame for visual inspection of
the palette before trusting the metrics.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Palette discovery (sequence-level)
# ---------------------------------------------------------------------------
def discover_palette(img_paths, n_instances, sample=30,
                     min_pixels_frac=0.001, bg=(0, 0, 0), merge_dist=40):
    """Aggregate dominant non-background colours across a sample of frames.
    Returns a list of BGR tuples sorted by total pixel count (most frequent
    first). The same palette is reused for the whole sequence so colour-ID
    mapping cannot drift between frames."""
    bg_arr = np.asarray(bg, dtype=np.uint8)
    step = max(1, len(img_paths) // sample)

    hist = {}  # quantised colour -> total count across sampled frames
    for p in img_paths[::step]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        min_pixels = int(min_pixels_frac * h * w)
        # 16-step quantisation per channel — kills JPEG noise
        coarse = (img & 0xF0).reshape(-1, 3)
        cols, cnts = np.unique(coarse, axis=0, return_counts=True)
        for c, n in zip(cols, cnts):
            if np.all(c == (bg_arr & 0xF0)) or n < min_pixels:
                continue
            key = tuple(int(v) for v in c)
            hist[key] = hist.get(key, 0) + int(n)

    # Sort by frequency, greedily merge near-duplicates within merge_dist
    ordered = sorted(hist.items(), key=lambda kv: -kv[1])
    palette = []
    for col, _ in ordered:
        col_arr = np.asarray(col, dtype=np.int16)
        if all(np.linalg.norm(col_arr - np.asarray(p, dtype=np.int16)) > merge_dist
               for p in palette):
            palette.append(col)
            if len(palette) >= n_instances:
                break
    return palette


# ---------------------------------------------------------------------------
# Per-frame extraction (palette snap)
# ---------------------------------------------------------------------------
def extract_instances(img_bgr, palette, min_pixels=200, bg=(0, 0, 0)):
    """Snap every pixel to the nearest palette colour (or background).
    Returns {palette_colour: binary_mask}."""
    bg_arr = np.asarray(bg, dtype=np.int16)
    pal = np.asarray(palette, dtype=np.int16)             # (K, 3)
    full = np.vstack([bg_arr[None, :], pal])              # (K+1, 3); 0 = bg
    h, w = img_bgr.shape[:2]
    pix = img_bgr.reshape(-1, 1, 3).astype(np.int16)      # (HW, 1, 3)
    d2 = ((pix - full[None, :, :]) ** 2).sum(axis=2)      # (HW, K+1)
    assign = np.argmin(d2, axis=1).reshape(h, w)

    instances = {}
    for k, c in enumerate(palette, start=1):
        mask = (assign == k)
        if mask.sum() >= min_pixels:
            instances[tuple(int(v) for v in c)] = mask
    return instances


# ---------------------------------------------------------------------------
# Cross-frame matching
# ---------------------------------------------------------------------------
def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0


def match_instances(inst_t, inst_tp1, iou_thr=0.2):
    """Hungarian assignment maximising IoU between masks at t and t+1.
    Returns (matched_pairs, unmatched_t, unmatched_tp1)."""
    cols_t, cols_tp1 = list(inst_t), list(inst_tp1)
    if not cols_t or not cols_tp1:
        return [], set(cols_t), set(cols_tp1)
    M = np.zeros((len(cols_t), len(cols_tp1)), dtype=np.float32)
    for i, ct in enumerate(cols_t):
        for j, cp in enumerate(cols_tp1):
            M[i, j] = iou(inst_t[ct], inst_tp1[cp])
    r, c = linear_sum_assignment(-M)
    matched, used_t, used_tp1 = [], set(), set()
    for i, j in zip(r, c):
        if M[i, j] >= iou_thr:
            matched.append((cols_t[i], cols_tp1[j], float(M[i, j])))
            used_t.add(cols_t[i])
            used_tp1.add(cols_tp1[j])
    return matched, set(cols_t) - used_t, set(cols_tp1) - used_tp1


# ---------------------------------------------------------------------------
# Sequence evaluation
# ---------------------------------------------------------------------------
def evaluate_sequence(img_paths, n_instances, iou_thr=0.2,
                      min_pixels=200, palette=None):
    if palette is None:
        palette = discover_palette(img_paths, n_instances=n_instances)
        print(f"Discovered palette ({len(palette)}): {palette}")

    rows, total_switches = [], 0
    prev_inst = None
    for k, p in enumerate(img_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read {p}")
        inst = extract_instances(img, palette, min_pixels=min_pixels)

        n_sw = n_birth = n_death = 0
        if prev_inst is not None:
            matched, lost, new = match_instances(prev_inst, inst, iou_thr)
            n_sw = sum(1 for ct, cp, _ in matched if ct != cp)
            n_death, n_birth = len(lost), len(new)
            total_switches += n_sw

        rows.append({"frame": k, "path": p.name,
                     "n_instances": len(inst),
                     "id_switches": n_sw,
                     "births": n_birth, "deaths": n_death})
        prev_inst = inst
    return rows, total_switches, palette


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------
def write_snap_check(img_path, palette, out_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    inst = extract_instances(img, palette)
    vis = np.zeros_like(img)
    for c, m in inst.items():
        vis[m] = c
    cv2.imwrite(str(out_path), vis)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path,
                    help="Directory of sequentially-named instance images")
    ap.add_argument("--glob", default="*.jpg",
                    help="File pattern (default: *.jpg)")
    ap.add_argument("--n-instances", type=int, required=True,
                    help="Expected number of cables in the scene")
    ap.add_argument("--out", type=Path, default=Path("id_switches.csv"))
    ap.add_argument("--iou-thr", type=float, default=0.2)
    ap.add_argument("--min-pixels", type=int, default=200)
    ap.add_argument("--merge-dist", type=float, default=40.0,
                    help="L2-BGR distance below which palette entries merge")
    ap.add_argument("--snap-check", type=Path, default=None,
                    help="Write a snapped version of the first frame to this "
                         "path for visual verification of the palette")
    args = ap.parse_args()

    paths = sorted(args.dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No images matched {args.dir}/{args.glob}")

    rows, total, palette = evaluate_sequence(
        paths, n_instances=args.n_instances,
        iou_thr=args.iou_thr, min_pixels=args.min_pixels,
    )

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    if args.snap_check is not None:
        write_snap_check(paths[0], palette, args.snap_check)
        print(f"Snap check:        {args.snap_check}")

    n_inst = [r["n_instances"] for r in rows]
    print(f"Frames:            {len(rows)}")
    print(f"Total ID switches: {total}")
    print(f"Total births:      {sum(r['births'] for r in rows)}")
    print(f"Total deaths:      {sum(r['deaths'] for r in rows)}")
    print(f"Mean #instances:   {np.mean(n_inst):.2f} "
          f"(min {np.min(n_inst)}, max {np.max(n_inst)})")
    print(f"CSV:               {args.out}")


if __name__ == "__main__":
    main()