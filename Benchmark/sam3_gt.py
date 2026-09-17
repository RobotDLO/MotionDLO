"""
Benchmark script: run SAM3 segmentation over a folder of images and save
  - binary mask  (white cable on black background)
  - overlay      (green mask blended onto the original image)
  - results.csv  (per-frame timing + detection flag)

Usage
-----
    python run_sam3_segmentation.py \
        --input_dir  /path/to/frames \
        --output_dir /path/to/out \
        --prompt     "(cable)"
"""

import argparse
import csv
import gc
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── resolve Segmentation.py from src/ ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "FrameBased"))
from Segmentation import SAM3Segmenter

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/path/to/Dataset/Single Cable/Orange_Cable/Speed_100/2/frames"
DEFAULT_OUTPUT = "/path/to/Results/Ground_truth/Sam3/Single Cable/Orange_Cable/Speed_100/2"
DEFAULT_PROMPT = "(cable)"


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths, key=lambda p: os.path.basename(p))


def save_mask(mask: np.ndarray, path: str):
    """Binary mask — white cable on black background."""
    cv2.imwrite(path, mask)


def save_overlay(bgr_image: np.ndarray, mask: np.ndarray, path: str):
    """Green mask blended onto the original image."""
    overlay = bgr_image.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)
    cv2.imwrite(path, overlay)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt",     default=DEFAULT_PROMPT)
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()

    mask_dir    = os.path.join(args.output_dir, "masks")
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(mask_dir,    exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {args.input_dir}")
    print(f"Found {len(image_paths)} images — prompt: \"{args.prompt}\"")

    seg = SAM3Segmenter(prompts=[args.prompt], mask_threshold=args.confidence)

    csv_path = os.path.join(args.output_dir, "results.csv")
    t_seg, t_save = [], []

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "segmentation_s", "save_s", "total_s", "detected"])

        for image_path in image_paths:
            stem = Path(image_path).stem
            bgr  = cv2.imread(image_path)

            # segmentation
            t0     = time.perf_counter()
            mask   = seg.segment(bgr)
            dt_seg = time.perf_counter() - t0

            detected = mask.sum() > 0

            # save mask + overlay
            t0 = time.perf_counter()
            save_mask(mask,    os.path.join(mask_dir,    f"mask_{stem}.png"))
            save_overlay(bgr, mask, os.path.join(overlay_dir, f"overlay_{stem}.png"))
            dt_save = time.perf_counter() - t0

            total = dt_seg + dt_save
            t_seg.append(dt_seg)
            t_save.append(dt_save)

            writer.writerow([stem, f"{dt_seg:.4f}", f"{dt_save:.4f}", f"{total:.4f}", detected])
            print(f"[{stem}]  seg: {dt_seg:.3f}s  save: {dt_save:.3f}s  detected: {detected}")

            gc.collect()

    n   = len(t_seg)
    avg = lambda lst: sum(lst) / n
    print(f"\n{'='*50}")
    print(f"{'Segmentation (avg):':<25} {avg(t_seg)*1000:.2f} ms")
    print(f"{'Save outputs (avg):':<25} {avg(t_save)*1000:.2f} ms")
    print(f"{'Total (avg):':<25} {(avg(t_seg)+avg(t_save))*1000:.2f} ms")
    print(f"{'='*50}")
    print(f"Masks:    {mask_dir}")
    print(f"Overlays: {overlay_dir}")
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    main()
