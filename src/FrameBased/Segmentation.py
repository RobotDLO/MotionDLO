"""
SAM3-based cable segmentation.

Usage
-----
    seg = SAM3Segmenter()
    mask = seg.segment(bgr_image)          # numpy (H,W) uint8, values 0/255
    mask, t = seg.segment_timed(bgr_image) # also returns elapsed seconds
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import binary_closing, remove_small_holes, remove_small_objects, disk


# ── Default prompts ────────────────────────────────────────────────────────────

DEFAULT_PROMPTS: List[str] = [
    "cable", "wire", "electrical cable", "power cable",
    "tube", "hose", "cord", "rope",
]

# ── Default thresholds ─────────────────────────────────────────────────────────

MASK_THRESHOLD    = 0.5
MIN_PIXEL_COUNT   = 300
CLOSING_DISK      = 6
MIN_AREA_FRAC     = 0.0005
MIN_AREA_ABS      = 400
FILL_HOLES_AREA   = 300
MIN_FINAL_BLOB    = 12000
MAX_OVERLAP_RATIO = 0.7


class SAM3Segmenter:
    """
    Reusable wrapper around SAM3 for cable segmentation.

    The model is loaded lazily on the first call to ``segment()``.

    Parameters
    ----------
    device : str or None
        ``"cuda"`` / ``"cpu"``.  Auto-detected when None.
    prompts : list[str]
        Text prompts passed to the SAM3 text encoder.
    mask_threshold : float
        Sigmoid threshold for converting logits to binary mask.
    min_pixel_count : int
        Minimum foreground pixels for a raw mask to be kept before merging.
    min_final_blob : int
        Minimum blob area (px) after cleaning; smaller blobs are dropped.
    """

    def __init__(
        self,
        device:          Optional[str] = None,
        prompts:         List[str]     = DEFAULT_PROMPTS,
        mask_threshold:  float         = MASK_THRESHOLD,
        min_pixel_count: int           = MIN_PIXEL_COUNT,
        min_final_blob:  int           = MIN_FINAL_BLOB,
    ):
        if device is None:
            import torch as _t
            device = "cuda" if _t.cuda.is_available() else "cpu"
        self.device          = device
        self.prompts         = prompts
        self.mask_threshold  = mask_threshold
        self.min_pixel_count = min_pixel_count
        self.min_final_blob  = min_final_blob
        self._processor      = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment cables in a single frame.
        """
        self._ensure_loaded()
        img_pil = self._to_pil(image)
        return self._run(img_pil)

    def segment_timed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        t0   = time.perf_counter()
        mask = self.segment(image)
        return mask, time.perf_counter() - t0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._processor is not None:
            return
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model           = build_sam3_image_model(device=self.device, eval_mode=True)
        self._processor = Sam3Processor(model)

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    def _run(self, img_pil: Image.Image) -> np.ndarray:
        h, w  = img_pil.size[1], img_pil.size[0]
        state = self._processor.set_image(img_pil)
        raw_masks = []

        for prompt in self.prompts:
            result = self._processor.set_text_prompt(state=state, prompt=prompt)
            masks  = result.get("masks", None)
            if masks is None or masks.numel() == 0:
                continue
            for i in range(masks.shape[0]):
                m  = masks[i, 0].detach().cpu().numpy()
                mb = (m > self.mask_threshold).astype(np.uint8) * 255
                if int(mb.sum()) >= self.min_pixel_count:
                    raw_masks.append(mb)

        raw_masks = SAM3Segmenter._deduplicate(raw_masks)

        merged = np.zeros((h, w), np.uint8)
        for m in raw_masks:
            merged = np.maximum(merged, m)

        if merged.sum() > 0:
            merged = SAM3Segmenter._clean(merged)
            merged = SAM3Segmenter._drop_small_blobs(merged, self.min_final_blob)

        return merged

    # ── Static mask-processing helpers ────────────────────────────────────────

    @staticmethod
    def _clean(mask: np.ndarray) -> np.ndarray:
        m = (mask > 127).astype(np.uint8)
        m = binary_closing(m, footprint=disk(CLOSING_DISK)).astype(np.uint8)
        min_area = max(int(m.size * MIN_AREA_FRAC), MIN_AREA_ABS)
        m = remove_small_objects(m.astype(bool), min_size=min_area)
        m = remove_small_holes(m, area_threshold=FILL_HOLES_AREA)
        m = binary_closing(m, footprint=disk(3))
        return m.astype(np.uint8) * 255

    @staticmethod
    def _drop_small_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        result = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                result[labels == i] = 255
        return result

    @staticmethod
    def _deduplicate(masks: List[np.ndarray], max_iou: float = MAX_OVERLAP_RATIO) -> List[np.ndarray]:
        if len(masks) <= 1:
            return masks
        masks = sorted(masks, key=lambda m: int(m.sum()), reverse=True)
        kept: List[np.ndarray] = []
        for m in masks:
            if not any(SAM3Segmenter._iou(m, k) > max_iou for k in kept):
                kept.append(m)
        return kept

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0