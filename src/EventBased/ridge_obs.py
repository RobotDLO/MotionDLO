#!/usr/bin/env python3
"""Ridge-based wire mask for the event branch.

Two binarizers via --ridge-binarize: 'nms' (default) suppresses non-maxima
along the ridge normal to reject the motion trail by geometry; 'hysteresis'
is the original intensity seed-and-grow, kept for A/B. Both reduce the result
to the largest elongated components before returning a bool mask.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import meijering, apply_hysteresis_threshold


def _keep_largest_elongated(wire_bool: np.ndarray,
                            min_length: float,
                            min_aspect: float,
                            top_k: int = 5) -> np.ndarray:
    """Keep the top_k largest components (by area) passing the
    rotation-aware minAreaRect aspect/length test."""
    u8 = wire_bool.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return np.zeros_like(wire_bool)


    w_all = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    h_all = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    diag  = np.sqrt(w_all * w_all + h_all * h_all)
    maybe = set((np.nonzero(diag >= min_length)[0] + 1).tolist())  # 1-based labels


    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []   # (area, label)
    for c in contours:
        cx0, cy0 = int(c[0, 0, 0]), int(c[0, 0, 1])
        i = int(lab[cy0, cx0])
        if i not in maybe:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        major, minor = (rw, rh) if rw >= rh else (rh, rw)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if minor > 0 and major >= min_length and (major / minor) >= min_aspect:
            candidates.append((area, i))
    if not candidates:
        return np.zeros_like(wire_bool)
    candidates.sort(reverse=True)
    keep_lut = np.zeros(n, dtype=bool)
    keep_lut[[lbl for _, lbl in candidates[:top_k]]] = True
    return keep_lut[lab]


def _sobel_hessian_ridge(img: np.ndarray, sigmas) -> np.ndarray:
    """Bright-ridge strength: most-negative Sobel-Hessian eigenvalue, maxed across scales."""
    out = np.zeros_like(img, dtype=np.float32)
    for s in sigmas:
        g   = cv2.GaussianBlur(img, (0, 0), float(s))
        s2  = float(s) * float(s)
        Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * s2
        Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * s2
        Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * s2
        half = (Ixx + Iyy) * 0.5
        disc = np.sqrt(np.maximum(((Ixx - Iyy) * 0.5) ** 2 + Ixy * Ixy, 0.0))
        lam2 = half - disc            # most-negative eigenvalue
        out  = np.maximum(out, np.maximum(-lam2, 0.0))
    return out


# Cached pixel-coordinate grid for the NMS remap. 
_GRID = {"h": 0, "w": 0, "ys": None, "xs": None}


def _coord_grid(H: int, W: int):
    if H > _GRID["h"] or W > _GRID["w"]:
        h = max(H, _GRID["h"]); w = max(W, _GRID["w"])
        ys, xs = np.mgrid[0:h, 0:w]
        _GRID.update(h=h, w=w,
                     ys=ys.astype(np.float32), xs=xs.astype(np.float32))
    return _GRID["ys"][:H, :W], _GRID["xs"][:H, :W]


def _ridge_nms(img: np.ndarray, sigma: float, floor: float) -> np.ndarray:

    g   = cv2.GaussianBlur(img, (0, 0), float(sigma))
    s2  = float(sigma) * float(sigma)
    # Scale in-place to avoid an extra full-frame temporary per derivative.
    Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3); Ixx *= s2
    Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3); Iyy *= s2
    Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3); Ixy *= s2
    half = (Ixx + Iyy) * 0.5
    disc = np.sqrt(np.maximum(((Ixx - Iyy) * 0.5) ** 2 + Ixy * Ixy, 0.0))
    lam2 = half - disc                       # most-negative eigenvalue
    v    = np.maximum(-lam2, 0.0).astype(np.float32)   # bright-ridge strength
    vmax = float(v.max())
    if vmax <= 0.0:
        return np.zeros(img.shape, dtype=bool)
    v /= vmax

    # Normal = eigenvector of lam2 (across the bright ridge): (Ixy, lam2 - Ixx).
    nx = Ixy.astype(np.float32)
    ny = (lam2 - Ixx).astype(np.float32)
    nrm = np.sqrt(nx * nx + ny * ny) + 1e-12
    nx /= nrm
    ny /= nrm

    H, W = v.shape
    ys, xs = _coord_grid(H, W)

    def _along(dx, dy):
        return cv2.remap(v, xs + dx, ys + dy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    vp = _along(nx,  ny)     # one step along +normal
    vm = _along(-nx, -ny)    # one step along -normal
    return (v >= vp) & (v >= vm) & (v > float(floor))


def ridge_wire_mask(R: np.ndarray, wire_d_px: float, args,
                    speed_px: float = 0.0) -> np.ndarray:
    """Extract a cable-width wire band from a recentness map (see module docstring)."""
    R = np.asarray(R, dtype=np.float32)
    H, W = R.shape
    empty = np.zeros((H, W), dtype=bool)

    ys, xs = np.nonzero(R > 0)
    if len(ys) == 0:
        return empty

    sigma_lo      = float(getattr(args, "ridge_sigma_lo",      1.5))
    sigma_hi_mult = float(getattr(args, "ridge_sigma_hi_mult", 1.0))
    hyst_high     = float(getattr(args, "ridge_hyst_high",     0.12))
    hyst_low      = float(getattr(args, "ridge_hyst_low",      0.04))
    min_length    = float(getattr(args, "ridge_min_length",    40.0))
    min_aspect    = float(getattr(args, "ridge_min_aspect",     2.0))
    binarize      = str(getattr(args, "ridge_binarize", "nms")).lower()
    nms_floor     = float(getattr(args, "ridge_nms_floor", 0.10))
    band_dilate   = float(getattr(args, "ridge_band_dilate", -1.0))

    # Speed-adaptive threshold ramp: at high speed the wire smears into a
    # broader, dimmer trail. Lower seeds + wider sigma match the smear.
    speed_lo = float(getattr(args, "ridge_speed_lo", 10.0))
    speed_hi = float(getattr(args, "ridge_speed_hi", 35.0))
    if speed_px >= speed_hi:
        alpha = 1.0
    elif speed_px <= speed_lo:
        alpha = 0.0
    else:
        alpha = (speed_px - speed_lo) / (speed_hi - speed_lo)

    if alpha > 0.0:
        hyst_high_fast    = float(getattr(args, "ridge_hyst_high_fast",    0.08))
        hyst_low_fast     = float(getattr(args, "ridge_hyst_low_fast",     0.04))
        sigma_hi_mult_fast = float(getattr(args, "ridge_sigma_hi_mult_fast", 2.5))
        hyst_high     = hyst_high     * (1 - alpha) + hyst_high_fast     * alpha
        hyst_low      = hyst_low      * (1 - alpha) + hyst_low_fast      * alpha
        sigma_hi_mult = sigma_hi_mult * (1 - alpha) + sigma_hi_mult_fast * alpha

    # Scale matched to the wire half-width so the tubeness filter peaks on a
    # cable of diameter ~wire_d_px.
    sigma_hi = max(sigma_lo, float(wire_d_px) * sigma_hi_mult)
    sigmas = np.linspace(sigma_lo, sigma_hi, 4)

    # Crop to the active-pixel bbox (+ Hessian-footprint margin).
    margin = int(round(3.0 * sigma_hi)) + 2
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(H, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(W, int(xs.max()) + margin + 1)
    R_crop = R[y0:y1, x0:x1]

    if binarize == "hysteresis":
        # ---- original intensity seed-and-grow (kept for A/B) -----------------
        ridge_filter = str(getattr(args, "ridge_filter", "sobel")).lower()
        if ridge_filter == "meijering":
            v = meijering(R_crop, sigmas=sigmas, black_ridges=False)
        else:
            v = _sobel_hessian_ridge(R_crop, sigmas)
        peak = float(v.max())
        if peak <= 0.0:
            return empty
        v /= peak
        rc = apply_hysteresis_threshold(v, hyst_low, hyst_high)
    else:
        # ---- NMS along the ridge normal (default) ----------------------------
        sigma_nms = max(sigma_lo, 0.5 * float(wire_d_px))
        rc = _ridge_nms(R_crop, sigma_nms, nms_floor)

    if not rc.any():
        return empty

    # Optional morphological close at high speed to bridge the smear gaps.
    if alpha > 0.0:
        close_fast_px = float(getattr(args, "ridge_close_fast_px", 7.0))
        close_r = max(1, int(round(close_fast_px * alpha)))
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        rc = cv2.morphologyEx(rc.astype(np.uint8), cv2.MORPH_CLOSE, k_close).astype(bool)

    # Dilate the (thin) ridge to a uniform cable-width band. Auto radius =
    # wire_d_px / 2; bridges sub-band NMS gaps as a side effect.
    if band_dilate <= 0.0:
        rad = max(1, int(round(0.5 * float(wire_d_px))))
    else:
        rad = int(round(band_dilate))
    if binarize != "hysteresis" and rad >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        rc = cv2.dilate(rc.astype(np.uint8), k).astype(bool)

    top_k = int(getattr(args, "ridge_top_k", 5))
    rc = _keep_largest_elongated(rc, min_length=min_length, min_aspect=min_aspect, top_k=top_k)
    if not rc.any():
        return empty

    wire_mask = np.zeros((H, W), dtype=bool)
    wire_mask[y0:y1, x0:x1] = rc
    return wire_mask