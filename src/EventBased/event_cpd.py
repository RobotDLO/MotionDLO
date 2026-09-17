"""
event_cpd.py
============
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
import numpy as np
import cv2
from scipy.ndimage import label, gaussian_filter1d
def _skeletonize(mask: np.ndarray) -> np.ndarray:
   
    u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    return cv2.ximgproc.thinning(u8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN) // 255
from EventBased.Polyline import Polyline

#  Hyper-parameters

@dataclass
class Params:
    num_nodes:        int   = 20     # M  – number of curve control nodes
    beta:             float = 20.0   # geodesic kernel width, main step (pixels)
    lam:              float = 0.0001 # regularisation strength λ
    alpha:            float = 0.01   # correspondence-prior weight α
    mu:               float = 0.1    # outlier ratio ∈ (0, 1)
    k_vis:            float = 0.05   # visibility decay  exp(-k_vis·d_min)
    max_iter:         int   = 25     # max EM iterations per frame
    tol:              float = 1e-4   # convergence threshold (mean node shift, px)
    # Pre-processing step
    beta_pre_proc:    float = 1.0
    lam_pre_proc:     float = 0.0001
    # Soft obs weighting 
    prune_threshold:  float = 80.0   # 1-σ of the soft weight Gaussian (px)
    # Motion constraint: chain Laplacian L^T L added to M-step A matrix.
    # Penalises ||W[i+1] - W[i]||² so adjacent nodes deform similarly.
    mu_motion:        float = 0.5
    # CPD correction scale: Y_new = Y0 + correction_scale*(CPD_result - Y0)
    # <1 → velocity carries bulk motion, CPD fine-tunes; prevents overshoot.
    correction_scale: float = 1.0



#  Arc-length resampling


def resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.tile(pts[0], (n, 1)).astype(np.float32)

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc      = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = arc[-1]
    if total < 1e-9:
        return np.tile(pts[0], (n, 1)).astype(np.float32)

    t_new = np.linspace(0.0, total, n)
    x_new = np.interp(t_new, arc, pts[:, 0])
    y_new = np.interp(t_new, arc, pts[:, 1])
    return np.column_stack([x_new, y_new]).astype(np.float32)



#  Skeleton path extraction  


def _endpoint_pixels(skel: np.ndarray):
    """Return (ys, xs) of degree-1 pixels in a 1-px-wide 8-connected skeleton."""
    k  = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k)
    ep = (nb == 2) & skel.astype(bool)
    return np.where(ep)


def _bfs_farthest(skel: np.ndarray, sy: int, sx: int):
    """BFS from (sy, sx). Returns (farthest_yx, parent_dict).
    """
    H, W    = skel.shape
    visited = np.zeros_like(skel, dtype=bool)
    visited[sy, sx] = True
    parent  = {}          # (ny, nx) -> (cy, cx)
    q       = deque([(sy, sx)])
    last    = (sy, sx)

    while q:
        cy, cx = q.popleft()
        last   = (cy, cx)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W \
                        and skel[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    parent[(ny, nx)] = (cy, cx)
                    q.append((ny, nx))

    return last, parent


def _skel_to_longest_path(skel: np.ndarray) -> np.ndarray | None:
    """Return the diameter path of the skeleton (tip-to-tip longest path).
    """
    lab, n_comp = label(skel)
    if n_comp == 0:
        return None
    counts    = np.bincount(lab.ravel())
    counts[0] = 0
    skel      = (lab == int(np.argmax(counts)))

    ys, xs = np.where(skel)
    if len(xs) < 2:
        return None

    ep_ys, ep_xs = _endpoint_pixels(skel)
    if len(ep_xs) >= 1:
        start = (int(ep_ys[0]), int(ep_xs[0]))
    else:
        idx   = int(np.argmin(xs))
        start = (int(ys[idx]), int(xs[idx]))

    # Pass 1: find the farthest point from start
    far1, _      = _bfs_farthest(skel, *start)
    # Pass 2: find the farthest point from far1, keeping parent pointers
    far2, parent = _bfs_farthest(skel, *far1)

    path = [far2]
    cur  = far2
    while cur != far1:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    return np.array([[x, y] for y, x in path], dtype=np.float32)


def extract_longest_path(mask: np.ndarray) -> np.ndarray | None:
    skel = _skeletonize(mask)
    return _skel_to_longest_path(skel)

#  Mask → point cloud / centerline
def mask_to_pointcloud(wire_mask: np.ndarray, max_pts: int = 500) -> np.ndarray | None:

    ys, xs = np.where(wire_mask)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts
    
def mask_to_skeleton_obs(
    wire_mask: np.ndarray, max_pts: int = 300
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # ── Primary: PCA centerline (gap-tolerant) ───────────────────────────
    centerline = mask_to_centerline(wire_mask, n_bins=80)

    # ── Refinement: aggressive-gap-bridged skeleton ───────────────────────
    skel_path: np.ndarray | None = None
    try:
        skel = _skeletonize(wire_mask)

        # Two rounds: first bridge small gaps (5 px), then larger ones (11 px)
        for ksize in (5, 11):
            k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            skel = cv2.dilate(skel, k, iterations=1)
            skel = _skeletonize(skel)

        skel      = Polyline.largest_component(skel.astype(bool))
        skel      = Polyline.prune_skeleton(skel, min_branch_length=10)
        skel_path = Polyline.skeleton_to_ordered_path(skel)
    except Exception:
        skel_path = None

    
    init_path: np.ndarray | None = None
    if centerline is not None and skel_path is not None:
        cl_span = float(np.linalg.norm(
            centerline.max(axis=0) - centerline.min(axis=0)))
        sk_span = float(np.linalg.norm(
            skel_path.max(axis=0) - skel_path.min(axis=0)))
       
        init_path = skel_path if sk_span >= 0.85 * cl_span else centerline
    elif centerline is not None:
        init_path = centerline
    elif skel_path is not None:
        init_path = skel_path

    if init_path is None or len(init_path) < 4:
        return None, None

    # ── Uniform subsampling for CPD observations ──────────────────────────
    N   = min(max_pts, len(init_path))
    idx = np.round(np.linspace(0, len(init_path) - 1, N)).astype(int)
    obs = init_path[idx].astype(np.float32)

    h_mask, w_mask = wire_mask.shape
    init_path[:, 0] = np.clip(init_path[:, 0], 0, w_mask - 1)
    init_path[:, 1] = np.clip(init_path[:, 1], 0, h_mask - 1)
    obs[:, 0] = np.clip(obs[:, 0], 0, w_mask - 1)
    obs[:, 1] = np.clip(obs[:, 1], 0, h_mask - 1)
    return obs, init_path


def mask_to_centerline(wire_mask: np.ndarray, n_bins: int = 80) -> np.ndarray | None:

    
    if wire_mask.dtype == np.bool_ and wire_mask.flags.c_contiguous:
        u8 = wire_mask.view(np.uint8)
    else:
        u8 = wire_mask.astype(np.uint8)
    nz = cv2.findNonZero(u8)
    if nz is None or len(nz) < 20:
        return None

    pts = nz[:, 0, :].astype(np.float64)          # (N,2) x,y
    mu  = pts.mean(axis=0)
    X   = pts - mu

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    v = Vt[0]                       # unit vector along wire
    n = np.array([-v[1], v[0]])     # unit normal (across wire)

    t = X @ v   # along-wire coordinate per pixel
    s = X @ n   # across-wire coordinate per pixel

  
    if t.max() - t.min() < 1.0:
        return None

    # Equal-count (quantile) binning
    sort_idx   = np.argsort(t)
    t_sorted   = t[sort_idx]
    s_sorted   = s[sort_idx]
    N_px       = len(t_sorted)
    bin_size   = max(N_px // n_bins, 1)


    starts   = np.arange(0, N_px, bin_size)
    counts   = np.minimum(starts + bin_size, N_px) - starts
    n_full   = N_px // bin_size            # bins of exactly bin_size pixels

    t_centers = np.add.reduceat(t_sorted, starts) / counts
    s_centers = np.empty(len(starts))
    if n_full > 0:
        s_centers[:n_full] = np.median(
            s_sorted[:n_full * bin_size].reshape(n_full, bin_size), axis=1)
    if len(starts) > n_full:               # trailing partial bin
        s_centers[n_full] = np.median(s_sorted[n_full * bin_size:])
    t_centers[-1] = t_sorted[-1]
    t_centers[0]  = t_sorted[0]

    if len(starts) < 4:
        return None

    cl = (mu[None, :]
          + t_centers[:, None] * v[None, :]
          + s_centers[:, None] * n[None, :]).astype(np.float32)
    # 'nearest' pads with the edge value; the default 'reflect' pulls the
    # tip points inward
    cl[:, 0] = gaussian_filter1d(cl[:, 0], sigma=3.0, mode='nearest')
    cl[:, 1] = gaussian_filter1d(cl[:, 1], sigma=3.0, mode='nearest')
    return cl



#  Geodesic kernel  


def _geodesic_kernel(Y: np.ndarray, beta: float):

    seg = np.linalg.norm(np.diff(Y, axis=0), axis=1)     # (M-1,)
    s   = np.concatenate([[0.0], np.cumsum(seg)])         # (M,)
    
    D = np.abs(s[:, None] - s[None, :])
    # D   = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)  # (M, M) # Euclidean distance between node positions.
    G   = (1.0 / (4.0 * beta ** 2)) \
          * np.exp(-np.sqrt(2.0) * D / beta) \
          * (2.0 * D + np.sqrt(2.0) * beta)
    return G, s



#  Main tracker


class GeodesicCurveTracker:


    def __init__(self, params: Params):
        self.p            = params
        self.Y:      np.ndarray | None = None   # (M, 2) float64
        self.sigma2: float             = 0.0    # carried across frames (MCT continuity)
        self._V:     np.ndarray | None = None   # (M, 2) per-node velocity
        self._initialized: bool        = False
        self._frame: int               = 0      # frame counter for debug
        self._small_span_frames: int   = 0      # consecutive frames with span < 50px

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def initialize(self, pts: np.ndarray) -> None:

        if pts is None or len(pts) < 4:
            return
        pts = np.asarray(pts, dtype=np.float64)
        # Smooth BFS staircase artefacts (8-connectivity zigzag ≈1px amplitude)
        # before resampling so nodes don't initialise into a zigzag local minimum.
        if len(pts) >= 8:
            pts = gaussian_filter1d(pts, sigma=2.0, axis=0)
        self.Y        = resample_polyline(pts, self.p.num_nodes).astype(np.float64)
        self.sigma2   = 0.0     # trigger fresh σ² init on first register()
        self._V       = None    # reset velocity on cold start
        self._initialized     = True
        self._small_span_frames = 0   # reset stuck-startup counter on any clean init

    def initialize_from_cloud(self, pts_xy: np.ndarray) -> bool:
        
        if pts_xy is None or len(pts_xy) < 10:
            return False
        mu       = pts_xy.mean(axis=0)
        X        = pts_xy - mu
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        t        = X @ Vt[0]
        ordered  = pts_xy[np.argsort(t)]
        self.initialize(ordered)
        return True

  
    #  CPD EM helper
    def _cpd_em(
        self,
        Y0:           np.ndarray,         # (M, 2) starting nodes
        obs:          np.ndarray,         # (N, 2) observations
        G:            np.ndarray,         # (M, M) geodesic kernel
        s:            np.ndarray,         # (M,)   arc-length per node
        lam:          float,
        sigma2_init:  float | None = None,
        alpha_prior:  np.ndarray | None = None,  # (M,) visibility weights
        Y_prior:      np.ndarray | None = None,  # (M, 2) prior positions
        sigma2_schedule: np.ndarray | None = None,
        obs_weights:  np.ndarray | None = None,  # (N,) soft obs weights ∈(0,1]
    ) -> tuple[np.ndarray, float]:

        M, D = Y0.shape
        N    = len(obs)

        # σ² initialisation
        if sigma2_init is None or sigma2_init <= 0.0:
            d_nn_sq = np.linalg.norm(
                Y0[:, None, :] - obs[None, :, :], axis=2
            ).min(axis=1) ** 2                             # (M,) node→nearest obs
            sigma2  = max(float(np.median(d_nn_sq)), 1.0)
        else:
            sigma2 = max(float(sigma2_init), 1e-6)

        Y     = Y0.copy()
        n_idx = np.arange(N)
        j_idx = np.arange(M)[:, None]                      # (M, 1)

        # M-step matrices that don't change across EM iterations
        eye_M = np.eye(M)
        LTL = None
        if self.p.mu_motion > 0.0:
            d_main = np.full(M, 2.0); d_main[0] = d_main[-1] = 1.0
            d_off  = np.full(M - 1, -1.0)
            LTL    = (np.diag(d_main)
                      + np.diag(d_off,  1)
                      + np.diag(d_off, -1)) * self.p.mu_motion

        for _iter in range(self.p.max_iter):
           
            if sigma2_schedule is not None:
                sigma2 = float(sigma2_schedule[min(_iter, len(sigma2_schedule) - 1)])

            # ── Euclidean distances (M, N) ─────────────────────────────────
            diff = Y[:, None, :] - obs[None, :, :]         # (M, N, 2)
            d_eu = np.linalg.norm(diff, axis=2)            # (M, N)

            # ── Two-anchor geodesic distances  ───────────
            m1      = np.argmin(d_eu, axis=0)              # (N,)
            m1_prev = np.clip(m1 - 1, 0, M - 1)
            m1_next = np.clip(m1 + 1, 0, M - 1)
            d_prev  = d_eu[m1_prev, n_idx]
            d_next  = d_eu[m1_next, n_idx]
            # at curve endpoints, force m2 to the only valid neighbour
            m2 = np.where(m1 == 0,     1,
                 np.where(m1 == M - 1, M - 2,
                 np.where(d_prev <= d_next, m1_prev, m1_next)))

            lo   = np.minimum(m1, m2)                      # (N,)
            hi   = np.maximum(m1, m2)                      # (N,)
            d_lo = d_eu[lo, n_idx]                         # (N,)
            d_hi = d_eu[hi, n_idx]                         # (N,)

            # vectorised geodesic distance² (M, N)
            s_j  = s[:, None]                              # (M, 1)
            s_lo = s[lo][None, :]                          # (1, N)
            s_hi = s[hi][None, :]                          # (1, N)
            lo_b = lo[None, :]                             # (1, N)
            hi_b = hi[None, :]                             # (1, N)

            below  = j_idx < lo_b                          # (M, N)
            above  = j_idx > hi_b                          # (M, N)
            geo_sq = np.where(below,
                              (np.abs(s_j - s_lo) + d_lo[None, :]) ** 2,
                     np.where(above,
                              (np.abs(s_j - s_hi) + d_hi[None, :]) ** 2,
                              0.0))                        # (M, N)
            
            #Adding the diameter of the DLO
            #Mean residual error (alignment quality)
            #d_min = d_eu.min(axis=1)        
            #d_bar = np.mean(d_min)          

            #Radius-dependent variance term (r_m from SAM-3)
            #term_radius = alpha * r_m**2 + sigma_min_sq

            #Residual-based upper bound
            #term_residual = d_bar**2 + eps

            #Final per-point variance (diameter-aware)
            #sigma2_m = np.minimum(term_radius, term_residual)

            
            # ── E-step ────────────────────────────────────────────────────
            P = np.exp(-0.5 * geo_sq / sigma2)             # (M, N)  --> with Diameter P = np.exp(-0.5 * geo_sq / sigma2_m[:, None])

            # k_vis node decay (TrackDLO §III-D)
            d_min = d_eu.min(axis=1)                       # (M,)
            vis   = np.exp(-self.p.k_vis * d_min)
            P    *= vis[:, None]

            # soft obs weighting: 
            if obs_weights is not None:
                P *= obs_weights[None, :]                  # broadcast (M, N)

            sigma2_c = float(sigma2_schedule[-1]) if sigma2_schedule is not None else sigma2
            c  = ((2.0 * np.pi * sigma2_c) ** (D / 2.0)) \
                 * (self.p.mu / (1.0 - self.p.mu)) \
                 * (M / N)
            P /= P.sum(axis=0, keepdims=True) + c         # (M, N)

            Pt1 = P.sum(axis=0)                            # (N,)
            P1  = P.sum(axis=1)                            # (M,)
            Np  = max(float(P1.sum()), 1e-10)
            PX  = P @ obs                                  # (M, 2)

            # ── M-step ────────────────────────────────────────────────────
            # P1[:,None] * G == diag(P1) @ G, without the (M,M) matmul
            sigma2_reg = float(sigma2_schedule[-1]) if sigma2_schedule is not None else sigma2
            A = P1[:, None] * G + lam * sigma2_reg * eye_M
            B = PX - P1[:, None] * Y0

            # motion constraint: chain Laplacian L^T L penalises
            # ||W[i+1] - W[i]||² so adjacent nodes deform similarly.
            # G smooths deformation magnitude, L^T L constrains differences.
            if LTL is not None:
                A += LTL

            # correspondence priors (TrackDLO §III-E)
            if alpha_prior is not None and Y_prior is not None:
                A += self.p.alpha * np.diag(alpha_prior)
                B += self.p.alpha * (Y_prior - Y0) * alpha_prior[:, None]

            W     = np.linalg.solve(A, B)                  # (M, 2)
            Y_new = Y0 + G @ W                             # (M, 2)

            # ── σ² update (skipped when annealing schedule controls σ²) ────
            if sigma2_schedule is None:
                trXtdPt1X = float(np.einsum('nd,n,nd', obs,    Pt1, obs))
                trPXtT    = float(np.einsum('md,md',   PX,     Y_new))
                trTtdP1T  = float(np.einsum('md,m,md', Y_new,  P1,  Y_new))
                sigma2    = max(
                    (trXtdPt1X - 2.0 * trPXtT + trTtdP1T) / (Np * D),
                    1e-6,
                )

            # ── Convergence check ─────────────────────────────────────────
            shift = float(np.linalg.norm(Y_new - Y)) / M
            Y     = Y_new
            if shift < self.p.tol:
                break

        return Y, sigma2, _iter + 1

    
    #  CPD registration step

    def register(self, obs: np.ndarray) -> bool | tuple[float, float, int, int]:

        if not self._initialized or self.Y is None:
            return False

        obs = np.asarray(obs, dtype=np.float64)
        if len(obs) < 3:
            return False

        Y_prev = self.Y.copy()
        M      = len(Y_prev)

        # ── Velocity warm-start  ────────────────
       
        if self._V is not None:
            G_ws, _  = _geodesic_kernel(Y_prev, self.p.beta)
            row_sum  = G_ws.sum(axis=1, keepdims=True)
            G_norm   = G_ws / np.where(row_sum > 1e-10, row_sum, 1.0)
            Y0       = Y_prev + G_norm @ self._V
        else:
            Y0 = Y_prev.copy()

        # ── σ² carry-forward  ───────────────────
       
        if self.sigma2 > 0.0:
            sigma2_init = min(max(self.sigma2, 1.0), 100.0)
        else:
            # Estimate target spacing from initial node configuration
            seg_lens    = np.linalg.norm(np.diff(Y0, axis=0), axis=1)
            arc_total   = float(seg_lens.sum())
            spacing     = arc_total / max(M - 1, 1)
            sigma2_init = max((spacing * 0.5) ** 2, 4.0)

        # ── Soft obs weighting: pre-proc uses a loose threshold ─────────     
        
        prune_pre  = self.p.prune_threshold * 3.0
        d_obs      = np.linalg.norm(
            obs[:, None, :] - Y0[None, :, :], axis=2
        ).min(axis=1)                                      # (N,)
        obs_w_pre  = np.exp(-d_obs ** 2 / (2.0 * prune_pre ** 2))
        obs_full   = obs.copy()
        # Drop points with effectively zero weight (uses loose threshold)
        keep       = obs_w_pre > 0.01
        if keep.sum() < 3:
            return False
        obs        = obs[keep]
        obs_w_pre  = obs_w_pre[keep]

        # ── Pre-processing CPD step (rough alignment from predicted pos) ──
        G_pre, s_pre = _geodesic_kernel(Y0, self.p.beta_pre_proc)
        Y_pre, _, _  = self._cpd_em(
            Y0, obs, G_pre, s_pre,
            lam         = self.p.lam_pre_proc,
            sigma2_init = sigma2_init,
            obs_weights = obs_w_pre,
        )

        # ── Tight obs weights for main step (distance from Y_pre) ─────────
        d_obs_main  = np.linalg.norm(
            obs[:, None, :] - Y_pre[None, :, :], axis=2
        ).min(axis=1)
        obs_w_main  = np.exp(-d_obs_main ** 2
                             / (2.0 * self.p.prune_threshold ** 2))

        # Visibility weights from pre-processed positions ─────────────────
        d_min_pre   = np.linalg.norm(
            Y_pre[:, None, :] - obs[None, :, :], axis=2
        ).min(axis=1)
        vis_weights = np.exp(-self.p.k_vis * d_min_pre)

        # ── Main CPD step ─────────────────────────────────────────────────
        G_main, s_main = _geodesic_kernel(Y0, self.p.beta)
        Y_cpd, sigma2_new, n_iters = self._cpd_em(
            Y0, obs, G_main, s_main,
            lam         = self.p.lam,
            sigma2_init = sigma2_init,
            alpha_prior = vis_weights,
            Y_prior     = Y_pre,
            obs_weights = obs_w_main,
        )

        Y_new = Y0 + self.p.correction_scale * (Y_cpd - Y0)

        # ── Arc-length resampling (TrackDLO §III-B: length preservation) ──
        Y_new = resample_polyline(Y_new, M).astype(np.float64)

        # ── Endpoint extension ────────────────────────────────────────────
       
        anchor = min(2, M - 1)                         # use 3rd node as inner anchor
        u_s_vec = Y_new[0] - Y_new[anchor]             # outward tangent at start
        u_e_vec = Y_new[-1] - Y_new[M - 1 - anchor]   # outward tangent at end
        u_s_len = float(np.linalg.norm(u_s_vec))
        u_e_len = float(np.linalg.norm(u_e_vec))
        if u_s_len > 1e-6 and u_e_len > 1e-6:
            u_s = u_s_vec / u_s_len                    # unit outward at start
            u_e = u_e_vec / u_e_len                    # unit outward at end
            seg_len   = float(np.linalg.norm(np.diff(Y_new, axis=0), axis=1).mean())
            max_ext   = max(seg_len * 3.0, 30.0)
            width_thr = seg_len * 2.0     # transverse tolerance

            u_s_perp  = np.array([-u_s[1],  u_s[0]])   # perpendicular at start
            u_e_perp  = np.array([-u_e[1],  u_e[0]])   # perpendicular at end

            t_s       = (obs_full - Y_new[0])  @ u_s
            s_s       = (obs_full - Y_new[0])  @ u_s_perp
            on_axis_s = (t_s > 2.0) & (np.abs(s_s) < width_thr)
            gap_start = float(np.clip(
                t_s[on_axis_s].max() if on_axis_s.any() else 0.0,
                0.0, max_ext))
            if gap_start > 2.0:
                Y_new[0]  = Y_new[0]  + u_s * gap_start

            t_e       = (obs_full - Y_new[-1]) @ u_e
            s_e       = (obs_full - Y_new[-1]) @ u_e_perp
            on_axis_e = (t_e > 2.0) & (np.abs(s_e) < width_thr)
            gap_end   = float(np.clip(
                t_e[on_axis_e].max() if on_axis_e.any() else 0.0,
                0.0, max_ext))
            if gap_end > 2.0:
                Y_new[-1] = Y_new[-1] + u_e * gap_end
            # resample again so spacing stays uniform after extension
            Y_new = resample_polyline(Y_new, M).astype(np.float64)

        # ── Update per-node velocity (exponential moving average) ─────────
        V_new = Y_new - Y_prev
        if self._V is None:
            self._V = V_new.copy()
        else:
            alpha_v = 0.85   # higher → faster response to changing wire speed
            self._V = alpha_v * V_new + (1.0 - alpha_v) * self._V

        # σ² floor: sub-pixel σ² 
        node_span = float(np.linalg.norm(Y_new.max(axis=0) - Y_new.min(axis=0)))
        if node_span < 10.0:
            return False

        # Clip nodes to obs bounding box
        obs_x_min = float(obs_full[:, 0].min()); obs_x_max = float(obs_full[:, 0].max())
        obs_y_min = float(obs_full[:, 1].min()); obs_y_max = float(obs_full[:, 1].max())
        Y_new[:, 0] = np.clip(Y_new[:, 0], obs_x_min, obs_x_max)
        Y_new[:, 1] = np.clip(Y_new[:, 1], obs_y_min, obs_y_max)

        # ── Diagnostics (computed before σ² is stored) ────────────────────
        node_shift  = float(np.linalg.norm(Y_new - Y0, axis=1).mean())
        final_drift = float(np.linalg.norm(
            Y_new[:, None, :] - obs[None, :, :], axis=2
        ).min(axis=1).mean())

        # ── σ² management: fit-tied cap ───────────────────────────────────
       
        sigma2_new = min(sigma2_new, final_drift ** 2 + 20.0)

        self.Y      = Y_new
        self.sigma2 = float(np.clip(sigma2_new, 4.0, 500.0))
        return node_shift, final_drift, n_iters, len(obs)

   
    #  High-level track() entry point
  

    def track(
        self,
        obs:              np.ndarray,
        reinit_threshold: float              = 40.0,   # px, reinit when avg node-to-obs drift exceeds this
        init_path:        np.ndarray | None  = None,   # ordered centerline/skeleton pts for (re)init
    ) -> bool:

        self._frame += 1
        fr = self._frame

        if obs is None or len(obs) < 3:
            return False

        # ── Stuck-startup recovery ────────────────────────────────────────
        if self._initialized and self.Y is not None:
            _cur_span = float(np.linalg.norm(
                self.Y.max(axis=0) - self.Y.min(axis=0)))
            if _cur_span < 50.0:
                self._small_span_frames += 1
            else:
                self._small_span_frames = 0

            if self._small_span_frames >= 3:
                self._small_span_frames = 0
                obs_f = np.asarray(obs, dtype=np.float64)
                self.initialize_from_cloud(obs_f)
                _t0 = time.perf_counter()
                result = self.register(obs_f)
                _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
                if not result and self.Y is not None:
                    seg = np.linalg.norm(np.diff(self.Y, axis=0), axis=1)
                    spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                    self.sigma2 = max((spacing * 0.4) ** 2, 16.0)
                new_span = float(np.linalg.norm(
                    self.Y.max(axis=0) - self.Y.min(axis=0))) if self.Y is not None else 0.0
                print(f"[fr={fr:04d}] UNSTUCK  span was <50px for 3 frames → PCA reinit"
                      f"  span={new_span:.0f}px  σ²={self.sigma2:.1f}  t={_elapsed_ms:.1f}ms")
                return self.Y is not None

        def _do_init():
            if init_path is not None and len(init_path) >= 4:
                path_span = float(np.linalg.norm(
                    init_path.max(axis=0) - init_path.min(axis=0)
                ))
                obs_span = float(np.linalg.norm(
                    obs.max(axis=0) - obs.min(axis=0)
                ))
                if path_span >= 0.5 * obs_span:
                    self.initialize(init_path)
                    return
            self.initialize_from_cloud(obs)

        def _span_str():
            if self.Y is None:
                return "span=---"
            sp = float(np.linalg.norm(self.Y.max(axis=0) - self.Y.min(axis=0)))
            flag = "  *** nodes bunched — not on wire ***" if sp < 50 else (
                   "  (partial coverage)"                  if sp < 150 else "")
            return f"span={sp:.0f}px{flag}"

        if not self._initialized:
            _do_init()
            _t0 = time.perf_counter()
            result = self.register(obs)
            _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            if not result:
                if self.Y is not None:
                    seg = np.linalg.norm(np.diff(self.Y, axis=0), axis=1)
                    spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                    self.sigma2 = max((spacing * 0.4) ** 2, 16.0)
                    print(f"[fr={fr:04d}] INIT     {_span_str()}  σ²={self.sigma2:.1f}"
                          f"  t={_elapsed_ms:.1f}ms"
                          f"  (CPD collapsed on first frame — σ² estimated from node spacing)")
                    return True   
                return False     
            if self.sigma2 < 4.0:
                seg = np.linalg.norm(np.diff(self.Y, axis=0), axis=1)
                spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                self.sigma2 = max((spacing * 0.4) ** 2, 16.0)
            _, init_drift, _iters, _pts = result
            print(f"[fr={fr:04d}] INIT     {_span_str()}  fit={init_drift:.0f}px"
                  f"  pts={_pts}  iters={_iters}  σ²={self.sigma2:.1f}  t={_elapsed_ms:.1f}ms")
            return True

        _t0 = time.perf_counter()
        result = self.register(obs)
        _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        if not result:
            _do_init()
            _t0 = time.perf_counter()
            self.register(obs)
            _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            if self.sigma2 > 400.0 or self.sigma2 < 4.0:
                if self.Y is not None:
                    seg = np.linalg.norm(np.diff(self.Y, axis=0), axis=1)
                    spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                    self.sigma2 = max((spacing * 0.4) ** 2, 16.0)
            print(f"[fr={fr:04d}] SNAP     {_span_str()}  σ²={self.sigma2:.1f}"
                  f"  t={_elapsed_ms:.1f}ms"
                  f"  (all obs were too far from nodes — snapped back to wire)")
            return self.Y is not None

        node_shift, drift, n_iters, n_pts = result
        vel_mag = float(np.linalg.norm(self._V, axis=1).mean()) if self._V is not None else 0.0

        s2 = self.sigma2

        # Stuck detector
        stuck = (s2 >= 475.0 and node_shift < 2.0 and drift > 15.0)

        status   = "TRACK"
        reinit_reason = ""
        if drift > reinit_threshold or stuck:
            reinit_reason = f"stuck (σ²={s2:.0f}, not moving)" if stuck else f"drift={drift:.0f}px exceeded {reinit_threshold:.0f}px threshold"
            status = "REINIT"
            _do_init()
            _t0 = time.perf_counter()
            reinit_result = self.register(obs)
            _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            if reinit_result:
                _, _, n_iters, n_pts = reinit_result
           
            if self.sigma2 > 400.0 or self.sigma2 < 4.0:
                if self.Y is not None:
                    seg = np.linalg.norm(np.diff(self.Y, axis=0), axis=1)
                    spacing = float(seg.mean()) if len(seg) > 0 else 20.0
                    self.sigma2 = max((spacing * 0.4) ** 2, 16.0)
            s2 = self.sigma2

        if status == "TRACK":
            print(f"[fr={fr:04d}] TRACK    {_span_str()}  "
                  f"fit={drift:.0f}px  speed={vel_mag:.0f}px/fr  "
                  f"pts={n_pts}  iters={n_iters}  σ²={s2:.1f}  t={_elapsed_ms:.1f}ms")
        else:
            print(f"[fr={fr:04d}] REINIT   {reinit_reason} → reset  "
                  f"{_span_str()}  pts={n_pts}  iters={n_iters}  σ²={s2:.1f}  t={_elapsed_ms:.1f}ms")

        return self.Y is not None

    
    #  External correction  (hybrid mode)
  

    def correct(self, Y_new: np.ndarray, sigma2_new: float) -> None:
        
        if Y_new is not None \
                and self.Y is not None \
                and len(Y_new) == len(self.Y):
            self.Y = np.asarray(Y_new, dtype=np.float64)
        if sigma2_new > 0.0:
            self.sigma2 = float(sigma2_new)
