"""
Simple Per-frame polyline fitting from binary cable masks.

Usage
-----
    polylines = FramePolyline.extract(mask)
    # polylines: list of (N, 2) float32 arrays, one per detected cable
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize, binary_closing, disk


# ── Defaults ───────────────────────────────────────────────────────────────────

NUM_NODES          = 60
SG_WINDOW          = 11
SG_POLY            = 3
MIN_CABLE_LENGTH   = 150    # px — discard shorter cables
MIN_COMPONENT_AREA = 600    # px² — discard tiny noise blobs
MAX_GAP_PX         = 200    # px — max endpoint gap for segment stitching
CLOSING_LARGE      = 8
CLOSING_SMALL      = 3


class FramePolyline:
    """
    Static-method toolkit for fitting polylines to cable masks.

    All methods are stateless; instantiation is not required.
    The main entry point is ``FramePolyline.extract(mask)``.
    """

    # ── Main entry point ───────────────────────────────────────────────────────

    def extract(
        mask:        np.ndarray,
        num_nodes:   int  = NUM_NODES,
        min_length:  int  = MIN_CABLE_LENGTH,
        return_skel: bool = False,
    ):
        """
        Fit polylines to a binary mask.
        """
        mb   = FramePolyline.binarize(mask)
        H, W = mb.shape

        closed  = FramePolyline.adaptive_closing(mb)
        binary  = FramePolyline.binarize(closed)

        nc, labels = cv2.connectedComponents(binary, connectivity=8)

        all_segs    = []
        all_skel_px = np.zeros((0, 2), np.float32)

        for lid in range(1, nc):
            comp = (labels == lid).astype(np.uint8)
            if comp.sum() < MIN_COMPONENT_AREA:
                continue
            skel = skeletonize(comp > 0).astype(np.uint8)
            if skel.sum() < 15:
                continue
            sy, sx = np.where(skel > 0)
            sp = np.column_stack([sx, sy]).astype(np.float32)
            all_skel_px = np.vstack([all_skel_px, sp]) if len(all_skel_px) else sp
            segs = [s for s in FramePolyline.extract_skeleton_segments(skel)
                    if len(s) >= 10]
            all_segs.extend(segs)

        if not all_segs:
            return ([], []) if return_skel else []

        conns  = FramePolyline.match_segments(all_segs, H, W)
        chains = FramePolyline.connect_segments(all_segs, conns)

        polylines = []
        skel_list = []
        for chain in chains:
            chain = FramePolyline.savgol_smooth(chain)
            Y     = FramePolyline.resample_nodes(chain, M=num_nodes)
            if Y is None:
                continue
            if FramePolyline.arc_length(Y) >= min_length:
                polylines.append(Y)
                skel_list.append(all_skel_px)

        return (polylines, skel_list) if return_skel else polylines

    # ── Low-level utilities ────────────────────────────────────────────────────

    def binarize(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (img > 127).astype(np.uint8)

    def arc_length(Y: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(Y, axis=0), axis=1)))

    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    # ── Adaptive morphological closing ────────────────────────────────────────

    def adaptive_closing(mask_bin: np.ndarray) -> np.ndarray:

        n, labels = cv2.connectedComponents(mask_bin.astype(np.uint8), connectivity=8)
        if n <= 2:
            disk_size = CLOSING_LARGE
        else:
            too_close = False
            lids = list(range(1, n))
            for i in range(len(lids)):
                for j in range(i + 1, len(lids)):
                    if FramePolyline._min_cc_dist(labels, lids[i], lids[j]) < 2 * CLOSING_LARGE:
                        too_close = True
                        break
                if too_close:
                    break
            disk_size = CLOSING_SMALL if too_close else CLOSING_LARGE
        return (binary_closing(mask_bin.astype(bool), footprint=disk(disk_size)).astype(np.uint8) * 255)

    def _min_cc_dist(labels: np.ndarray, lid1: int, lid2: int) -> float:
        y1, x1 = np.where(labels == lid1)
        y2, x2 = np.where(labels == lid2)
        if len(y1) == 0 or len(y2) == 0:
            return np.inf
        s1 = np.random.choice(len(y1), min(len(y1), 500), replace=False)
        s2 = np.random.choice(len(y2), min(len(y2), 500), replace=False)
        p1   = np.column_stack([x1[s1], y1[s1]]).astype(np.float32)
        p2   = np.column_stack([x2[s2], y2[s2]]).astype(np.float32)
        diff = p1[:, None, :] - p2[None, :, :]
        return float(np.sqrt((diff**2).sum(axis=-1)).min())

    # ── Skeleton graph + segment extraction ───────────────────────────────────

    def extract_skeleton_segments(skel: np.ndarray, min_len: int = 10) -> List[np.ndarray]:
        """
        Walk the 1-px skeleton graph and return a list of ordered point arrays,
        one per branch.  Junctions are resolved by collinearity so that crossing
        cables are not merged.
        """
        pts, adj, deg = FramePolyline._build_skel_graph(skel)
        if pts is None:
            return []

        critical = (deg == 1) | (deg >= 3)
        visited  = set()
        segs     = []

        # Pre-compute crossing-junction pairings
        cp = {}
        for junc in np.where(deg >= 4)[0]:
            brs = [{'nb': nb, 'tang': FramePolyline._branch_tangent(pts, junc, nb, adj)}
                   for nb in adj[junc]]
            if len(brs) == 4:
                best_p, best_s = None, -1
                for pairing, score in FramePolyline._pair_collinear(brs):
                    if score > best_s:
                        best_s, best_p = score, pairing
                cp[junc] = [(brs[a]['nb'], brs[b]['nb']) for a, b in best_p]
            elif len(brs) >= 3:
                best, bs = (0, 1), -1
                for ii in range(len(brs)):
                    for jj in range(ii + 1, len(brs)):
                        s = abs(np.dot(brs[ii]['tang'], brs[jj]['tang']))
                        if s > bs:
                            bs, best = s, (ii, jj)
                cp[junc] = [(brs[best[0]]['nb'], brs[best[1]]['nb'])]

        def allowed_next(prev, curr):
            if curr not in cp:
                return [nb for nb in adj[curr] if nb != prev]
            for a, b in cp[curr]:
                if a == prev: return [b]
                if b == prev: return [a]
            return []

        def walk(start, nb0):
            seg = [start]
            prev, curr = start, nb0
            while True:
                seg.append(curr)
                visited.add(tuple(sorted([prev, curr])))
                if curr != start and deg[curr] >= 3:
                    nxt = allowed_next(prev, curr)
                    if len(nxt) == 1:
                        ek = tuple(sorted([curr, nxt[0]]))
                        if ek in visited: break
                        prev, curr = curr, nxt[0]
                    else:
                        break
                else:
                    nbs = [nb for nb in adj[curr] if nb != prev]
                    if len(nbs) != 1: break
                    ek = tuple(sorted([curr, nbs[0]]))
                    if ek in visited: break
                    prev, curr = curr, nbs[0]
            return seg

        for start in np.where(deg == 1)[0]:
            for nb in adj[start]:
                ek = tuple(sorted([start, nb]))
                if ek in visited: continue
                seg = walk(start, nb)
                if len(seg) >= min_len:
                    segs.append(pts[seg].astype(np.float32))

        for start in np.where(critical)[0]:
            for nb in adj[start]:
                ek = tuple(sorted([start, nb]))
                if ek in visited: continue
                seg = walk(start, nb)
                if len(seg) >= min_len:
                    segs.append(pts[seg].astype(np.float32))

        return segs

    def _build_skel_graph(skel: np.ndarray):
        y, x = np.where(skel > 0)
        pts  = np.column_stack([x, y])
        if len(pts) < 2:
            return None, [], np.array([])
        idx = {(xi, yi): i for i, (xi, yi) in enumerate(pts)}
        adj = [[] for _ in range(len(pts))]
        for i, (xi, yi) in enumerate(pts):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0: continue
                    nb = (xi + dx, yi + dy)
                    if nb in idx and idx[nb] not in adj[i]:
                        adj[i].append(idx[nb])
        return pts, adj, np.array([len(a) for a in adj])

    def _branch_tangent(pts, start, nxt, adj, n=8):
        path = [start]
        prev, curr = start, nxt
        for _ in range(n):
            path.append(curr)
            nbs = [nb for nb in adj[curr] if nb != prev]
            if len(nbs) != 1: break
            prev, curr = curr, nbs[0]
        if len(path) < 2:
            return np.array([1.0, 0.0])
        v = pts[path[-1]] - pts[path[0]]
        return v / (np.linalg.norm(v) + 1e-9)

    def _pair_collinear(branches):
        assert len(branches) == 4
        tang = [b['tang'] for b in branches]
        for pairing in [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]:
            yield pairing, sum(abs(np.dot(tang[a], tang[b])) for a, b in pairing)

    # ── Segment gap-stitching ─────────────────────────────────────────────────

    def match_segments(segs: List[np.ndarray], H: int, W: int) -> List[dict]:
        """Score all endpoint pairs; return sorted connection candidates."""
        if len(segs) <= 1:
            return []
        feats = [FramePolyline._seg_feats(s) for s in segs]
        conns = []
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                fi, fj = feats[i], feats[j]
                if fi is None or fj is None: continue
                for ei, di, ej, dj, pi, pj in [
                    ('s', -fi['sd'], 's', -fj['sd'], fi['sp'], fj['sp']),
                    ('s', -fi['sd'], 'e',  fj['ed'], fi['sp'], fj['ep']),
                    ('e',  fi['ed'], 's', -fj['sd'], fi['ep'], fj['sp']),
                    ('e',  fi['ed'], 'e',  fj['ed'], fi['ep'], fj['ep']),
                ]:
                    gap = float(np.linalg.norm(pi - pj))
                    if gap > MAX_GAP_PX: continue
                    aln = -float(np.dot(di, dj))
                    if aln < -0.5: continue
                    score = aln * (1.0 - (gap / MAX_GAP_PX) ** 0.5)
                    conns.append(dict(i=i, ei=ei, j=j, ej=ej, gap=gap, score=score))
        conns.sort(key=lambda c: c['score'], reverse=True)
        return conns

    def connect_segments(segs: List[np.ndarray], conns: List[dict]) -> List[np.ndarray]:
        """Union-find merge of segments into chains, bridged by Bézier curves."""
        n   = len(segs)
        use = {i: {'s': None, 'e': None} for i in range(n)}
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x

        for c in conns:
            i, ei, j, ej = c['i'], c['ei'], c['j'], c['ej']
            if use[i][ei] or use[j][ej]: continue
            use[i][ei] = (j, ej, c['gap'])
            use[j][ej] = (i, ei, c['gap'])
            par[find(i)] = find(j)

        groups: Dict[int, list] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        merged = []
        for _, grp in groups.items():
            if len(grp) == 1:
                merged.append(segs[grp[0]]); continue
            ss, se = grp[0], 's'
            for si in grp:
                if use[si]['s'] is None: ss, se = si, 's'; break
                if use[si]['e'] is None: ss, se = si, 'e'; break
            parts, vis_set = [], set()
            cur, cend = ss, se
            while cur is not None and cur not in vis_set:
                vis_set.add(cur)
                seg = segs[cur]
                parts.append(seg if cend == 's' else seg[::-1])
                nend = 'e' if cend == 's' else 's'
                conn = use[cur][nend]
                if conn is None or conn[0] in vis_set: break
                nxt, nside, gap = conn
                ns     = segs[nxt]
                ns_sub = ns[:min(20, len(ns))] if nside == 's' else ns[-min(20, len(ns)):][::-1]
                bridge = FramePolyline._bezier_bridge(
                    parts[-1][-min(20, len(parts[-1])):],
                    parts[-1][-1], ns_sub[0], ns_sub,
                    n=max(20, int(gap / 1.5)))
                parts.append(bridge[1:])
                cur, cend = nxt, nside
            if parts:
                merged.append(np.vstack(parts).astype(np.float32))
        return merged

    def _seg_feats(seg, n=15):
        if len(seg) < 2: return None
        n = max(2, min(n, len(seg) // 2))
        return dict(
            sp=seg[0], ep=seg[-1],
            sd=FramePolyline._unit(seg[min(n, len(seg) - 1)] - seg[0]),
            ed=FramePolyline._unit(seg[-1] - seg[max(0, len(seg) - 1 - n)]),
        )

    def _bezier_bridge(s1, p1, p2, s2, n=20):
        def tang(s, front):
            k = min(5, len(s) - 1)
            v = (s[k] - s[0]) if front else (s[-1] - s[-1 - k])
            return v / (np.linalg.norm(v) + 1e-9)
        t1, t2 = tang(s1, False), tang(s2, True)
        d  = np.linalg.norm(p2 - p1) * 0.4
        p0, p1c, p2c, p3 = p1, p1 + t1 * d, p2 - t2 * d, p2
        t  = np.linspace(0, 1, n)
        c  = (((1 - t) ** 3)[:, None] * p0 + (3 * (1 - t) ** 2 * t)[:, None] * p1c
              + (3 * (1 - t) * t ** 2)[:, None] * p2c + (t ** 3)[:, None] * p3)
        return c.astype(np.float32)

    # ── Smoothing + resampling ─────────────────────────────────────────────────

    def savgol_smooth(pts: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
        """Savitzky-Golay smoothing along x and y independently."""
        N = len(pts)
        w = min(window, N) if N % 2 == 1 else min(window, N - 1 if N > 1 else 1)
        if w % 2 == 0: w += 1
        if w < poly + 2 or w > N: return pts.copy()
        x = savgol_filter(pts[:, 0], window_length=w, polyorder=poly)
        y = savgol_filter(pts[:, 1], window_length=w, polyorder=poly)
        return np.column_stack([x, y]).astype(np.float32)

    def resample_nodes(pts: np.ndarray, M: int = NUM_NODES) -> Optional[np.ndarray]:
        """Resample a polyline to M equidistant nodes via spline interpolation."""
        if pts is None or len(pts) < 2: return None
        mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
        pts  = pts[mask]
        if len(pts) < 2: return None
        if len(pts) < 4:
            idx = np.round(np.linspace(0, len(pts) - 1, M)).astype(int)
            return pts[idx].astype(np.float32)
        d     = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        chord = np.concatenate([[0.], np.cumsum(d)])
        chord /= chord[-1] + 1e-12
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], u=chord, s=0, k=min(3, len(pts) - 1))
            xu, yu = splev(np.linspace(0, 1, M), tck)
            return np.column_stack([xu, yu]).astype(np.float32)
        except Exception:
            idx = np.round(np.linspace(0, len(pts) - 1, M)).astype(int)
            return pts[idx].astype(np.float32)