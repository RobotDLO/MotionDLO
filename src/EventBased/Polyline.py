import numpy as np
import cv2
from scipy.ndimage import label
from skimage.morphology import skeletonize


class Polyline:

    def _pca_axis(points_xy: np.ndarray):
        """
        points_xy: (N,2) array with columns [x,y]
        returns: unit direction vector v (2,), unit normal n (2,), mean (2,)
        """
        mu = points_xy.mean(axis=0)
        X = points_xy - mu
        # covariance 2x2
        C = (X.T @ X) / max(len(points_xy) - 1, 1)
        # eigenvectors sorted by largest eigenvalue
        evals, evecs = np.linalg.eigh(C)
        v = evecs[:, np.argmax(evals)]
        v = v / (np.linalg.norm(v) + 1e-12)
        n = np.array([-v[1], v[0]], dtype=np.float32)  # perpendicular
        return v.astype(np.float32), n.astype(np.float32), mu.astype(np.float32)


    def split_two_edges(points_xy: np.ndarray, min_points_per_edge: int = 50):
        """
        Split edge pixels into two sets corresponding to the two wire edges.
        Uses PCA: split by the sign of projection onto the normal direction.
        Returns (edgeA_points, edgeB_points) each (Na,2)/(Nb,2) or (None, None).
        """
        if points_xy is None or len(points_xy) < 2 * min_points_per_edge:
            return None, None

        v, n, mu = Polyline._pca_axis(points_xy)
        proj_n = (points_xy - mu) @ n  # scalar per point

        # robust split: use median as separating hyperplane
        m = np.median(proj_n)
        A = points_xy[proj_n <= m]
        B = points_xy[proj_n > m]

        if len(A) < min_points_per_edge or len(B) < min_points_per_edge:
            return None, None

        return A, B


    def order_and_resample_polyline(points_xy: np.ndarray, n_samples: int = 60, smooth_win: int = 7):
        """
        Order points along the wire (PCA major axis) and resample to n_samples vertices.
        Returns polyline int32 array shape (n_samples,1,2) suitable for cv2.polylines, or None.
        """
        if points_xy is None or len(points_xy) < n_samples:
            return None

        v, n, mu = Polyline._pca_axis(points_xy)
        t = (points_xy - mu) @ v  # coordinate along the wire
        idx = np.argsort(t)
        pts_sorted = points_xy[idx].astype(np.float32)

        # Resample by index (works well when edge pixels are dense)
        N = len(pts_sorted)
        sample_idx = np.linspace(0, N - 1, n_samples).astype(np.int32)
        poly = pts_sorted[sample_idx]

        # Optional smoothing (moving average in vertex space)
        if smooth_win is not None and smooth_win >= 3:
            k = int(smooth_win)
            if k % 2 == 0:
                k += 1
            pad = k // 2
            poly_pad = np.vstack([poly[0:1].repeat(pad, axis=0), poly, poly[-1:].repeat(pad, axis=0)])
            kernel = np.ones((k, 1), dtype=np.float32) / k
            poly_s = np.zeros_like(poly_pad)
            # separable moving average
            poly_s[:, 0] = np.convolve(poly_pad[:, 0], kernel[:, 0], mode="same")
            poly_s[:, 1] = np.convolve(poly_pad[:, 1], kernel[:, 0], mode="same")
            poly = poly_s[pad:-pad]

        poly_int = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        return poly_int


    def largest_component(mask: np.ndarray) -> np.ndarray:
        lab, n = label(mask)
        if n == 0:
            return mask
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        keep = np.argmax(counts)
        return (lab == keep)


    def prune_skeleton(skel: np.ndarray, min_branch_length: int = 15) -> np.ndarray:
        """
        Remove short branches from a skeleton by iteratively removing endpoint
        pixels (pixels with exactly 1 8-connected skeleton neighbour).
        """
        s = skel.astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        for _ in range(min_branch_length):
            nb = cv2.filter2D(s, -1, k) * s   # neighbour count + self
            endpoints = (nb == 2) & s.astype(bool)  # exactly 1 neighbour
            if not endpoints.any():
                break
            s[endpoints] = 0
        return s.astype(bool)


    def skeleton_centerline(wire_mask: np.ndarray) -> np.ndarray:
        """
        Compute the 1-px skeleton of wire_mask.

        Steps
        -----
        1. Skeletonise the mask.
        2. Dilate the skeleton by 3 px and re-skeletonise — bridges fragments
           caused by thin spots or sharp curves in the mask.
        3. Keep the largest connected skeleton component.

        Parameters
        ----------
        wire_mask : (H,W) bool

        Returns
        -------
        (H,W) bool
        """
        skel = skeletonize(wire_mask).astype(np.uint8)

        # Bridge fragments: dilate → re-skeletonize
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skel = cv2.dilate(skel, k, iterations=1)
        skel = skeletonize(skel.astype(bool)).astype(bool)

        skel = Polyline.largest_component(skel)
        return skel


    def skeleton_to_ordered_path(skel: np.ndarray) -> np.ndarray | None:
        """
        Extract a clean tip-to-tip ordered path from a 1-px skeleton.
        """
        skel = skel.astype(bool)
        H, W = skel.shape

        ys, xs = np.where(skel)
        if len(xs) < 2:
            return None

        # find an endpoint to start from
        k  = np.ones((3, 3), np.uint8)
        nb = cv2.filter2D(skel.astype(np.uint8), -1, k) * skel.astype(np.uint8)
        ep_mask = (nb == 2) & skel
        ep_ys, ep_xs = np.where(ep_mask)

        if len(ep_xs) >= 1:
            sy, sx = int(ep_ys[0]), int(ep_xs[0])
        else:
            # closed loop or junction-only → start from leftmost pixel
            idx    = int(np.argmin(xs))
            sy, sx = int(ys[idx]), int(xs[idx])

        def _bfs_backtrack(start_y: int, start_x: int):
            """
            BFS on skel from (start_y, start_x).
            Returns the backtracked path from start to the farthest pixel,
            as an ordered list of (y, x) tuples, and the far-tip coordinate.
            """
            # Use 2-D arrays for O(1) parent lookups
            par_y = np.full((H, W), -1, dtype=np.int16)
            par_x = np.full((H, W), -1, dtype=np.int16)
            visited = np.zeros((H, W), dtype=bool)

            visited[start_y, start_x] = True
            queue = [(start_y, start_x)]
            last_y, last_x = start_y, start_x

            while queue:
                cy, cx = queue.pop(0)
                last_y, last_x = cy, cx
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W \
                                and skel[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            par_y[ny, nx] = cy
                            par_x[ny, nx] = cx
                            queue.append((ny, nx))

            # backtrack: last → start (gives the main path, no branches)
            path = []
            cy, cx = last_y, last_x
            while cy != -1 and cx != -1:
                path.append((cy, cx))
                py = int(par_y[cy, cx])
                px = int(par_x[cy, cx])
                if py == -1:   # reached the start (parent sentinel)
                    break
                cy, cx = py, px

            path.reverse()   # start → far tip
            return path, (last_y, last_x)

        # two-pass: first to find far tip, second for clean ordered path
        _, far      = _bfs_backtrack(sy, sx)
        path, _     = _bfs_backtrack(far[0], far[1])

        pts = np.array([[x, y] for y, x in path], dtype=np.float32)
        return pts
