import sys

import numpy as np
import cv2
from skimage.measure import label as sk_label, regionprops
import EventBased.Filter as ev_filter
from EventBased.sae import sae
from EventBased.Visualization import Visualization
from EventBased.event_cpd import GeodesicCurveTracker, Params, mask_to_skeleton_obs, mask_to_pointcloud
from metavision_core.event_io import EventsIterator
import config





def _keep_elongated(
    wire_bool:  np.ndarray,
    min_aspect: float = 3.5,
    min_length: float = 80.0,
) -> np.ndarray:
    """Keep all connected components that are elongated and long enough to be cable."""
    labeled = sk_label(wire_bool)
    out     = np.zeros_like(wire_bool)
    for p in regionprops(labeled):
        if p.minor_axis_length > 0:
            aspect = p.major_axis_length / p.minor_axis_length
            if aspect >= min_aspect and p.major_axis_length >= min_length:
                out[labeled == p.label] = True
    return out


def adaptive_kernels(wire_diameter_px: float):
    """Derive morphological kernel sizes from wire diameter."""
    # Closing: bridge gaps up to ~half the wire width
    close_size = int(round(wire_diameter_px * 0.5)) | 1  # ensure odd
    close_size = max(close_size, 5)

    # Opening: remove noise smaller than ~15% of wire width
    open_size = int(round(wire_diameter_px * 0.15)) | 1
    open_size = max(open_size, 3)

    # Iterations: thicker wires need more passes
    close_iters = max(1, int(round(wire_diameter_px / 25)))

    # Min area: wire is elongated, expect at least 10× diameter in length
    min_area = int(wire_diameter_px * wire_diameter_px * 3)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))

    return close_k, close_iters, open_k, min_area

# ----------------------------- minimal pipeline -----------------------------

def main():
    args = config.parse_args()
    
    np.random.seed(42)
    it = EventsIterator(
        input_path="/path/to/Dataset/Single_Cable/Blue_Cable/Speed_100/1/events.raw",
        delta_t=args.delta_t_us,
        mode="delta_t",
    )
    h, w = it.get_size()

    # Polarity-separated time surfaces
    T_on  = np.full((h, w), -10**18, dtype=np.int64)
    T_off = np.full((h, w), -10**18, dtype=np.int64)

    trail, trail_out, afk, afk_out, stc_filter, stc_out = ev_filter.filter_events(it, w, h, args)

    # FIX3: velocity warm-start state
    _prev_centroid = None
    _prev_t        = None
    _velocity      = np.zeros(2)   # (vx, vy) in px/µs
 
    _speed_records: list[float] = []   # vel_mag in px/fr for every valid TRACK frame
    
    tracker = GeodesicCurveTracker(Params(
        num_nodes        = args.spline_samples,
        beta             = 20.0,
        lam              = 0.0001,
        mu               = 0.1,
        k_vis            = 0.05,
        max_iter         = 30,
        tol              = 1e-4,
        beta_pre_proc    = 1.0,
        lam_pre_proc     = 0.0001,
        prune_threshold  = 300.0,
    ))
    print(f"[CONFIG] num_nodes = {tracker.p.num_nodes}")
    if args.show:
        cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

    import os
    os.makedirs("recording", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_debug = cv2.VideoWriter("recording/debug.mp4", fourcc, 30.0, (w, h))
    writer_cpd   = cv2.VideoWriter("recording/cpd.mp4",   fourcc, 30.0, (w, h))

    for evs in it:
        if len(evs) == 0:
            continue

        # --- filter ---
        ev = evs
        # if trail is not None:
        #     trail.process_events(ev, trail_out)
        #     ev = trail_out.numpy()
        # if afk is not None:
        #     afk.process_events(ev, afk_out)
        #     ev = afk_out.numpy()
        if stc_filter is not None:
            stc_filter.process_events(ev, stc_out)
            ev = stc_out.numpy()
        if len(ev) == 0:
            continue

        xs = ev["x"].astype(np.int32, copy=False)
        ys = ev["y"].astype(np.int32, copy=False)
        ts = ev["t"].astype(np.int64, copy=False)
        ps = ev["p"]
        t_now = int(ts[-1])

        on = ps if ps.dtype == np.bool_ else (ps > 0)

        # --- update SAE ---
        if np.any(on):
            T_on[ys[on], xs[on]] = ts[on]
        if np.any(~on):
            T_off[ys[~on], xs[~on]] = ts[~on]

        dt_batch_us = int(ts[-1] - ts[0])
        if dt_batch_us <= 0:
            continue
        event_rate  = len(ev) / (dt_batch_us * 1e-6)
        tau_us_dyn  = ev_filter.adaptive_tau_us(event_rate, tau_min=800, tau_max=10000)  # FIX1: shorter tau → R shows current position not ghost trail

        # --- recentness map ---
        R = sae.compute_recentness(T_on=T_on, T_off=T_off, t_now_us=t_now, tau_us=tau_us_dyn)

        # --- motion discrimination---
       
        R_gated = R

        # --- wire mask ---
        active = R_gated[R_gated > 0]
        if len(active) == 0:
            continue
        
        wire_d = 50  # from your measurement
        close_k, close_iters, open_k, min_area = adaptive_kernels(wire_d)



        R_u8 = np.clip(R_gated * 255, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(R_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        wire_thr = max(float(otsu_val) / 255.0, 0.05)  # Otsu finds natural gap between noise and cable R values
        wire_mask = (R_gated >= wire_thr).astype(np.uint8)
      
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, close_k, iterations=close_iters)  # reduced: less blob growth from closing
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN,  open_k, iterations=1)
        wire_mask = wire_mask.astype(bool)

        wire_mask = _keep_elongated(wire_mask, min_aspect=3.5)

        mask_count = int(wire_mask.sum())
        if mask_count < 30 or mask_count > wire_mask.size * 0.3:
            continue

        # --- observations and init path ---

        _, init_path = mask_to_skeleton_obs(wire_mask, max_pts=300)
        obs = mask_to_pointcloud(wire_mask, max_pts=500)
        if obs is None:
            continue

        # Trust-region obs filtering: centered on skeleton 
       
        if init_path is not None and len(obs) > 0:
            dists = np.min(np.linalg.norm(obs[:, None, :] - init_path[None, :, :], axis=2), axis=1)
            obs_gated = obs[dists < 30.0]
            if len(obs_gated) >= 10:
                obs = obs_gated

        #velocity warm-start — shift init_path by predicted displacement
        if tracker.Y is not None and _prev_centroid is not None and _prev_t is not None:
            dt = t_now - _prev_t
            if dt > 0:
                cur_centroid = tracker.Y.mean(axis=0)
                _velocity    = (cur_centroid - _prev_centroid) / dt
        predicted_shift = _velocity * (t_now - _prev_t) if _prev_t is not None else np.zeros(2)
        if init_path is not None and np.linalg.norm(predicted_shift) > 0:
            init_path = init_path + predicted_shift[None, :]

        tracker.track(obs, init_path=init_path)

        #update velocity state after track
        if tracker.Y is not None:
            _prev_centroid = tracker.Y.mean(axis=0)
            _prev_t        = t_now
        if tracker._V is not None:
                        vel_mag = float(np.linalg.norm(tracker._V, axis=1).mean())
                        # exclude bunched frames: span collapses to near-zero
                        span = float(np.linalg.norm(tracker.Y[-1] - tracker.Y[0]))
                        if span > 50.0:   # px — same threshold as bunched detector
                            _speed_records.append(vel_mag)

        # --- visualise ---
        if args.show:
            Visualization.visualize_wire(R, mask=wire_mask, skel=None, win="debug", scale=1.0)

            frame = cv2.cvtColor(
                np.clip(R * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            if obs is not None:
                for pt in obs.astype(np.int32):
                    cv2.circle(frame, (pt[0], pt[1]), 1, (0, 200, 0), -1)
            if init_path is not None and len(init_path) >= 2:
                cv2.polylines(
                    frame,
                    [init_path.astype(np.int32).reshape(-1, 1, 2)],
                    False, (255, 100, 0), 1, cv2.LINE_AA,
                )
            if tracker.Y is not None:
                polyline_cv = tracker.Y.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [polyline_cv], False, (0, 255, 255), 2, cv2.LINE_AA)
                for pt in tracker.Y.astype(np.int32):
                    cv2.circle(frame, (pt[0], pt[1]), 3, (0, 128, 255), -1)
            cv2.imshow("cpd", frame)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
# ── speed summary ────────────────────────────────────────────────────────
    if _speed_records:
        import statistics
        delta_t_s   = args.delta_t_us * 1e-6
        speeds_px_s = [v / delta_t_s for v in _speed_records]
        stable      = speeds_px_s[10:]          # skip first 10 frames (EMA warmup)
        if len(stable) < 2:
            print("\n[speed] Not enough stable frames to compute summary.")
        else:
            median_all  = statistics.median(stable)
            median_last = statistics.median(stable[-50:])
            print(f"\n{'─'*50}")
            print(f"  Speed summary ({len(stable)} frames, first 10 skipped)")
            print(f"  Median (all)    : {median_all:7.1f} px/s")
            print(f"  Median (last 50): {median_last:7.1f} px/s")
            print(f"{'─'*50}")
    else:
        print("\n[speed] No valid frames recorded.")


if __name__ == "__main__":
    main()
