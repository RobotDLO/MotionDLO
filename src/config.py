import argparse


# Arguments for Event based settings
def parse_args():
    p = argparse.ArgumentParser()

    # ── Event iterator ────────────────────────────────────────────────────────
    p.add_argument("--delta-t-us", type=int, default=2000)
    p.add_argument("--tau-us", type=int, default=20000)

    # ── Recentness τ (adaptive decay) ─────────────────────────────────────────
    p.add_argument("--tau-min-us", type=int, default=15000,
            help="Lower bound for adaptive recentness τ (high event rate).")
    p.add_argument("--tau-max-us", type=int, default=60000,
            help="Upper bound for adaptive recentness τ (low event rate).")
    p.add_argument("--tau-rate-lo", type=float, default=50000.0,
            help="Event rate (ev/s) at/below which τ = tau_max_us.")
    p.add_argument("--tau-rate-hi", type=float, default=200000.0,
            help="Event rate (ev/s) at/above which τ = tau_min_us.")

    # ── Hot-pixel filter ──────────────────────────────────────────────────────
    p.add_argument("--disable-hotpixel", action="store_true",
            help="Disable the per-batch hot-pixel rate cap.")
    p.add_argument("--hotpixel-max-rate-hz", type=float, default=5000.0,
            help="Drop events from any pixel firing above this rate within a batch "
                 "(hot/stuck-pixel guard). Wire pixels fire <<5 kHz.")

    # ── ANF / STC filters ─────────────────────────────────────────────────────
    p.add_argument("--anf-threshold", dest="anf_threshold", type=int, default=20000,
                    help="ActivityNoiseFilter activity time threshold (us).")
    p.add_argument("--disable-anf", dest="disable_anf", action="store_true")

    p.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int, default=40000,
                    help="SpatioTemporalContrast threshold (us).")
    p.add_argument("--stc-cut-trail", dest="stc_cut_trail", action="store_true", default=False,
                    help="STC cut_trail flag.")
    p.add_argument("--disable-stc", dest="disable_stc", action="store_true")

    # ── Wire-mask thresholding ─────────────────────────────────────────────────
    p.add_argument("--wire-thr-floor", type=float, default=0.05,
            help="Lower clamp on the Otsu recentness threshold.")
    p.add_argument("--wire-thr-cap", type=float, default=0.15,
            help="Upper clamp on the Otsu threshold so dim wire tails survive. "
                 "Set to 1.0 to disable the cap (pure Otsu).")

    # ── Wire-mask morphology (kernels derived from wire diameter) ─────────────
    p.add_argument("--mask-close-mult", type=float, default=1.5,
            help="Close kernel size = round(wire_d_px * this) | 1.")
    p.add_argument("--mask-open-mult", type=float, default=0.3,
            help="Open kernel size = round(wire_d_px * this) | 1.")
    p.add_argument("--mask-close-iter-div", type=float, default=25.0,
            help="Close iterations = max(2, round(wire_d_px / this)).")

    # ── Wire-mask component filter (_keep_elongated) ──────────────────────────
    p.add_argument("--mask-min-aspect", type=float, default=2.5,
            help="Minimum major/minor aspect ratio to keep a component.")
    p.add_argument("--mask-min-length", type=float, default=20.0,
            help="Minimum major-axis length (px) to keep a component.")

    # ── Ridge wire mask (drop-in for Otsu+morphology) ─────────────────────────
    p.add_argument("--use-ridge-mask", action="store_true",
            help="Replace Otsu+morphology wire mask with the ridge filter.")
    p.add_argument("--ridge-filter", type=str, default="sobel",
            choices=["meijering", "sobel"],
            help="Ridge response filter. 'sobel' = OpenCV Sobel Hessian "
                 "(fast, real-time-friendly; default). 'meijering' = skimage "
                 "neuriteness (marginally cleaner, ~12x slower).")
    p.add_argument("--ridge-binarize", type=str, default="nms",
            choices=["nms", "hysteresis"],
            help="Ridge binarizer: 'nms' (geometric, default) or 'hysteresis' (A/B).")
    p.add_argument("--mask-gpu", action="store_true",
            help="Run the ridge wire-mask path on the GPU at full resolution. "
                 "Needs torch with CUDA and --ridge-binarize nms, otherwise "
                 "falls back to CPU. Overrides --mask-scale.")
    p.add_argument("--mask-scale", type=int, default=0,
            help="Downscale factor for the ridge wire-mask path; recentness, "
                 "ridge and skeleton run at 1/N, CPD stays full-res. "
                 "0 = auto (~12 px wire at low res), 1 = full resolution.")

    # ── Scale space ───────────────────────────────────────────────────────────
    p.add_argument("--ridge-sigma-lo", type=float, default=1.5,
            help="Lower Gaussian sigma for ridge scale space (px).")
    p.add_argument("--ridge-sigma-hi-mult", type=float, default=1.0,
            help="Upper sigma = wire_d_px * this.")

    # ── NMS path ──────────────────────────────────────────────────────────────
    p.add_argument("--ridge-nms-floor", type=float, default=0.10,
            help="Absolute floor on the normalised ridge response (NMS path). "
                 "Lower recovers dimmer wire sections; raise to reject noise.")
    p.add_argument("--ridge-band-dilate", type=float, default=-1.0,
            help="Band dilation radius in px after NMS. "
                 "<=0 = auto (wire_d_px/2). 0 keeps the raw 1-px ridge.")

    # ── Hysteresis path ───────────────────────────────────────────────────────
    p.add_argument("--ridge-hyst-high", type=float, default=0.15,
            help="Ridge seed threshold (fraction of per-frame max). "
                 "Raise to reject noise; lower to recover dim sections.")
    p.add_argument("--ridge-hyst-low", type=float, default=0.06,
            help="Ridge grow threshold. Lower fills more across gaps.")

    # ── Speed-adaptive ramp ───────────────────────────────────────────────────
    p.add_argument("--ridge-speed-lo", type=float, default=10.0,
            help="Node speed (px/frame) at/below which slow/thin settings are used.")
    p.add_argument("--ridge-speed-hi", type=float, default=35.0,
            help="Node speed (px/frame) at/above which fast/filled settings are used.")
    p.add_argument("--ridge-hyst-high-fast", type=float, default=0.08,
            help="Seed threshold at high speed (lower → seeds on the dim smear).")
    p.add_argument("--ridge-hyst-low-fast", type=float, default=0.04,
            help="Grow threshold at high speed (lower → fills the smear).")
    p.add_argument("--ridge-sigma-hi-mult-fast", type=float, default=2.5,
            help="Upper-sigma multiplier at high speed (wider → matches smear width).")
    p.add_argument("--ridge-close-fast-px", type=float, default=7.0,
            help="Morphological close radius in px at high speed (bridges smear gaps).")

    # ── Component filter ──────────────────────────────────────────────────────
    p.add_argument("--ridge-min-length", type=float, default=40.0,
            help="Min major-axis length (px) to keep a ridge component.")
    p.add_argument("--ridge-min-aspect", type=float, default=2.0,
            help="Min major/minor aspect ratio for ridge components.")
    p.add_argument("--ridge-top-k", type=int, default=5,
            help="Keep the top-K largest elongated components. "
                 "K>1 handles wires interrupted by static segments. "
                 "Raise --ridge-min-length if spurious short fragments appear.")

    # ── Spatial prior (corridor) ──────────────────────────────────────────────
    p.add_argument("--use-corridor", action="store_true",
            help="Gate R to a speed-scaled tube around the predicted curve before the ridge.")
    p.add_argument("--corridor-margin-px", type=float, default=40.0,
            help="Base half-width of the corridor in px (before the speed term).")
    p.add_argument("--corridor-speed-gain", type=float, default=1.5,
            help="Extra corridor half-width per px/frame of mean node speed.")

    # ── CPD / display / misc ──────────────────────────────────────────────────
    p.add_argument("--min-points", type=int, default=300)
    p.add_argument("--max-cloud", type=int, default=3000)
    p.add_argument("--n-ctrl", type=int, default=8)
    p.add_argument("--spline-samples", type=int, default=10)

    p.add_argument("--show", action="store_true", help="OpenCV debug window")
    p.add_argument("--save-vis", action="store_true",
                    help="Save the cpd visualisation of every iteration as a PNG into a "
                         "'results' folder next to the input .raw file. Clears the folder "
                         "contents on start. Independent of --show.")
    p.add_argument("--debug-images", dest="debug_images", action="store_true",
                    help="Write SAM3 mask/frame/overlay dumps and event-space debug "
                         "snapshots into a 'sam3_debug' folder. Off by default.")
    p.add_argument("--display-fps", type=int, default=0,
                    help="Cap display framerate (0 = uncapped)")
    p.add_argument("--display-scale", type=float, default=2.0,
                    help="Initial on-screen size of the debug windows as a multiple "
                         "of the event-sensor resolution.")

    # ── Hybrid tracking mode ──────────────────────────────────────────────────
    p.add_argument("--mode", type=str, default="hybrid",
                    choices=["event-only", "frame-only", "hybrid"],
                    help="Tracking mode")
    p.add_argument("--input-path", type=str,
                    default="path/to/events.raw",
                    help="Path to .raw event file (empty string = live camera)")
    p.add_argument("--camera-device", type=int, default=0,
                    help="OpenCV camera index for the RGB camera")
    p.add_argument("--frame-interval-ms", type=int, default=1000,
                    help="How often the frame worker runs SAM3 (ms)")

    # ── Paths (frame branch + outputs) ────────────────────────────────────────
    p.add_argument("--frames-dir", type=str,
                default="path/to/frames",
                help="Directory of RGB frames (offline mode).")
    p.add_argument("--manifest", type=str,
                default="path/to/manifest.json",
                help="Path to manifest.json mapping filename → event-time us.")
    p.add_argument("--homography", type=str,
                default="path/to/rgb_to_event_H.npy",
                help="Path to .npy file storing the RGB→event 3x3 homography H.")

    p.add_argument("--polyline-out", type=str, default="tracker_polylines.npz",
                help="Output path for the per-iteration tracker polylines.")
    p.add_argument("--sam3-polyline-out", type=str, default="sam3_polylines.npz",
                help="Output path for the per-trigger SAM3 polylines (event space).")
    p.add_argument("--timing-csv", type=str, default="timing.csv",
                help="Output CSV path for per-iteration timings.")

    # ── Motion-state detector (ρ_t hysteresis) ────────────────────────────────
    p.add_argument("--rho-thr", type=float, default=50.0,
                help="Threshold on count of events on wire_mask within rho_window_us. "
                     "Above → motion, at-or-below → static.")
    p.add_argument("--rho-window-us", type=int, default=20000,
                help="Width of the window over which ρ_t is accumulated, in µs.")
    p.add_argument("--rho-hysteresis-n", type=int, default=3,
                help="Number of consecutive windows ρ_t must stay on one side "
                     "of ρ_thr before the state flips.")

    # ── SAM3 worker ───────────────────────────────────────────────────────────
    p.add_argument("--sam3-retry-k", type=int, default=3,
                help="Maximum number of SAM3 retries on empty mask / extraction "
                     "failure within a single static period.")
    p.add_argument("--sam3-min-wire-d", type=float, default=3.0,
                help="Minimum SAM3-measured wire diameter (px) to accept and refresh "
                     "the morphology kernels.")

    # ── CPD tracker parameters ────────────────────────────────────────────────
    p.add_argument("--cpd-beta",        type=float, default=20.0)
    p.add_argument("--cpd-lam",         type=float, default=0.0001)
    p.add_argument("--cpd-mu",          type=float, default=0.1)
    p.add_argument("--cpd-k-vis",       type=float, default=0.05)
    p.add_argument("--cpd-max-iter",    type=int,   default=8)
    p.add_argument("--cpd-tol",         type=float, default=1e-4)
    p.add_argument("--cpd-beta-pre",    type=float, default=1.0)
    p.add_argument("--cpd-lam-pre",     type=float, default=0.0001)
    p.add_argument("--cpd-prune-thr",   type=float, default=300.0)

    # ── Wire-diameter fallback for boot-time morphology ───────────────────────
    p.add_argument("--default-wire-diameter-px", type=float, default=50.0,
                help="wire_d used for morphology kernels before SAM3 returns its "
                     "first measurement.")

    # ── Legacy / unused (kept so old scripts don't break) ────────────────────
    p.add_argument("--init-from-frame", action="store_true", default=True)
    p.add_argument("--init-timeout-s", type=float, default=2.0)
    p.add_argument("--blend-tau-ms", type=float, default=50.0)
    p.add_argument("--correction-max-age-ms", type=float, default=500.0)
    p.add_argument("--count-window-us", type=int, default=20000)
    p.add_argument("--count-threshold", type=int, default=3)
    p.add_argument("--hibernate-timeout-ms", type=float, default=500.0)
    p.add_argument("--hibernate-event-frac", type=float, default=0.05)

    return p.parse_args()