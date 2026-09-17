
import numpy as np
try:
    from metavision_sdk_cv import (
        ActivityNoiseFilterAlgorithm,
        SpatioTemporalContrastAlgorithm,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def adaptive_tau_us(event_rate, tau_min=4000, tau_max=50000, r_lo=5e4, r_hi=3e5):
    """
    event_rate in events/sec.
    Higher rate -> smaller tau (faster decay).
    """
    if event_rate <= r_lo:
        return tau_max
    if event_rate >= r_hi:
        return tau_min
    # linear interpolation in [r_lo, r_hi]
    a = (event_rate - r_lo) / (r_hi - r_lo)
    return int(tau_max + a * (tau_min - tau_max))


def _make_anf(w, h, args):
    
    if not _SDK_AVAILABLE or getattr(args, "disable_anf", True):
        return None, None
    f = ActivityNoiseFilterAlgorithm(w, h, getattr(args, "anf_threshold", 20000))
    return f, f.get_empty_output_buffer()


def _make_stc(w, h, args):
    if not _SDK_AVAILABLE or args.disable_stc:
        return None, None
    print(f"Instantiating SpatioTemporalContrastAlgorithm with thresh={args.stc_filter_thr}")
    f = SpatioTemporalContrastAlgorithm(
        width=w, height=h, threshold=args.stc_filter_thr, cut_trail=args.stc_cut_trail,
    )
    return f, f.get_empty_output_buffer()

def _hotpixel_cap(ev, w, max_rate_hz):
    n = len(ev)
    if n == 0:
        return ev
    dt_s = max((int(ev["t"][-1]) - int(ev["t"][0])) / 1e6, 1e-6)
    cap  = max(int(max_rate_hz * dt_s), 1)
    idx  = ev["y"].astype(np.int64) * w + ev["x"]
    cnt  = np.bincount(idx)
    return np.ascontiguousarray(ev[cnt[idx] <= cap])


class EventFilterChain:
    """Builds and runs the anf → stc Metavision filter chain."""

    def __init__(self, w, h, args):
        self.w = w
        self.disable_hotpixel    = getattr(args, "disable_hotpixel", False)
        self.hotpixel_max_rate_hz = getattr(args, "hotpixel_max_rate_hz", 5000.0)
        self.anf, self.anf_out = _make_anf(w, h, args)
        self.stc, self.stc_out = _make_stc(w, h, args)

    def apply(self, ev):
        if not self.disable_hotpixel:
            ev = _hotpixel_cap(ev, self.w, self.hotpixel_max_rate_hz)
            if len(ev) == 0:
                return ev
        if self.anf is not None:
            self.anf.process_events(ev, self.anf_out)
            ev = self.anf_out.numpy()
        if self.stc is not None:
            self.stc.process_events(ev, self.stc_out)
            ev = self.stc_out.numpy()
        return ev

def filter_events(it, w, h, args):
    
    chain = EventFilterChain(w, h, args)
    return (
        None,      None,
        None,      None,
        chain.stc, chain.stc_out,
    )
