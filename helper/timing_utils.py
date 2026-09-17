"""Shared timing helpers so the frame-based and video-based scripts report
per-frame latency the same way.

Latency is per frame in ms. The first `warmup` samples are dropped (lazy
model load, first-call CUDA capture, first propagation step); default 1,
raise it if the GPU needs more. Reports mean +/- sample std (ddof=1),
p50 and p95.

frame-based feeds in per-frame SAM 3 segmentation times (s); video-based
feeds in per-frame propagation step times (s). add_prompt is a one-time
cost and is reported separately.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np


def summarize(times_s: List[float], warmup: int = 1, label: str = "") -> Dict:
    t = list(times_s)
    dropped = min(warmup, max(len(t) - 1, 0))   # never drop the only sample
    kept = t[dropped:]
    arr = np.asarray(kept, dtype=float) * 1000.0  # -> ms
    if arr.size == 0:
        return {"label": label, "n": 0, "warmup_dropped": dropped}
    return {
        "label":          label,
        "n":              int(arr.size),
        "warmup_dropped": dropped,
        "mean_ms":        float(arr.mean()),
        "std_ms":         float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50_ms":         float(np.percentile(arr, 50)),
        "p95_ms":         float(np.percentile(arr, 95)),
        "min_ms":         float(arr.min()),
        "max_ms":         float(arr.max()),
        "total_s":        float(arr.sum() / 1000.0),
    }


def fmt(s: Dict) -> str:
    if s.get("n", 0) == 0:
        return f"{s.get('label', '')}: no samples (warmup_dropped={s.get('warmup_dropped', 0)})"
    return (f"{s['label']}: {s['mean_ms']:.1f} +/- {s['std_ms']:.1f} ms/frame "
            f"(n={s['n']}, p50={s['p50_ms']:.1f}, p95={s['p95_ms']:.1f} ms, "
            f"warmup dropped={s['warmup_dropped']})")


def budget_line(mean_ms: float, budget_ms: float) -> str:
    """One-line real-time verdict against an update budget (e.g. event branch)."""
    ratio = mean_ms / budget_ms if budget_ms > 0 else float("inf")
    verdict = "WITHIN" if mean_ms <= budget_ms else "EXCEEDS"
    return (f"per-frame {mean_ms:.1f} ms vs budget {budget_ms:.1f} ms "
            f"-> {verdict} ({ratio:.1f}x budget)")


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)