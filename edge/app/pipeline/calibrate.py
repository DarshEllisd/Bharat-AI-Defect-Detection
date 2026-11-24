"""Per-product calibration on the phone (ADR-021): threshold from the deployed model's own image scores on
~20 good frames under current conditions, saved until the operator re-calibrates.

Rule: threshold = 99th percentile of the good scores (ADR-019's rule) + margin, where
margin = max(MARGIN_SIGMA x std, MIN_MARGIN_FRAC x p99). With ~20 frames the 99th percentile is essentially
the maximum; the margin keeps a slightly unluckier good frame from tripping the alarm. The relative floor
matters on a fixed stand: near-identical calibration frames give std ~ 0, and without it the threshold
would equal the highest calibration score, so the same good item could trip it (found by a unit test).
The score statistics are stored so every inspection can be traced to the threshold that judged it.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

MIN_FRAMES = 10
MARGIN_SIGMA = 1.0
MIN_MARGIN_FRAC = 0.10


def threshold_from_scores(scores: list[float], min_frames: int = MIN_FRAMES) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    if len(s) < min_frames:
        raise ValueError(f"need at least {min_frames} good frames, got {len(s)}")
    p99, sd = float(np.percentile(s, 99)), float(s.std())
    margin = max(MARGIN_SIGMA * sd, MIN_MARGIN_FRAC * p99)
    return {"threshold": p99 + margin, "p99": p99, "std": sd, "margin": margin, "mean": float(s.mean()),
            "min": float(s.min()), "max": float(s.max()), "n_frames": int(len(s)),
            "rule": f"p99 + max({MARGIN_SIGMA:g} x std, {MIN_MARGIN_FRAC:.0%} x p99) of good-frame scores"}


class CalibrationStore:
    """state/calibration.json: {product: {id, ts, threshold, ...stats}}"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, product: str) -> dict | None:
        return self.data.get(product)

    def save(self, product: str, scores: list[float], min_frames: int = MIN_FRAMES, model: str = "") -> dict:
        entry = threshold_from_scores(scores, min_frames)
        entry["model"] = model          # the calibration belongs to this exact model file (P-71)
        entry.update({"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "scores": [round(float(v), 6) for v in scores]})
        self.data[product] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)
        return entry
