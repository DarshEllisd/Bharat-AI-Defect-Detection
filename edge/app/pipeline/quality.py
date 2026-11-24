"""Frame check before a frame is judged or used for calibration (week-4 live test, P-68).

The models only know images taken the way the training photos were (ADR-020). The live fabric session showed
frames far outside that: a black frame (camera covered) and shots 2-3x closer than training (towel loop period
41-47 px vs 12.6-34.1 px in all training/test photos). Such frames get a "retake" message instead of a verdict, and aren't logged.
  - brightness: mean grey level of the 1024 px square must be >= MIN_BRIGHTNESS;
  - texture scale (products with `scale_px` in categories.toml, i.e. fabric): the dominant spatial period of
    the texture in the 1024 px square (same estimator used to diagnose P-51) must lie inside the range.
Blur is NOT a reject reason: genuine defect photos are often soft (training cut photos: sharpness 41-459).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_BRIGHTNESS = 40.0
MIN_TEXTURE_STD = 1.0


def texture_period(square_bgr: np.ndarray) -> float:
    """Dominant spatial period (px, in the 1024 px square) from the radially averaged power spectrum."""
    g = cv2.resize(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (512, 512),
                   interpolation=cv2.INTER_AREA)
    g -= cv2.GaussianBlur(g, (0, 0), 25)
    if g.std() < MIN_TEXTURE_STD:     # featureless (e.g. a flat test frame): nothing to measure
        return float("nan")
    win = np.hanning(512).astype(np.float32)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g * win[:, None] * win[None]))) ** 2
    y, x = np.indices(f.shape)
    r = np.hypot(x - 256, y - 256).astype(int)
    prof = np.bincount(r.ravel(), f.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    k = 8 + int(np.argmax(prof[8:120] * np.arange(8, 120)))
    return 1024.0 / k


def check(square_bgr: np.ndarray, product_cfg: dict) -> dict:
    """-> {ok, message, brightness, period}. message is operator guidance when not ok."""
    out = {"ok": True, "message": "", "brightness": float(square_bgr.mean()), "period": None}
    if out["brightness"] < MIN_BRIGHTNESS:
        out.update(ok=False, message=f"Image too dark (brightness {out['brightness']:.0f}). "
                                     "Check the lamp and that nothing covers the camera, then retake.")
        return out
    rng = product_cfg.get("scale_px")
    if rng:
        p = texture_period(square_bgr)
        if np.isnan(p):
            return out
        out["period"] = round(p, 1)
        lo, hi = rng
        if p > hi:
            out.update(ok=False, message=f"Camera too CLOSE to the {product_cfg.get('label', 'item')} "
                                         f"(texture scale {p:.0f} px, expected {lo}-{hi}). Raise the phone / lower the item, then retake.")
        elif p < lo:
            out.update(ok=False, message=f"Camera too FAR from the {product_cfg.get('label', 'item')} "
                                         f"(texture scale {p:.0f} px, expected {lo}-{hi}). Lower the phone / raise the item, then retake.")
    return out
