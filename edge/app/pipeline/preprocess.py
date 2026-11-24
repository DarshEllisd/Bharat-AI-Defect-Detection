"""Shared preprocessing for the phone pipeline (P-60: must match training exactly).

Training data path (weeks 2-3):
  capture frame 3072x4096 -> capture-tool square ROI (2458 px) -> resize to 1024 px, cv2.INTER_AREA
  (merge_own_data.py, saved as JPEG q95) -> model input:
    anomaly: resize to 256*T px with PIL bilinear (antialiased) -> T*T tiles of 256 px ->
             /255 -> ImageNet mean/std, NCHW float32. PIL, not OpenCV: INTER_AREA (box filter) changed
             fabric texture enough to drop AUROC 0.918 -> 0.877 (parity check, P-63).
    YOLO:    resize to 320 px (Ultralytics: cv2.INTER_AREA when shrinking) -> RGB /255, NCHW float32
Edge path (this module): the same steps and the same resampling functions. Parity with the
training code is checked by edge/scripts/parity_preprocess.py.
Needs numpy, OpenCV and Pillow (all available in Termux; Pillow 13.0 on both phone and laptop).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image  # Termux: python-pillow (13.0 on the phone)

STORE_PX = 1024
TILE = 256
YOLO_PX = 320
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def to_square(frame_bgr: np.ndarray, roi_xywh: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Crop the capture square (if a ROI is given; the browser normally sends only the square) and resize
    to the stored-training size (1024 px, INTER_AREA) - the same step merge_own_data.py applied."""
    img = frame_bgr
    if roi_xywh is not None:
        x, y, s, _ = roi_xywh
        img = img[y:y + s, x:x + s]
    if img.shape[0] != img.shape[1]:
        raise ValueError(f"expected a square image, got {img.shape[:2]}")
    if img.shape[0] != STORE_PX:
        interp = cv2.INTER_AREA if img.shape[0] > STORE_PX else cv2.INTER_LINEAR
        img = cv2.resize(img, (STORE_PX, STORE_PX), interpolation=interp)
    return img


def _to_nchw(rgb_float01: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb_float01.transpose(2, 0, 1)[None], dtype=np.float32)


def anomaly_inputs(square_bgr: np.ndarray, tiles: int = 1) -> list[np.ndarray]:
    """1024 px square -> list of T*T normalised 1x3x256x256 tiles (row-major), as in training."""
    size = TILE * tiles
    rgb = Image.fromarray(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB)).resize((size, size), Image.BILINEAR)
    x = _to_nchw(np.asarray(rgb, dtype=np.float32) / 255.0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return [np.ascontiguousarray(x[:, :, r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE])
            for r in range(tiles) for c in range(tiles)]


def stitch(tile_maps: list[np.ndarray], tiles: int) -> np.ndarray:
    """T*T maps of 256x256 (row-major) -> one (256*T)x(256*T) map."""
    rows = [np.hstack([m.reshape(TILE, TILE) for m in tile_maps[r * tiles:(r + 1) * tiles]]) for r in range(tiles)]
    return np.vstack(rows)


def yolo_input(square_bgr: np.ndarray) -> np.ndarray:
    """1024 px square -> 1x3x320x320 RGB in [0,1] (square, so no letterbox padding is needed)."""
    img = cv2.resize(square_bgr, (YOLO_PX, YOLO_PX), interpolation=cv2.INTER_AREA)
    return _to_nchw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
