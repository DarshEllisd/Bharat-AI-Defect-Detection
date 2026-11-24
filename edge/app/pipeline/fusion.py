"""Hybrid fusion (ADR-005): anomaly-map regions + YOLO boxes -> typed defect records.

1. Regions: connected components of the anomaly map at >= threshold. A region's extent = its pixels that
   are >= max(threshold, HALO_FRAC x its own peak), which trims the blurred halo the 56->256 upsampling puts
   around very strong defects. (A first version grew regions down to 0.5 x threshold; with a calibrated
   threshold that swallowed normal texture - wood liquid regions covered a median 50% of the square in the
   phone replay, P-67.) Above-threshold pixels closer than MERGE_FRAC of the map width are grouped into
   one region (a dilated mask is labelled), so one stain doesn't become many "defects"; areas still count
   only the real above-threshold pixels.
2. Each region is matched to the overlapping YOLO box with the highest confidence (IoU >= MATCH_IOU or the
   box centre inside the region bbox):
     matched   -> type from YOLO, source "fused"
     unmatched -> "unknown_anomaly", source "anomaly_only"
3. YOLO boxes with conf >= detector_only_min_conf that match no region -> source "detector_only".
"""
from __future__ import annotations

import cv2
import numpy as np

from .record import DefectRecord

HALO_FRAC = 0.25
MERGE_FRAC = 0.03                # regions closer than ~3% of the map width are one defect
MATCH_IOU = 0.1
MIN_REGION_PX_FRAC = 0.0002      # ignore specks smaller than 0.02% of the map


def decode_yolo(out: np.ndarray, names: list[str], prefix: str, conf: float, nms_iou: float,
                input_px: int = 320) -> list[dict]:
    """YOLO (1, 4+nc, N) with boxes in px of the input -> [{cls, type, conf, bbox(norm xyxy)}], filtered to
    the product's classes (prefix), class-wise NMS."""
    o = out.reshape(4 + len(names), -1)
    keep_cls = [i for i, n in enumerate(names) if n.startswith(prefix)]
    if not keep_cls:
        return []
    scores = o[4:][keep_cls]                       # (k, N)
    best = scores.argmax(0)
    best_conf = scores[best, np.arange(scores.shape[1])]
    idx = np.where(best_conf >= conf)[0]
    dets = []
    for i in idx:
        cx, cy, w, h = o[:4, i] / input_px
        dets.append({"cls": keep_cls[best[i]], "conf": float(best_conf[i]),
                     "bbox": [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]})
    dets.sort(key=lambda d: -d["conf"])
    kept: list[dict] = []
    for d in dets:
        if all(k["cls"] != d["cls"] or iou(k["bbox"], d["bbox"]) < nms_iou for k in kept):
            kept.append(d)
    for d in kept:
        d["type"] = names[d["cls"]][len(prefix):]
        d["bbox"] = [min(1.0, max(0.0, v)) for v in d["bbox"]]
    return kept


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def regions(amap: np.ndarray, threshold: float) -> list[dict]:
    """Regions above threshold on the anomaly map -> [{bbox(norm), area_frac, peak}]"""
    h, w = amap.shape
    above = (amap >= threshold).astype(np.uint8)
    if not above.any():
        return []
    k = max(3, int(round(MERGE_FRAC * w)) | 1)
    grouped = cv2.dilate(above, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, _, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    peak_comp = lab.flat[int(np.argmax(amap))]      # the component holding the image's maximum
    out = []
    for c in range(1, n):
        mask = (lab == c) & (above > 0)
        # tiny specks are ignored, EXCEPT the region with the image's maximum: if the image score is above the
        # threshold, the verdict must show where (live small cuts/threads were passed silently - P-73)
        if mask.sum() < MIN_REGION_PX_FRAC * h * w and c != peak_comp:
            continue
        peak = float(amap[mask].max())
        core = mask & (amap >= max(threshold, HALO_FRAC * peak))
        ys, xs = np.nonzero(core)
        out.append({"bbox": [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h],
                    "area_frac": float(core.sum() / (h * w)), "peak": peak})
    out.sort(key=lambda r: -r["peak"])
    return out


def _centre_in(box: list[float], region_box: list[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region_box[0] <= cx <= region_box[2] and region_box[1] <= cy <= region_box[3]


def fuse(amap: np.ndarray, threshold: float, dets: list[dict], detector_only_min_conf: float) -> list[DefectRecord]:
    regs = regions(amap, threshold)
    used = set()
    defects = []
    for r in regs:
        cands = [(i, d) for i, d in enumerate(dets) if iou(d["bbox"], r["bbox"]) >= MATCH_IOU or _centre_in(d["bbox"], r["bbox"])]
        if cands:
            i, d = max(cands, key=lambda c: c[1]["conf"])
            used.update(i for i, _ in cands)
            defects.append(DefectRecord(type=d["type"], source="fused", bbox=r["bbox"], area_frac=r["area_frac"],
                                        peak_score=r["peak"], yolo_conf=d["conf"]))
        else:
            defects.append(DefectRecord(type="unknown_anomaly", source="anomaly_only", bbox=r["bbox"],
                                        area_frac=r["area_frac"], peak_score=r["peak"]))
    for i, d in enumerate(dets):
        if i not in used and d["conf"] >= detector_only_min_conf:
            b = d["bbox"]
            defects.append(DefectRecord(type=d["type"], source="detector_only", bbox=b,
                                        area_frac=(b[2] - b[0]) * (b[3] - b[1]), peak_score=0.0, yolo_conf=d["conf"]))
    return defects
