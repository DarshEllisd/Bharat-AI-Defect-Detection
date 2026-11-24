"""One Inspect press -> one InspectionRecord (logged), and per-product calibration.

inspect(): capture square -> 1024 px -> anomaly map (NPU, tiled for fabric) + YOLO (CPU) -> fusion ->
severity/disposition -> files (square JPEG, heatmap overlay, defect crops) -> SQLite row.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import preprocess as pp
from .calibrate import MIN_FRAMES, CalibrationStore
from .fusion import fuse
from .quality import check as frame_check
from .record import InspectionRecord
from .severity import SeverityRules
from .vision import VisionRuntime
from ..storage.db import InspectionDB

CROP_PAD = 0.15        # crop = defect bbox + 15% padding each side
SEV_COLORS = {1: (0, 200, 255), 2: (0, 140, 255), 3: (0, 0, 255), 4: (180, 0, 255)}   # BGR


class FrameRejected(ValueError):
    """The frame isn't suitable for judging (too dark, wrong camera distance); carries operator guidance."""

    def __init__(self, info: dict) -> None:
        super().__init__(info["message"])
        self.info = info


class Inspector:
    def __init__(self, vision: VisionRuntime, rules: SeverityRules, db: InspectionDB, calib: CalibrationStore,
                 data_dir: Path | str) -> None:
        self.v, self.rules, self.db, self.calib = vision, rules, db, calib
        self.data_dir = Path(data_dir)

    def model_id(self, product: str) -> str:
        """Identity of the deployed anomaly model file (name + size + mtime): a calibration is only valid for
        the exact model it was made with; a retrained model of the same name is a different model (P-71)."""
        f = Path(self.v.cfg["models_dir"]) / self.v.cfg["products"][product]["anomaly_pte"]
        try:
            st = f.stat()
            return f"{f.name}:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return f.name

    def calibration(self, product: str) -> dict | None:
        c = self.calib.get(product)
        return c if c and c.get("model") == self.model_id(product) else None

    def threshold(self, product: str) -> tuple[float, bool, str]:
        c = self.calibration(product)
        if c:
            return float(c["threshold"]), True, c["id"]
        return float(self.v.cfg["products"][product]["default_threshold"]), False, ""

    def check_frame(self, product: str, square: np.ndarray) -> dict:
        info = frame_check(square, self.v.cfg["products"][product])
        if not info["ok"]:
            raise FrameRejected(info)
        return info

    # --- calibration ---------------------------------------------------------------------------
    def calibrate(self, product: str, frames_bgr: list[np.ndarray], min_frames: int | None = None) -> dict:
        scores = []
        for f in frames_bgr:
            sq = pp.to_square(f)
            self.check_frame(product, sq)
            amap, _ = self.v.anomaly(product, sq)
            scores.append(float(amap.max()))
        entry = self.calib.save(product, scores, min_frames or MIN_FRAMES, model=self.model_id(product))
        self.db.add_calibration(product, entry)
        return entry

    # --- inspection ----------------------------------------------------------------------------
    def inspect(self, product: str, frame_bgr: np.ndarray, note: str = "") -> InspectionRecord:
        t0 = time.perf_counter()
        square = pp.to_square(frame_bgr)
        frame_info = self.check_frame(product, square)      # raises FrameRejected -> "retake", nothing logged
        t_sq = time.perf_counter()
        amap, ta = self.v.anomaly(product, square)
        dets, ty = self.v.detect(product, square)
        t_v = time.perf_counter()
        thr, calibrated, cal_id = self.threshold(product)
        rule_set = self.v.cfg["products"][product]["rules"]
        defects = fuse(amap, thr, dets, self.rules.detector_only_min_conf)
        for d in defects:
            self.rules.grade(d, rule_set, thr)
        defects.sort(key=lambda d: (-d.level, -d.peak_score))
        severity, disposition = self.rules.disposition(defects)
        t_f = time.perf_counter()

        rid = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        day = self.data_dir / rid[:8]
        day.mkdir(parents=True, exist_ok=True)
        img_path = day / f"{rid}.jpg"
        cv2.imwrite(str(img_path), square, [cv2.IMWRITE_JPEG_QUALITY, 92])
        heat_path = day / f"{rid}_heat.jpg"
        cv2.imwrite(str(heat_path), self.overlay(square, amap, thr, defects), [cv2.IMWRITE_JPEG_QUALITY, 85])
        s = square.shape[0]
        for i, d in enumerate(defects):
            x1, y1, x2, y2 = d.bbox
            px, py = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
            a, b = int(max(0, (x1 - px) * s)), int(max(0, (y1 - py) * s))
            c, e = int(min(s, (x2 + px) * s)), int(min(s, (y2 + py) * s))
            if c - a >= 4 and e - b >= 4:
                cp = day / f"{rid}_d{i}.jpg"
                cv2.imwrite(str(cp), square[b:e, a:c], [cv2.IMWRITE_JPEG_QUALITY, 90])
                d.crop_path = str(cp)
        t_s = time.perf_counter()

        timings = {"square": (t_sq - t0) * 1e3, **{k: round(v, 2) for k, v in {**ta, **ty}.items()},
                   "vision_total": (t_v - t_sq) * 1e3, "fusion_severity": (t_f - t_v) * 1e3, "save": (t_s - t_f) * 1e3}
        timings["total"] = (time.perf_counter() - t0) * 1e3
        rec = InspectionRecord(
            id=rid, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), product=product, disposition=disposition,
            severity=severity, image_score=float(amap.max()), threshold=thr, calibrated=calibrated,
            calibration_id=cal_id, defects=defects, timings_ms={k: round(float(v), 2) for k, v in timings.items()},
            image_path=str(img_path), heatmap_path=str(heat_path),
            models={"anomaly": self.v.cfg["products"][product]["anomaly_pte"], "yolo": self.v.cfg["yolo"]["pte"]},
            note=" ".join(x for x in (note, f"texture {frame_info['period']} px" if frame_info["period"] else "") if x))
        self.db.add_inspection(rec)
        return rec

    @staticmethod
    def overlay(square: np.ndarray, amap: np.ndarray, thr: float, defects) -> np.ndarray:
        """Heatmap (red = at/above threshold) blended on the image, defect boxes coloured by severity."""
        s = square.shape[0]
        m = cv2.resize(amap.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(np.clip(m / max(2 * thr, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(square, 0.55, heat, 0.45, 0)
        for d in defects:
            x1, y1, x2, y2 = (int(v * s) for v in d.bbox)
            col = SEV_COLORS.get(d.level, (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
            cv2.putText(out, f"{d.type} {d.severity}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        return out
