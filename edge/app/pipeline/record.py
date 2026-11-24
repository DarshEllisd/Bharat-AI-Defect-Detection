"""Inspection records: the one data structure every stage produces/consumes (logging, UI, week-5 advisor).

Coordinates are normalised to the inspected square (0..1, x right, y down). JSON round-trips exactly
(`InspectionRecord.from_dict(r.to_dict()) == r`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

SCHEMA_VERSION = 1


@dataclass
class DefectRecord:
    type: str                      # e.g. "hole", "thread", "unknown_anomaly"
    source: str                    # "fused" (anomaly region + YOLO), "anomaly_only", "detector_only"
    bbox: list[float]              # [x1, y1, x2, y2] normalised
    area_frac: float               # region area / square area (anomaly pixels; YOLO box area if detector_only)
    peak_score: float              # max anomaly-map value inside the region (0 if detector_only)
    yolo_conf: float = 0.0         # confidence of the matched YOLO box (0 if none)
    level: int = 0                 # 1 LOW .. 4 CRITICAL
    severity: str = ""             # level name
    reasons: list[str] = field(default_factory=list)   # why this level (base / bumps / caps)
    crop_path: str = ""


@dataclass
class InspectionRecord:
    id: str
    ts: str                        # ISO time on the phone
    product: str
    disposition: str               # PASS / REWORK / HOLD / SCRAP
    severity: str                  # worst defect level name, NONE if no defect
    image_score: float             # max of the anomaly map
    threshold: float
    calibrated: bool               # False -> the laptop default threshold was used (ADR-021)
    calibration_id: str = ""
    defects: list[DefectRecord] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    image_path: str = ""
    heatmap_path: str = ""
    models: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    note: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InspectionRecord":
        d = dict(d)
        d["defects"] = [DefectRecord(**x) for x in d.get("defects", [])]
        return cls(**d)
