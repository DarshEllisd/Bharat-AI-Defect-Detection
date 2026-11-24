"""Rule-based severity and disposition (ADR-006), driven by config/severity.toml. Deterministic and
explainable: every level comes with the reasons that produced it."""
from __future__ import annotations

import tomllib
from pathlib import Path

from .record import DefectRecord

DEFAULT_RULES = Path(__file__).resolve().parents[2] / "config/severity.toml"


class SeverityRules:
    def __init__(self, path: Path | str = DEFAULT_RULES) -> None:
        with open(path, "rb") as fh:
            self.cfg = tomllib.load(fh)
        self.levels: list[str] = self.cfg["levels"]

    def grade(self, d: DefectRecord, rule_set: str, threshold: float) -> DefectRecord:
        """Set d.level, d.severity, d.reasons in place (and return d)."""
        types = self.cfg["types"][rule_set]
        entry = types.get(d.type) or types["unknown_anomaly"]
        level = int(entry["base"])
        reasons = [f"base {self.levels[level]} for {d.type if d.type in types else 'unknown_anomaly'}: {entry['why']}"]
        b = self.cfg["bumps"]
        if d.area_frac >= b["large_area_frac"] * (1 - 1e-9):
            level += 1
            reasons.append(f"+1 large: area {d.area_frac:.1%} >= {b['large_area_frac']:.0%} of the square")
        if threshold > 0 and d.peak_score >= b["strong_ratio"] * threshold * (1 - 1e-9):   # tolerant at the boundary
            level += 1
            reasons.append(f"+1 strong: peak {d.peak_score:.3g} >= {b['strong_ratio']:g}x threshold {threshold:.3g}")
        cap = entry.get("max")
        if cap is not None and level > int(cap):
            level = int(cap)
            reasons.append(f"capped at {self.levels[level]}: {d.type} is at most {self.levels[level]} (severity.toml)")
        level = min(level, len(self.levels) - 1)
        if d.source == "detector_only" and not self.cfg["detector_only"].get("counts_for_verdict", True):
            # user decision: YOLO alone is a note, not a verdict (the anomaly model didn't confirm it)
            reasons.append(f"NOTE only: YOLO saw '{d.type}' but the anomaly model didn't confirm it - "
                           f"would be {self.levels[level]}, not used for the verdict")
            d.level, d.severity, d.reasons = 0, "NOTE", reasons
            return d
        d.level, d.severity, d.reasons = level, self.levels[level], reasons
        return d

    def disposition(self, defects: list[DefectRecord]) -> tuple[str, str]:
        """(worst severity name, disposition) for an item."""
        worst = max((d.level for d in defects), default=0)
        name = self.levels[worst]
        return name, self.cfg["disposition"][name]

    @property
    def detector_only_min_conf(self) -> float:
        return float(self.cfg["detector_only"]["min_conf"])
