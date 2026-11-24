"""W4-E: replay mode - run labelled photos through the FULL phone pipeline (no camera), from the Termux app:

    python ~/bharatai/edge/scripts/replay.py            # own wood + fabric val/test sets in ~/bharatai/replay_data

For each product:
  1. "calibrate" on the val-good photos with the same rule the operator's Calibrate button uses (ADR-021);
  2. inspect every test photo (Inspector: NPU anomaly + YOLO + fusion + severity + files + SQLite, logged in
     ~/bharatai/replay/ so the real inspection log stays clean);
  3. report image AUROC (vs the laptop's 16a8w numbers), false alarms / recall at the phone-calibrated
     threshold, dispositions, and per-stage latency p50/p95.
Writes ~/bharatai/results/replay_w4.json
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home() / "bharatai"
sys.path.insert(0, str(HOME / "edge"))
os.environ.setdefault("BHARATAI_HOME", str(HOME))
from app.pipeline.calibrate import CalibrationStore, threshold_from_scores  # noqa: E402
from app.pipeline.inspector import Inspector  # noqa: E402
from app.pipeline.severity import SeverityRules  # noqa: E402
from app.pipeline.vision import VisionRuntime, load_config  # noqa: E402
from app.pipeline.vision_client import VisionClient, VisionServer  # noqa: E402
from app.storage.db import InspectionDB  # noqa: E402

DATA = HOME / "replay_data"          # <product>/{val/good, test/<label>}/*.jpg  (copied from the laptop datasets)
OUT = HOME / "replay"
LAPTOP_16A8W_AUROC = {"wood": 1.000, "fabric": 0.917}   # week 3, simulated quantised graph on the laptop
EXTS = {".jpg", ".jpeg", ".png"}


def auroc(y, s):
    y, s = np.array(y), np.array(s)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q)) if v else float("nan")


def main() -> None:
    cfg = load_config()
    run = HOME / "run"
    run.mkdir(parents=True, exist_ok=True)
    srv = VisionServer.start(str(HOME / "edge-bin"), str(HOME / "npu"), str(run / "vision.sock"), str(run / "vision_server.log"))
    report = {"device_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        vision = VisionRuntime(VisionClient(str(run / "vision.sock")), cfg)
        calib = CalibrationStore(OUT / "calibration.json")
        insp = Inspector(vision, SeverityRules(), InspectionDB(OUT / "replay.db"), calib, OUT / "images")
        for product in ("wood", "fabric"):
            root = DATA / product
            if not root.exists():
                continue
            val = sorted(p for p in (root / "val/good").iterdir() if p.suffix.lower() in EXTS)
            # replay only: wood has just 5 val-good photos; the live Calibrate button requires >= 10 (takes 20)
            entry = insp.calibrate(product, [cv2.imread(str(p)) for p in val], min_frames=5)
            y, s, rows, timings = [], [], [], []
            for d in sorted(x for x in (root / "test").iterdir() if x.is_dir()):
                for p in sorted(x for x in d.iterdir() if x.suffix.lower() in EXTS):
                    r = insp.inspect(product, cv2.imread(str(p)), note=f"replay {d.name}/{p.name}")
                    y.append(int(d.name != "good")); s.append(r.image_score)
                    rows.append({"file": f"{d.name}/{p.name}", "label": d.name, "score": round(r.image_score, 5),
                                 "disposition": r.disposition, "defects": [f"{x.type}:{x.severity}" for x in r.defects]})
                    timings.append(r.timings_ms)
            y_arr, s_arr = np.array(y), np.array(s)
            thr = entry["threshold"]
            per_label = {}
            for row in rows:
                per_label.setdefault(row["label"], []).append(row["score"] >= thr)
            stages = {k: {"p50": round(pct([t[k] for t in timings if k in t], 50), 1),
                          "p95": round(pct([t[k] for t in timings if k in t], 95), 1)}
                      for k in ("square", "pre_anomaly", "npu", "anomaly_total", "pre_yolo", "yolo_cpu", "yolo_total",
                                "vision_total", "fusion_severity", "save", "total")}
            disp = {}
            for row in rows:
                disp.setdefault(row["label"], {}).setdefault(row["disposition"], 0)
                disp[row["label"]][row["disposition"]] += 1
            report[product] = {
                "n_val_good": len(val), "n_test": len(rows),
                "phone_calibration": {k: round(v, 5) if isinstance(v, float) else v for k, v in entry.items() if k != "scores"},
                "image_auroc_phone": round(auroc(y_arr, s_arr), 4),
                "image_auroc_laptop_16a8w": LAPTOP_16A8W_AUROC[product],
                "false_alarms": f"{int((s_arr[y_arr == 0] >= thr).sum())}/{int((y_arr == 0).sum())}",
                "recall": f"{int((s_arr[y_arr == 1] >= thr).sum())}/{int((y_arr == 1).sum())}",
                "flagged_per_label": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_label.items())},
                "dispositions_per_label": disp,
                "latency_ms": stages,
            }
            (OUT / f"replay_rows_{product}.json").write_text(json.dumps(rows, indent=1))
    finally:
        srv.stop()
    out = HOME / "results/replay_w4.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("REPLAY_DONE")


if __name__ == "__main__":
    main()
