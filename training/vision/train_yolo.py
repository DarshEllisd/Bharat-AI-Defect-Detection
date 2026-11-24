"""W3-C/D - train YOLO11n on the combined MVTec + own dataset, then evaluate it (ADR-018/019).

    ~/bharatai-build/train-venv/bin/python training/vision/train_yolo.py            # train + evaluate
    ~/bharatai-build/train-venv/bin/python training/vision/train_yolo.py --eval-only

Training: COCO-pretrained yolo11n.pt, 320 px, standard Ultralytics augmentation with a slightly
stronger brightness range (hsv_v 0.5 instead of 0.4: phone photos vary in exposure), early stopping
on the validation split.
Evaluation (ADR-019):
  1. VALIDATION split -> operating confidence = the one that maximises the class-mean F1 curve.
  2. TEST, reported separately for the MVTec test split and the own real-photo test split:
     mAP50, mAP50-95 (all and per class) and precision/recall at the confidence chosen in step 1.
Outputs: ~/bharatai-build/models/yolo/yolo11n_combined/{weights/best.pt, eval.json, ...}
"""
import argparse
import json
from pathlib import Path

import os

import numpy as np
import yaml
from ultralytics import YOLO, settings

# No usage analytics leave this machine (organisation data policy) - Ultralytics enables them by default.
settings.update({"sync": False})

B = Path.home() / "bharatai-build"
os.chdir(B)  # Ultralytics' AMP check looks for yolo11n.pt in the CWD (already there; avoids a download into the repo)
DATA = B / "datasets/yolo_combined/data.yaml"
PROJECT = B / "models/yolo"
NAME = "yolo11n_combined"
IMGSZ = 320


def split_yaml(tag: str, test_dirs: list[str]) -> Path:
    """data.yaml copy whose 'test' is only the given directories (to report MVTec and own separately)."""
    cfg = yaml.safe_load(DATA.read_text())
    cfg["test"] = test_dirs
    out = DATA.parent / f"data_test_{tag}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def at_conf(metrics, conf: float) -> dict:
    """Class-mean precision/recall/F1 at a confidence, from Ultralytics' curves (classes present only)."""
    i = int(np.argmin(np.abs(metrics.box.px - conf)))
    p, r, f1 = (float(np.mean(c[:, i])) for c in (metrics.box.p_curve, metrics.box.r_curve, metrics.box.f1_curve))
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def per_class(metrics, names: dict) -> dict:
    return {names[int(c)]: {"mAP50": round(float(metrics.box.all_ap[j, 0]), 4),
                            "mAP50_95": round(float(metrics.box.all_ap[j].mean()), 4)}
            for j, c in enumerate(metrics.box.ap_class_index)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    run_dir = PROJECT / NAME

    if not args.eval_only:
        YOLO(str(B / "yolo11n.pt")).train(
            data=str(DATA), imgsz=IMGSZ, epochs=args.epochs, patience=args.patience, batch=args.batch,
            hsv_v=0.5, seed=0, deterministic=True, workers=4, project=str(PROJECT), name=NAME,
            exist_ok=True, plots=True, verbose=False)

    model = YOLO(str(run_dir / "weights/best.pt"))
    names = model.names
    common = dict(imgsz=IMGSZ, batch=args.batch, conf=0.001, iou=0.6, plots=False, verbose=False,
                  project=str(run_dir), exist_ok=True)

    val = model.val(data=str(DATA), split="val", name="eval_val", **common)
    mean_f1 = val.box.f1_curve.mean(0)
    conf = float(val.box.px[int(np.argmax(mean_f1))])
    res = {"operating_conf": round(conf, 3), "conf_rule": "argmax of class-mean F1 on the validation split",
           "val": {"mAP50": round(float(val.box.map50), 4), "mAP50_95": round(float(val.box.map), 4),
                   **at_conf(val, conf)}}

    cfg = yaml.safe_load(DATA.read_text())
    for tag, dirs in (("mvtec", [d for d in cfg["test"] if "yolo_mvtec" in d]),
                      ("own", [d for d in cfg["test"] if "/own/" in d])):
        m = model.val(data=str(split_yaml(tag, dirs)), split="test", name=f"eval_test_{tag}", **common)
        res[f"test_{tag}"] = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                              **at_conf(m, conf), "per_class": per_class(m, names)}

    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per_class"} if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
