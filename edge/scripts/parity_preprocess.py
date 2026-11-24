"""W4-B/D: parity of the phone preprocessing (edge/app/pipeline/preprocess.py) with the training code.

    ~/bharatai-build/train-venv/bin/python edge/scripts/parity_preprocess.py

For own_wood (1 tile) and own_fabric_t2 (2x2 tiles), on their val + test images:
  - input difference: max / mean abs difference of the normalised model inputs (edge vs training);
  - fp32 model image scores both ways: correlation, max relative difference, AUROC both ways.
Training path = train_efficientad.load_rgb (PIL bilinear, antialiased) + ImageNet normalisation + tiles.
Acceptance: AUROC difference <= 0.005 and score correlation >= 0.999.
Writes edge/scripts/parity_preprocess.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training/vision"))
sys.path.insert(0, str(REPO))
from efficientad_model import IMAGENET_MEAN, IMAGENET_STD, EfficientADS  # noqa: E402
from train_efficientad import load_rgb, tiles_of  # noqa: E402

from edge.app.pipeline.preprocess import anomaly_inputs, to_square  # noqa: E402

B = Path.home() / "bharatai-build"
EXTS = {".jpg", ".jpeg", ".png"}


def auroc(y, s):
    y, s = np.array(y), np.array(s)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    report = {}
    for name, product in (("own_wood", "wood"), ("own_fabric_t2", "fabric")):
        cfg = json.loads((B / "models/efficientad" / name / "config.json").read_text())
        T = int(cfg.get("tiles", 1))
        model = EfficientADS(256)
        model.load_state_dict(torch.load(B / "models/efficientad" / name / "model.pt", map_location="cpu", weights_only=True))
        model.eval().to(dev)
        root = B / "datasets/anomaly_own" / product
        items = [(p, int(p.parent.name != "good")) for d in [root / "val/good"] + sorted((root / "test").iterdir())
                 for p in sorted(d.glob("*")) if p.suffix.lower() in EXTS]
        diffs, s_train, s_edge, y_test = [], [], [], []
        with torch.no_grad():
            for p, y in items:
                tr = [((tl - mean) / std).unsqueeze(0) for tl in tiles_of(load_rgb(p, 256 * T), T)]
                ed = [torch.from_numpy(t) for t in anomaly_inputs(to_square(cv2.imread(str(p))), T)]
                diffs.append(max(float((a - b).abs().max()) for a, b in zip(tr, ed)))
                st = max(float(model(t.to(dev)).max()) for t in tr)
                se = max(float(model(t.to(dev)).max()) for t in ed)
                if "/test/" in p.as_posix():
                    s_train.append(st); s_edge.append(se); y_test.append(y)
        s_train, s_edge = np.array(s_train), np.array(s_edge)
        report[name] = {
            "images": len(items), "tiles": T,
            "input_max_abs_diff_max": round(max(diffs), 4), "input_max_abs_diff_median": round(float(np.median(diffs)), 4),
            "score_correlation": round(float(np.corrcoef(s_train, s_edge)[0, 1]), 5),
            "score_max_rel_diff": round(float(np.max(np.abs(s_edge - s_train) / np.maximum(np.abs(s_train), 1e-6))), 4),
            "auroc_training_preproc": round(auroc(y_test, s_train), 4),
            "auroc_edge_preproc": round(auroc(y_test, s_edge), 4),
        }
        r = report[name]
        r["pass"] = abs(r["auroc_training_preproc"] - r["auroc_edge_preproc"]) <= 0.005 and r["score_correlation"] >= 0.999
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
