"""W3-D - evaluate a trained EfficientAD-S model (protocol: ADR-019).

    ~/bharatai-build/train-venv/bin/python training/vision/evaluate_efficientad.py --name mvtec_wood \\
        --val ~/bharatai-build/datasets/anomaly/wood/val/good \\
        --test ~/bharatai-build/data/mvtec/wood/test --masks ~/bharatai-build/data/mvtec/wood/ground_truth
    ... --name own_wood --val .../anomaly_own/wood/val/good --test .../anomaly_own/wood/test \\
        --boxes .../anomaly_own/wood/test_boxes.json

Image score = max of the anomaly map (EfficientAD). Metrics on TEST:
  image AUROC (good=0 vs defect=1); pixel AUROC if --masks (MVTec); box hit rate if --boxes (own:
  share of labelled defect boxes whose max map value reaches the threshold).
Operating threshold = 99th percentile of VALIDATION good image scores (decided on val only);
reported on test as false-alarm rate (good flagged) and recall (defects flagged), also per type.
Writes ~/bharatai-build/models/efficientad/<name>/{eval.json, test_scores.csv} and a heatmap
overview to training/vision/reports/eval_<name>.jpg.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from efficientad_model import EfficientADS, preprocess  # noqa: E402

MODELS = Path.home() / "bharatai-build/models/efficientad"
REPORTS = Path(__file__).resolve().parent / "reports"
EXTS = {".png", ".jpg", ".jpeg"}
IMG = 256


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in EXTS)


def amap(model: EfficientADS, path: Path, dev: torch.device, tiles: int = 1) -> np.ndarray:
    """Anomaly map at tiles*IMG px. tiles > 1: the image is resized to tiles*IMG, scored tile by tile
    (each tile exactly like a training tile) and the tile maps are stitched (P-57)."""
    rgb = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy())
    x = preprocess(rgb, IMG * tiles).to(dev)
    out = np.zeros((IMG * tiles, IMG * tiles), np.float32)
    with torch.no_grad():
        for r in range(tiles):
            for c in range(tiles):
                tl = x[:, :, r * IMG:(r + 1) * IMG, c * IMG:(c + 1) * IMG]
                out[r * IMG:(r + 1) * IMG, c * IMG:(c + 1) * IMG] = model(tl)[0, 0].cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--boxes", type=Path, default=None)
    ap.add_argument("--percentile", type=float, default=99.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientADS(IMG)
    model.load_state_dict(torch.load(MODELS / args.name / "model.pt", map_location="cpu", weights_only=True))
    model.eval().to(dev)
    cfg = json.loads((MODELS / args.name / "config.json").read_text())
    T = int(cfg.get("tiles", 1))
    S = IMG * T  # map / evaluation resolution

    val_scores = [float(amap(model, p, dev, T).max()) for p in images(args.val)]
    thr = float(np.percentile(val_scores, args.percentile))

    boxes = json.loads(args.boxes.read_text()) if args.boxes else {}
    rows, y, s, pix_y, pix_s, overview = [], [], [], [], [], []
    hits = total_boxes = 0
    for tdir in sorted(d for d in args.test.iterdir() if d.is_dir()):
        label = tdir.name
        for p in images(tdir):
            m = amap(model, p, dev, T)
            score = float(m.max())
            is_def = int(label != "good")
            y.append(is_def); s.append(score)
            rows.append([f"{label}/{p.name}", label, is_def, f"{score:.5f}", int(score >= thr)])
            if args.masks is not None:
                if is_def:
                    mp = args.masks / label / f"{p.stem}_mask.png"
                    mask = cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (S, S),
                                      interpolation=cv2.INTER_NEAREST) > 127 if mp.exists() else np.zeros((S, S), bool)
                else:
                    mask = np.zeros((S, S), bool)
                pix_y.append(mask.ravel()); pix_s.append(m.ravel())
            key = f"test/{label}/{p.name}"
            for x1, y1, x2, y2 in (boxes.get(key, {}).get("boxes_xyxy_norm", []) if is_def else []):
                total_boxes += 1
                xa, ya = int(x1 * S), int(y1 * S)
                xb, yb = max(xa + 1, int(np.ceil(x2 * S))), max(ya + 1, int(np.ceil(y2 * S)))
                hits += int(m[ya:yb, xa:xb].max() >= thr)
            if len([o for o in overview if o[1] == label]) < 2:
                overview.append((p, label, m, score))

    y_arr, s_arr = np.array(y), np.array(s)
    res = {
        "name": args.name, "test_dir": str(args.test), "tiles": T,
        "n_val_good": len(val_scores), "n_test_good": int((y_arr == 0).sum()), "n_test_defect": int(y_arr.sum()),
        "image_auroc": round(float(roc_auc_score(y_arr, s_arr)), 4) if 0 < y_arr.sum() < len(y_arr) else None,
        "threshold": round(thr, 5), "threshold_rule": f"{args.percentile}th percentile of val good image scores",
        "false_alarm_rate": round(float((s_arr[y_arr == 0] >= thr).mean()), 4),
        "recall": round(float((s_arr[y_arr == 1] >= thr).mean()), 4),
    }
    per_type = {}
    for r in rows:
        if r[2]:
            per_type.setdefault(r[1], []).append(r[4])
    res["recall_per_type"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_type.items())}
    if pix_y:
        res["pixel_auroc"] = round(float(roc_auc_score(np.concatenate(pix_y), np.concatenate(pix_s))), 4)
    if total_boxes:
        res["box_hit_rate"] = f"{hits}/{total_boxes}"

    out = MODELS / args.name
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    with (out / "test_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "label", "is_defect", "score", "flagged"]); w.writerows(rows)

    # overview: 2 per test label, image | heatmap
    REPORTS.mkdir(parents=True, exist_ok=True)
    tiles = []
    for p, label, m, score in overview:
        img = cv2.resize(cv2.imread(str(p)), (IMG, IMG))
        m = cv2.resize(m, (IMG, IMG), interpolation=cv2.INTER_AREA) if m.shape[0] != IMG else m
        heat = cv2.applyColorMap(np.clip(m / max(thr * 2, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        pair = np.hstack([img, cv2.addWeighted(img, 0.5, heat, 0.5, 0)])
        flag = "FLAG" if score >= thr else "ok"
        cv2.putText(pair, f"{label} {score:.2f} {flag}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(pair)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(REPORTS / f"eval_{args.name}.jpg"),
                np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
