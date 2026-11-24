"""W2 task D/F - merge our own wood/fabric photos into the training datasets (ADR-016/017).

    source ~/bharatai-build/et-venv/bin/activate
    python training/data/merge_own_data.py            # dry run: show defect groups and splits
    python training/data/merge_own_data.py --apply    # write datasets + reports

What it does
  1. Defect "instances" = one physical defect. The capture tool's instance id is refined by time:
     a gap > GROUP_GAP_S seconds between consecutive shots starts a new group (the user didn't
     always press "New defect"). Good photos are grouped into bursts (gap > BURST_GAP_S).
  2. Splits are by GROUP, never by photo, so near-identical shots of one defect or one burst can't
     sit on both sides (ADR-017):
       good    -> ~70% train / 10% val / 20% test   (anomaly model; train also = YOLO background)
       defects -> classes with >= 2 groups: 1 group held out as the real-object TEST set;
                  classes with 1 group: train only (YOLO needs them; the anomaly model never trains
                  on defects, so every real defect is still used in the anomaly test set).
  3. Each photo is cropped to the capture tool's square ROI (what the models see), resized to
     IMG_SIZE px, and its Label Studio boxes are converted (clipped to the ROI).
Outputs (WSL filesystem):
  $DATASETS/own/{images,labels}/{train,val,test}          YOLO, class ids = yolo_mvtec/data.yaml
  $DATASETS/yolo_combined/data.yaml                        MVTec + own, train/val/test lists
  $DATASETS/anomaly_own/<product>/{train,val}/good, test/{good,<defect>}, test_boxes.json
Repo outputs: training/data/reports/own_splits.csv, own_stats.md, qa_own_<product>.jpg
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HOME = Path.home()
RAW = HOME / "bharatai-build/data/own/raw"
LABELS = HOME / "bharatai-build/data/own/labels"
DATASETS = HOME / "bharatai-build/datasets"
REPORTS = Path(__file__).resolve().parent / "reports"
GROUP_GAP_S = 20      # > 20 s between shots of one label/instance -> a different physical defect
BURST_GAP_S = 4       # good photos: > 4 s -> a new burst / new position
IMG_SIZE = 1024       # stored crop size (training resizes further to 256-320)
MIN_KEEP = 0.3        # drop a box if < 30% of it lies inside the ROI
SEED = 42
PRODUCTS = ["wood", "fabric"]


def ts(meta: dict) -> datetime:
    return datetime.fromisoformat(meta["captured_at"])


def load_photos(product: str) -> list[dict]:
    photos = []
    for jpg in sorted((RAW / product).glob("*/*.jpg")):
        meta = json.loads(jpg.with_suffix(".json").read_text())
        photos.append({"path": jpg, "file": jpg.name, "label": meta["label"], "meta": meta, "t": ts(meta)})
    return photos


def assign_groups(photos: list[dict]) -> None:
    """Group ids: good -> bursts; defects -> (date, instance) refined by time gaps."""
    by_key = defaultdict(list)
    for p in photos:
        key = ("good",) if p["label"] == "good" else (p["label"], p["t"].date().isoformat(), p["meta"].get("instance_id"))
        by_key[key].append(p)
    for key, items in by_key.items():
        items.sort(key=lambda p: p["t"])
        gap = BURST_GAP_S if key[0] == "good" else GROUP_GAP_S
        g, prev = 0, None
        for p in items:
            if prev is not None and (p["t"] - prev).total_seconds() > gap:
                g += 1
            prev = p["t"]
            p["group"] = f"{p['label']}-burst{g:02d}" if key[0] == "good" else \
                f"{p['label']}-{key[1].replace('-', '')}-i{key[2]}-g{g}"


def assign_splits(photos: list[dict], fixed: dict[str, str] | None = None) -> None:
    """Split by group. `fixed` (file -> split, from the previous own_splits.csv) keeps every photo that
    was already merged in its split, so adding a capture session never reshuffles existing data
    (results stay comparable, and no stale copies are left in old split folders). Only NEW groups are
    split, with the same rules applied to the new groups alone (ADR-017 update 2026-11-22)."""
    fixed = fixed or {}
    rng = random.Random(SEED)
    by_label = defaultdict(lambda: defaultdict(list))
    for p in photos:
        by_label[p["label"]][p["group"]].append(p)
    for label, groups in by_label.items():
        split_of = {}
        for g, items in groups.items():
            known = [fixed[p["file"]] for p in items if p["file"] in fixed]
            if known:
                split_of[g] = Counter(known).most_common(1)[0][0]
        names = sorted(g for g in groups if g not in split_of)
        rng.shuffle(names)
        if label == "good":
            n = len(names)
            if n:
                n_test, n_val = max(1, round(n * 0.2)), (max(1, round(n * 0.1)) if n >= 3 else 0)
                split_of.update({g: "test" for g in names[:n_test]})
                split_of.update({g: "val" for g in names[n_test:n_test + n_val]})
                split_of.update({g: "train" for g in names[n_test + n_val:]})
        else:
            split_of.update({g: "train" for g in names})
            # one physical defect held out as the real test set, if the class has >= 2 groups and
            # none of the already-merged groups is a test group
            if len(groups) >= 2 and "test" not in split_of.values() and names:
                split_of[names[0]] = "test"
        for g, items in groups.items():
            for p in items:
                p["split"] = split_of[g]


def crop_and_boxes(p: dict, boxes: list[dict]) -> tuple[np.ndarray, list[tuple], int]:
    x0, y0, s, _ = p["meta"]["roi_xywh"]
    img = cv2.imread(str(p["path"]))
    crop = cv2.resize(img[y0:y0 + s, x0:x0 + s], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    out, dropped = [], 0
    for b in boxes:
        bx1, by1, bx2, by2 = b["x"] - x0, b["y"] - y0, b["x"] + b["w"] - x0, b["y"] + b["h"] - y0
        cx1, cy1, cx2, cy2 = max(0, bx1), max(0, by1), min(s, bx2), min(s, by2)
        inside = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        if inside < MIN_KEEP * b["w"] * b["h"]:
            dropped += 1
            continue
        out.append((b["label"], (cx1 + cx2) / 2 / s, (cy1 + cy2) / 2 / s, (cx2 - cx1) / s, (cy2 - cy1) / s))
    return crop, out, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reshuffle", action="store_true",
                        help="ignore the previous own_splits.csv and re-split everything (not recommended)")
    args = parser.parse_args()

    names = [l.split(":", 1)[1].strip() for l in (DATASETS / "yolo_mvtec/data.yaml").read_text().splitlines()
             if l.startswith("  ") and ":" in l]
    class_id = {n: i for i, n in enumerate(names)}

    fixed = {}
    if not args.reshuffle and (REPORTS / "own_splits.csv").exists():
        with (REPORTS / "own_splits.csv").open() as fh:
            fixed = {r["file"]: r["split"] for r in csv.DictReader(fh)}
        print(f"keeping the splits of {len(fixed)} already-merged photos (own_splits.csv)")
    rows, stats = [], {}
    for product in PRODUCTS:
        photos = load_photos(product)
        assign_groups(photos)
        assign_splits(photos, fixed)
        boxes_by_file = {e["file"]: e["boxes"] for e in json.loads((LABELS / f"boxes_{product}.json").read_text())}

        print(f"\n== {product}: {len(photos)} photos")
        table = defaultdict(lambda: Counter())
        for p in photos:
            table[(p["label"], p["group"])][p["split"]] += 1
        for label in sorted({k[0] for k in table}):
            groups = sorted(k[1] for k in table if k[0] == label)
            if label == "good":
                per = Counter(s for (lb, g), c in table.items() if lb == label for s in c.elements())
                print(f"  good: {len(groups)} bursts -> {dict(per)}")
                continue
            for g in groups:
                c = table[(label, g)]
                print(f"  {g:36s} {sum(c.values()):3d} photos -> {next(iter(c))}")

        if not args.apply:
            continue
        stat = Counter()
        qa = []
        test_boxes = {}
        for p in photos:
            boxes = boxes_by_file.get(p["file"], []) if p["label"] != "good" else []
            if p["label"] != "good" and not boxes:
                print(f"  WARNING no boxes for {p['file']} - skipped")
                continue
            crop, yolo_boxes, dropped = crop_and_boxes(p, boxes)
            stat["boxes_dropped_outside_roi"] += dropped
            stem = f"{product}__{p['label']}__{p['path'].stem}"
            # YOLO dataset (all own photos; good = background images)
            img_dir = DATASETS / "own/images" / p["split"]
            lbl_dir = DATASETS / "own/labels" / p["split"]
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / f"{stem}.txt").write_text("".join(
                f"{class_id[f'{product}_{lb}']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for lb, cx, cy, w, h in yolo_boxes))
            stat[f"yolo_{p['split']}"] += 1
            stat[f"boxes_{p['split']}"] += len(yolo_boxes)
            # anomaly dataset: good split as-is; EVERY real defect goes to test (never trained on)
            if p["label"] == "good":
                adir = DATASETS / "anomaly_own" / product / ("test" if p["split"] == "test" else p["split"]) / "good"
            else:
                adir = DATASETS / "anomaly_own" / product / "test" / p["label"]
                test_boxes[f"test/{p['label']}/{stem}.jpg"] = {
                    "group": p["group"], "yolo_split": p["split"],
                    "boxes_xyxy_norm": [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in yolo_boxes]}
            adir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(adir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append([p["file"], product, p["label"], p["group"], p["split"], len(yolo_boxes)])
            if p["label"] != "good" and sum(1 for q in qa if q[1] == p["label"]) < 3:
                qa.append((crop, p["label"], yolo_boxes))
        (DATASETS / "anomaly_own" / product / "test_boxes.json").write_text(json.dumps(test_boxes, indent=1))
        stats[product] = dict(stat)

        # QA grid: 3 per defect label, boxes drawn on the crop
        tiles = []
        for crop, label, bxs in qa:
            t = crop.copy()
            for _, cx, cy, w, h in bxs:
                cv2.rectangle(t, (int((cx - w / 2) * IMG_SIZE), int((cy - h / 2) * IMG_SIZE)),
                              (int((cx + w / 2) * IMG_SIZE), int((cy + h / 2) * IMG_SIZE)), (0, 0, 255), 6)
            t = cv2.resize(t, (256, 256))
            cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros((256, 256, 3), np.uint8))
        cv2.imwrite(str(REPORTS / f"qa_own_{product}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])

    if not args.apply:
        print("\n(dry run - nothing written)")
        return

    # combined YOLO dataset: MVTec + own, same class ids
    m, o = DATASETS / "yolo_mvtec/images", DATASETS / "own/images"
    combined = DATASETS / "yolo_combined"
    combined.mkdir(parents=True, exist_ok=True)
    yaml = [f"path: {DATASETS}"]
    for split in ("train", "val", "test"):
        yaml.append(f"{split}:")
        yaml += [f"  - {m / split}", f"  - {o / split}"]
    yaml.append("names:")
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (combined / "data.yaml").write_text("\n".join(yaml) + "\n")

    with (REPORTS / "own_splits.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "product", "label", "group", "split", "boxes"])
        w.writerows(rows)
    md = ["# Own photos - merged dataset statistics", "",
          "Generated by `training/data/merge_own_data.py`. Per-photo split: `own_splits.csv`.", ""]
    for product, st in stats.items():
        md += [f"## {product}", "", "| | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(st.items())] + [""]
    (REPORTS / "own_stats.md").write_text("\n".join(md))
    print(f"\nwritten: {DATASETS / 'own'}, {DATASETS / 'anomaly_own'}, {combined / 'data.yaml'}")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
