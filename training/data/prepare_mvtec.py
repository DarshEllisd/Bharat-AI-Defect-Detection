"""W2 - turn raw MVTec AD into the project's YOLO and anomaly datasets (ADR-017).

    source ~/bharatai-build/et-venv/bin/activate
    python training/data/prepare_mvtec.py

Inputs : $DATA/mvtec/<category>/{train/good, test/<type>, ground_truth/<type>/*_mask.png}
Outputs (on the WSL filesystem, symlinks to the raw images - no copies):
  $DATASETS/yolo_mvtec/{images,labels}/{train,val,test}/ + data.yaml
  $DATASETS/anomaly/<category>/{train/good, test/<type>, ground_truth/<type>}
Repo outputs (small, tracked): training/data/reports/
  data_stats.md / data_stats.json   counts per category, class and split
  qa_boxes_<category>.jpg           boxes drawn on sample images, for a visual check

Rules (ADR-017):
  - classes are "<category>_<defect>"; fabric_* classes are reserved for our own fabric photos
  - "combined" defects are excluded from YOLO (kept for anomaly evaluation)
  - one box per connected component of a defect mask; specks below MIN_AREA_FRAC are dropped
  - test defects -> YOLO 60/20/20 train/val/test, stratified per category+defect, fixed seed
  - test/good -> half final evaluation (YOLO test negatives + anomaly test), half YOLO negatives
  - up to TRAIN_GOOD_NEGATIVES train/good images per category also serve as YOLO background images
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CATEGORIES = ["cable", "carpet", "grid", "leather", "metal_nut", "screw", "tile", "transistor", "wood"]
# Own-photo classes with no MVTec counterpart, reserved now so class ids stay stable (ADR-016/017)
RESERVED_CLASSES = ["fabric_color", "fabric_cut", "fabric_hole", "fabric_thread"]
EXCLUDED_TYPES = {"good", "combined"}
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
MIN_AREA_FRAC = 5e-5          # ~52 px on a 1024x1024 image
TRAIN_GOOD_NEGATIVES = 30
SEED = 42

HOME = Path.home()
REPO_ROOT = Path(__file__).resolve().parents[2]


def mask_to_boxes(mask_path: Path) -> tuple[list[tuple[float, float, float, float]], int]:
    """Normalised YOLO boxes (cx, cy, w, h), one per connected component; also returns #specks dropped."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), connectivity=8)
    boxes, dropped = [], 0
    for i in range(1, n):  # 0 = background
        x, y, bw, bh, area = stats[i]
        if area < MIN_AREA_FRAC * h * w:
            dropped += 1
            continue
        boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes, dropped


def split_list(items: list, fractions=SPLIT_FRACTIONS) -> dict[str, list]:
    """Deterministic shuffle + split; small groups still get >= 1 val and test item when possible."""
    items = sorted(items)
    random.Random(SEED).shuffle(items)
    n = len(items)
    n_val = max(1, round(n * fractions[1])) if n >= 3 else 0
    n_test = max(1, round(n * fractions[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def qa_grid(samples: list[tuple[Path, list]], out: Path, tile: int = 256, cols: int = 4) -> None:
    """Draw boxes on up to 12 sample images and tile them into one JPEG."""
    tiles = []
    for img_path, boxes in samples[:12]:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for cx, cy, bw, bh in boxes:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, w // 200))
        img = cv2.resize(img, (tile, tile))
        cv2.putText(img, img_path.parent.name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec", type=Path, default=HOME / "bharatai-build/data/mvtec")
    parser.add_argument("--datasets", type=Path, default=HOME / "bharatai-build/datasets")
    parser.add_argument("--reports", type=Path, default=REPO_ROOT / "training/data/reports")
    args = parser.parse_args()

    yolo_root = args.datasets / "yolo_mvtec"
    anomaly_root = args.datasets / "anomaly"
    args.reports.mkdir(parents=True, exist_ok=True)

    # --- class list (stable order: sorted MVTec classes, then reserved own-photo classes) ---
    mvtec_classes = sorted(
        f"{c}_{t.name}" for c in CATEGORIES for t in (args.mvtec / c / "test").iterdir()
        if t.is_dir() and t.name not in EXCLUDED_TYPES
    )
    names = mvtec_classes + [r for r in RESERVED_CLASSES if r not in mvtec_classes]
    class_id = {n: i for i, n in enumerate(names)}

    stats = {"categories": {}, "classes": names, "boxes_per_class_split": defaultdict(lambda: defaultdict(int)),
             "images_per_split": defaultdict(int), "dropped_specks": 0, "missing_masks": []}

    for cat in CATEGORIES:
        root = args.mvtec / cat
        cat_stats = {"train_good": 0, "test_good": 0, "test_defects": {}}
        qa_samples = []

        # anomaly: all train/good
        train_good = sorted((root / "train/good").glob("*.png"))
        cat_stats["train_good"] = len(train_good)
        for p in train_good:
            link(p, anomaly_root / cat / "train/good" / p.name)

        # test/good: half final eval, half YOLO negatives (train/val 75/25)
        test_good = sorted((root / "test/good").glob("*.png"))
        cat_stats["test_good"] = len(test_good)
        shuffled = test_good[:]
        random.Random(SEED).shuffle(shuffled)
        half = len(shuffled) // 2
        eval_good, neg_good = shuffled[:half], shuffled[half:]
        n_neg_train = round(len(neg_good) * 0.75)
        good_splits = {"test": eval_good, "train": neg_good[:n_neg_train], "val": neg_good[n_neg_train:]}
        for p in eval_good:
            link(p, anomaly_root / cat / "test/good" / p.name)
        # plus some train/good as extra YOLO background images
        bg = train_good[:]
        random.Random(SEED + 1).shuffle(bg)
        good_splits["train"] = good_splits["train"] + bg[:TRAIN_GOOD_NEGATIVES]
        for split, imgs in good_splits.items():
            for p in imgs:
                stem = f"{cat}__good__{p.parent.parent.name}_{p.stem}"
                link(p, yolo_root / "images" / split / f"{stem}.png")
                (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
                (yolo_root / "labels" / split / f"{stem}.txt").write_text("")
                stats["images_per_split"][split] += 1

        # defects
        for tdir in sorted(d for d in (root / "test").iterdir() if d.is_dir() and d.name != "good"):
            imgs = sorted(tdir.glob("*.png"))
            cat_stats["test_defects"][tdir.name] = len(imgs)
            if tdir.name == "combined":  # anomaly evaluation only
                for p in imgs:
                    link(p, anomaly_root / cat / "test/combined" / p.name)
                    link(root / "ground_truth/combined" / f"{p.stem}_mask.png",
                         anomaly_root / cat / "ground_truth/combined" / f"{p.stem}_mask.png")
                continue
            cls = f"{cat}_{tdir.name}"
            for split, items in split_list(imgs).items():
                for p in items:
                    mask = root / "ground_truth" / tdir.name / f"{p.stem}_mask.png"
                    if not mask.exists():
                        stats["missing_masks"].append(str(mask.relative_to(args.mvtec)))
                        continue
                    boxes, dropped = mask_to_boxes(mask)
                    stats["dropped_specks"] += dropped
                    stem = f"{cat}__{tdir.name}__{p.stem}"
                    link(p, yolo_root / "images" / split / f"{stem}.png")
                    (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
                    (yolo_root / "labels" / split / f"{stem}.txt").write_text(
                        "".join(f"{class_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes))
                    stats["images_per_split"][split] += 1
                    stats["boxes_per_class_split"][cls][split] += len(boxes)
                    if split == "test":  # anomaly test = the same held-out defects as YOLO test
                        link(p, anomaly_root / cat / "test" / tdir.name / p.name)
                        link(mask, anomaly_root / cat / "ground_truth" / tdir.name / mask.name)
                    if sum(1 for s, _ in qa_samples if s.parent == p.parent) < 2:  # 2 per defect type
                        qa_samples.append((p, boxes))
        stats["categories"][cat] = cat_stats
        qa_grid(qa_samples, args.reports / f"qa_boxes_{cat}.jpg")
        print(f"[ok] {cat:11s} train_good={cat_stats['train_good']:4d} test_good={cat_stats['test_good']:3d} "
              f"defect types={len(cat_stats['test_defects'])}")

    # --- data.yaml ---
    yaml_lines = [f"path: {yolo_root}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (yolo_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    # --- reports ---
    stats["boxes_per_class_split"] = {k: dict(v) for k, v in stats["boxes_per_class_split"].items()}
    stats["images_per_split"] = dict(stats["images_per_split"])
    (args.reports / "data_stats.json").write_text(json.dumps(stats, indent=2))

    md = ["# MVTec AD - prepared dataset statistics", "",
          "Generated by `training/data/prepare_mvtec.py` (ADR-017). Images are symlinks to the raw dataset.", "",
          "## Per category", "", "| Category | train/good | test/good | Defect types (test images) |", "|---|---|---|---|"]
    for cat, s in stats["categories"].items():
        defects = ", ".join(f"{t} {n}" for t, n in s["test_defects"].items())
        md.append(f"| {cat} | {s['train_good']} | {s['test_good']} | {defects} |")
    md += ["", "## YOLO images per split (incl. negatives)", "", "| Split | Images |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(stats["images_per_split"].items())]
    md += ["", f"## YOLO boxes per class ({len(names)} classes; reserved for own photos: "
           f"{', '.join(RESERVED_CLASSES)})", "", "| Class | train | val | test |", "|---|---|---|---|"]
    for cls in names:
        b = stats["boxes_per_class_split"].get(cls, {})
        md.append(f"| {cls} | {b.get('train', 0)} | {b.get('val', 0)} | {b.get('test', 0)} |")
    md += ["", f"Mask specks dropped (< {MIN_AREA_FRAC} of image area): {stats['dropped_specks']}",
           f"Missing masks: {len(stats['missing_masks'])}"]
    (args.reports / "data_stats.md").write_text("\n".join(md) + "\n")
    print(f"classes={len(names)} images={stats['images_per_split']} dropped_specks={stats['dropped_specks']} "
          f"missing_masks={len(stats['missing_masks'])}")
    print(f"data.yaml -> {yolo_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
