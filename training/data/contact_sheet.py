"""Make a small contact sheet of captured photos (a few per product/label) for a quick quality check.

    python training/data/contact_sheet.py --out training/data/reports/own_contact_sheet.jpg

Draws the capture tool's square region of interest (from the JSON sidecar) on each thumbnail.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path.home() / "bharatai-build/data/own/raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--tile", type=int, default=240)
    parser.add_argument("--crop", action="store_true", help="show only the square ROI (what the models see)")
    parser.add_argument("--labels", default="", help="comma list of product/label to include (default: all)")
    parser.add_argument("--contains", default="", help="only files whose name contains this (e.g. a date 20261121)")
    args = parser.parse_args()
    only = {s.strip() for s in args.labels.split(",") if s.strip()}

    tiles = []
    for label_dir in sorted(p for p in args.raw.glob("*/*") if p.is_dir()):
        jpgs = sorted(p for p in label_dir.glob("*.jpg") if args.contains in p.name)
        if not jpgs or (only and f"{label_dir.parent.name}/{label_dir.name}" not in only):
            continue
        step = max(1, len(jpgs) // args.per_label)
        for jpg in jpgs[::step][: args.per_label]:
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            side = jpg.with_suffix(".json")
            meta = json.loads(side.read_text()) if side.exists() else {}
            h0, w0 = img.shape[:2]
            if meta.get("roi_xywh"):
                x, y, w, h = meta["roi_xywh"]
                if args.crop:
                    img = img[y:y + h, x:x + w]
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 160, 255), max(3, img.shape[1] // 60))
            h0, w0 = img.shape[:2]
            s = args.tile / max(h0, w0)
            img = cv2.resize(img, (int(w0 * s), int(h0 * s)))
            canvas = np.zeros((args.tile, args.tile, 3), np.uint8)
            canvas[: img.shape[0], : img.shape[1]] = img
            when = (meta.get("captured_at") or "")[11:19]  # HH:MM:SS - helps see where one defect ends
            tag = f"{label_dir.name} i{meta.get('instance_id') or '-'} {when}"
            cv2.putText(canvas, tag, (4, args.tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            tiles.append(canvas)
    if not tiles:
        raise SystemExit("no photos found")
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
