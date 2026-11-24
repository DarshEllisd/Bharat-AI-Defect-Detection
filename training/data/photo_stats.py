"""List own photos with capture time and colour statistics of the square ROI, to tell apart
different materials (e.g. a printed vs plain fabric, a light vs dark board) before excluding any.

    python training/data/photo_stats.py --product wood --label good
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RAW = Path.home() / "bharatai-build/data/own/raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True)
    parser.add_argument("--label", default="*")
    args = parser.parse_args()
    print(f"{'file':52s} {'time':8s} inst  bright  sat  hue   R   G   B")
    for jpg in sorted((RAW / args.product).glob(f"{args.label}/*.jpg")):
        meta = json.loads(jpg.with_suffix(".json").read_text())
        img = cv2.imread(str(jpg), cv2.IMREAD_REDUCED_COLOR_4)   # quarter size is plenty for stats
        x, y, w, h = (v // 4 for v in meta["roi_xywh"])
        roi = img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        b, g, r = roi.reshape(-1, 3).mean(0)
        print(f"{jpg.parent.name + '/' + jpg.name:52s} {meta['captured_at'][11:19]} {str(meta.get('instance_id') or '-'):>4} "
              f"{hsv[..., 2].mean():6.1f} {hsv[..., 1].mean():5.1f} {np.median(hsv[..., 0]):4.0f} {r:4.0f}{g:4.0f}{b:4.0f}")


if __name__ == "__main__":
    main()
