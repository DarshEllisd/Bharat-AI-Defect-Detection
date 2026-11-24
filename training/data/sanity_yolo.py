"""W2 acceptance check: the prepared YOLO dataset loads and trains for one epoch (10% of images).

    source ~/bharatai-build/et-venv/bin/activate && cd ~/bharatai-build
    python /mnt/c/.../training/data/sanity_yolo.py

Not a real training run (W3); this only proves labels, paths and class ids are consistent.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

# default: the MVTec-only dataset; pass another data.yaml (e.g. yolo_combined) as the first argument
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "bharatai-build/datasets/yolo_mvtec/data.yaml"

model = YOLO("yolo11n.pt")
results = model.train(data=str(DATA), epochs=1, imgsz=320, fraction=0.1, batch=8, workers=2,
                      device="cpu", plots=False, project="/tmp/yolo_sanity", name="run", exist_ok=True,
                      verbose=False)
print("SANITY OK - box mAP50 after 1 epoch on 10% (not meaningful, just proves the pipeline):",
      round(float(results.box.map50), 4))
