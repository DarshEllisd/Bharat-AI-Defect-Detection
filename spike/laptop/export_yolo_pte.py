"""W1 spike - export a stock YOLO11n to ExecuTorch .pte (XNNPACK) at several input sizes.

This is a latency proxy for the vision pipeline; the real defect models come in W3.
Run inside the venv created by build_android_runners.sh so the .pte matches the runtime:

    source ~/bharatai-build/et-venv/bin/activate
    pip install "ultralytics>=8.3.220"
    python spike/laptop/export_yolo_pte.py --sizes 320 640
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "spike" / "artifacts" / "models" / "vision"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(args.weights).stem
    for size in args.sizes:
        model = YOLO(args.weights)
        # Ultralytics writes a "<stem>_executorch_model/" directory holding the .pte + metadata.yaml
        export_dir = Path(model.export(format="executorch", imgsz=size))
        pte_files = sorted(export_dir.glob("*.pte")) if export_dir.is_dir() else [export_dir]
        if not pte_files:
            raise SystemExit(f"No .pte produced in {export_dir}")
        dest = OUT_DIR / f"{stem}_{size}.pte"
        shutil.copy2(pte_files[0], dest)
        print(f"[ok] {dest}  ({dest.stat().st_size / 1e6:.1f} MB, input 1x3x{size}x{size})")
        # Rename the export dir so the next size doesn't overwrite it
        if export_dir.is_dir():
            target = export_dir.with_name(f"{export_dir.name}_{size}")
            shutil.rmtree(target, ignore_errors=True)
            export_dir.rename(target)


if __name__ == "__main__":
    main()
