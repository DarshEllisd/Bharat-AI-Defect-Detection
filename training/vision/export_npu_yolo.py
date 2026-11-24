"""W3-E - int8 export of the trained YOLO11n for the phone's NPU, with accuracy check.

    bash training/vision/runs/export_npu.sh yolo [--quant 16a8w] [--calib 128]

Runs in the et-venv (CPU torch + ExecuTorch + QNN SDK env):
  1. best.pt -> fused DetectionModel in export mode (one tensor out: 1 x (4+48) x 2100 at 320 px).
  2. PT2E quantisation with the QNN quantizer (8a8w default, 16a8w fallback; per-channel conv
     weights), calibrated on --calib real TRAINING images, letterboxed exactly like inference.
  3. Accuracy: Ultralytics' own validator runs the fp32 export-mode graph and the quantised graph
     (CPU simulation) on val, MVTec test and own test - same NMS/mAP code for both - by
     replacing the loaded model's forward. Acceptance (ADR-019): mAP50 drop <= 5 points.
  4. Lowering: QNN .pte for SSG2115P + fp32 XNNPACK .pte; phone check input (one own test image)
     with the simulated quantised output as reference + manifest.txt for 07_npu_probe.sh.
Output: ~/bharatai-build/models/npu/yolo11n_combined_<quant>/
"""
import argparse
import json
import os

# Import matplotlib (used inside Ultralytics' validator) BEFORE ExecuTorch/QNN: imported after the QNN
# SDK's native libraries are loaded, matplotlib's font module segfaulted (P-61).
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402,F401
import matplotlib.pyplot  # noqa: E402,F401
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "spike/laptop/npu"))
from export_qnn_probe import compile_spec  # noqa: E402

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype  # noqa: E402
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset, QnnExecuTorchBackendType  # noqa: E402
from executorch.backends.qualcomm.utils.utils import to_edge_transform_and_lower_to_qnn  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402

settings.update({"sync": False})  # P-47
SOC = "SSG2115P"
B = Path.home() / "bharatai-build"
RUN = B / "models/yolo/yolo11n_combined"
DATA = B / "datasets/yolo_combined/data.yaml"
IMGSZ = 320
os.chdir(B)


def export_mode(model: torch.nn.Module) -> torch.nn.Module:
    for m in model.modules():
        # in-place SiLU exports as aten.silu_, which the QNN quantizer doesn't annotate (left
        # unquantised / off the NPU) - use the out-of-place op (P-61)
        if isinstance(m, torch.nn.SiLU):
            m.inplace = False
        if m.__class__.__name__ == "Detect":
            m.export, m.format, m.dynamic = True, "executorch", False
    with torch.no_grad():  # caches anchors/strides for 320 px (as Ultralytics' exporter does)
        model(torch.zeros(1, 3, IMGSZ, IMGSZ))
    return model


class NormalisedBoxes(torch.nn.Module):
    """YOLO's output mixes box coordinates (0-320 px) and class scores (0-1) in ONE tensor; with one shared
    8-bit range the scores round to 0 (mAP 0, P-62). Boxes are divided by the input size inside the graph
    (as Ultralytics' int8 TFLite export does), so every channel is in 0-1. Callers multiply rows 0-3 by IMGSZ."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return torch.cat([y[:, :4] / IMGSZ, y[:, 4:]], dim=1)


def letterboxed(path: str) -> torch.Tensor:
    img = LetterBox((IMGSZ, IMGSZ), auto=False)(image=cv2.imread(path))
    return torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def run_val(graph, tag: str, splits: dict) -> dict:
    """Ultralytics validator with the model's forward replaced by `graph` (batch-1 static)."""
    y = YOLO(str(RUN / "weights/best.pt"))
    y.model.fuse()

    def fwd(x, *a, **k):
        # The validator's rectangular batches add a centred border (320 px image -> 352x352 frame), but the
        # exported graph is fixed at 320x320: crop the centred 320 px region, then shift boxes back (P-61).
        h, w = x.shape[2:]
        dy, dx = (h - IMGSZ) // 2, (w - IMGSZ) // 2
        xc = x[:, :, dy:dy + IMGSZ, dx:dx + IMGSZ].float()
        out = torch.cat([graph(xc[i:i + 1]) for i in range(x.shape[0])])
        out[:, :4] *= IMGSZ  # graphs output normalised boxes (P-62)
        out[:, 0] += dx
        out[:, 1] += dy
        return out
    y.model.forward = fwd
    out = {}
    for split, data in splits.items():
        # workers=0: forking data-loader workers from a process with the QNN SDK loaded segfaulted (P-61)
        m = y.val(data=str(data), split="test" if split != "val" else "val", imgsz=IMGSZ, batch=8, workers=0,
                  conf=0.001, iou=0.6, device="cpu", plots=False, verbose=False,
                  project=str(RUN), name=f"npu_eval_{tag}_{split}", exist_ok=True)
        out[split] = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["8a8w", "16a8w"], default="8a8w")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = B / "models/npu" / f"yolo11n_combined_{args.quant}"
    out.mkdir(parents=True, exist_ok=True)

    model = NormalisedBoxes(export_mode(YOLO(str(RUN / "weights/best.pt")).model.fuse().eval().float())).eval()
    cfg = yaml.safe_load(DATA.read_text())
    train = [str(p) for d in cfg["train"] for p in sorted(Path(d).iterdir())]
    random.Random(0).shuffle(train)
    calib = train[: args.calib]
    example = (letterboxed(calib[0]),)
    report = {"soc": SOC, "quant": args.quant, "calibration": f"{len(calib)} training images (seed 0)",
              "output_shape": list(model(example[0]).shape),
              "output_format": "rows 0-3 = cx, cy, w, h / 320 (normalised, P-62); rows 4.. = class scores"}

    quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=getattr(QcomChipset, SOC))
    quantizer.set_default_quant_config(
        quant_dtype=QuantDtype.use_16a8w if args.quant == "16a8w" else QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True)
    gm = prepare_pt2e(torch.export.export(model, example, strict=False).module(), quantizer)
    with torch.no_grad():
        for p in calib:
            gm(letterboxed(p))
    qgm = convert_pt2e(gm)

    if not args.skip_eval:
        splits = {"val": DATA, "test_mvtec": DATA.parent / "data_test_mvtec.yaml",
                  "test_own": DATA.parent / "data_test_own.yaml"}
        with torch.no_grad():
            report["fp32_export_graph"] = run_val(model, "fp32", splits)
            report[args.quant] = run_val(qgm, args.quant, splits)
        report["delta_mAP50_points"] = {s: round(100 * (report[args.quant][s]["mAP50"] - report["fp32_export_graph"][s]["mAP50"]), 2)
                                        for s in splits}
        report["accept_mAP50_drop_le_5"] = all(v >= -5 for v in report["delta_mAP50_points"].values())
        print(json.dumps(report, indent=2))

    # phone check input: first own test image with a defect label
    own_test = Path([d for d in cfg["test"] if "/own/" in d][0])
    def has_box(p: Path) -> bool:
        lbl = own_test.parent.parent / "labels/test" / f"{p.stem}.txt"
        return lbl.exists() and bool(lbl.read_text().strip())
    pick = next(p for p in sorted(own_test.iterdir()) if has_box(p))
    x = letterboxed(str(pick))
    x.numpy().astype(np.float32).tofile(out / "own.input.raw")
    with torch.no_grad():
        qgm(x).numpy().astype(np.float32).tofile(out / "own.ref.raw")
        model(x).numpy().astype(np.float32).tofile(out / "own.fp32.raw")
    report["phone_input"] = pick.name
    pte = f"yolo11n_combined_{args.quant}_{SOC}.pte"
    try:
        edge = to_edge_transform_and_lower_to_qnn(qgm, example, compile_spec(SOC, fp16=False))
        (out / pte).write_bytes(edge.to_executorch().buffer)
        (out / "manifest.txt").write_text(f"yolo11n_combined_{args.quant} {pte} own.input.raw own.ref.raw v73 {SOC} yolo_{args.quant}\n")
        report["qnn"] = f"ok ({(out / pte).stat().st_size / 1e6:.1f} MB)"
    except Exception as e:  # noqa: BLE001
        report["qnn"] = f"{type(e).__name__}: {e}"
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        ep = torch.export.export(model, example, strict=False)
        (out / "yolo11n_combined_fp32_xnnpack.pte").write_bytes(
            to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch().buffer)
        report["xnnpack_fp32"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["xnnpack_fp32"] = f"{type(e).__name__}: {e}"
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("fp32_export_graph", args.quant)}, indent=2))


if __name__ == "__main__":
    main()
