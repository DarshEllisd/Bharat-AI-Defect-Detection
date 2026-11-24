"""W2 task E - can EfficientAD-S run int8 on the phone's Hexagon NPU (SSG2115P stand-in, ADR-012)?

Exports, into spike/artifacts/npu/models_ead/:
  efficientad_int8_SSG2115P.pte   QNN int8, calibrated on real MVTec wood images
  efficientad_fp32_xnnpack.pte    CPU fallback (XNNPACK) for comparison
  ead.input.raw                   one MVTec wood test image (hole defect), preprocessed, float32
  *.ref.raw                       CPU reference outputs (anomaly map 1x1x256x256, float32)
  manifest.txt                    for spike/phone/07_npu_probe.sh (QNN model only)
  ead_report.json                 compile status, delegation, int8-vs-fp32 error

Weights are untrained (W3 trains them): this checks ops, compilation, NPU execution and latency,
not detection accuracy. Run inside the et-venv with the QNN SDK environment (export_v73_ead.sh).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "training" / "vision"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from efficientad_model import EfficientADS, preprocess  # noqa: E402
from export_qnn_probe import compile_spec  # noqa: E402

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset  # noqa: E402
from executorch.backends.qualcomm.utils.utils import to_edge_transform_and_lower_to_qnn  # noqa: E402

SOC = "SSG2115P"
MVTEC_WOOD = Path.home() / "bharatai-build/data/mvtec/wood"


def load(path: Path) -> torch.Tensor:
    return preprocess(torch.from_numpy(np.array(Image.open(path).convert("RGB"))))


def quantize_with_images(model: torch.nn.Module, example: tuple, calib: list[torch.Tensor]) -> torch.nn.Module:
    """PT2E int8 with the QNN quantizer, calibrated on real images (not random noise)."""
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
    from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=getattr(QcomChipset, SOC))
    gm = torch.export.export(model, example, strict=False).module()
    gm = prepare_pt2e(gm, quantizer)
    with torch.no_grad():
        for x in calib:
            gm(x)
    return convert_pt2e(gm)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "spike/artifacts/npu/models_ead")
    parser.add_argument("--calib", type=int, default=32, help="number of MVTec wood train images")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    report: dict = {"soc": SOC, "weights": "untrained (feasibility only)"}

    model = EfficientADS(256).eval()
    train_imgs = sorted((MVTEC_WOOD / "train/good").glob("*.png"))
    random.Random(0).shuffle(train_imgs)
    calib = [load(p) for p in train_imgs[: args.calib]]
    test_img = sorted((MVTEC_WOOD / "test/hole").glob("*.png"))[0]
    x = load(test_img)
    example = (x,)
    x.numpy().astype(np.float32).tofile(args.out / "ead.input.raw")
    report["input"] = f"mvtec wood/test/hole/{test_img.name}"

    with torch.no_grad():
        fp32_out = model(x)
    report["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    # --- CPU fallback: fp32 XNNPACK ---
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower

        ep = torch.export.export(model, example, strict=False)
        prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
        (args.out / "efficientad_fp32_xnnpack.pte").write_bytes(prog.buffer)
        fp32_out.numpy().astype(np.float32).tofile(args.out / "efficientad_fp32_xnnpack.ref.raw")
        report["xnnpack_fp32"] = "ok"
        print("[ok]     efficientad_fp32_xnnpack")
    except Exception as e:
        report["xnnpack_fp32"] = f"{type(e).__name__}: {e}"
        print(f"[FAILED] xnnpack fp32: {report['xnnpack_fp32']}")

    # --- NPU: int8 QNN for the 2 MB-VTCM V73 stand-in ---
    name = f"efficientad_int8_{SOC}"
    try:
        qmodel = quantize_with_images(model, example, calib)
        with torch.no_grad():
            int8_ref = qmodel(x)
        report["int8_sim_vs_fp32_cosine"] = round(cosine(int8_ref, fp32_out), 4)
        edge = to_edge_transform_and_lower_to_qnn(qmodel, example, compile_spec(SOC, fp16=False))
        try:  # how much of the graph landed on the NPU
            from executorch.devtools.backend_debug import get_delegation_info

            info = get_delegation_info(edge.exported_program().graph_module)
            report["delegation"] = {"delegated_nodes": info.num_delegated_nodes,
                                    "non_delegated_nodes": info.num_non_delegated_nodes}
        except Exception as e:
            report["delegation"] = f"unavailable: {e}"
        prog = edge.to_executorch()
        (args.out / f"{name}.pte").write_bytes(prog.buffer)
        int8_ref.numpy().astype(np.float32).tofile(args.out / f"{name}.ref.raw")
        (args.out / "manifest.txt").write_text(f"{name} {name}.pte ead.input.raw {name}.ref.raw v73 {SOC} ead_int8\n")
        report["qnn_int8"] = "ok"
        print(f"[ok]     {name}  (int8 sim vs fp32 cosine {report['int8_sim_vs_fp32_cosine']})")
    except Exception as e:
        report["qnn_int8"] = f"{type(e).__name__}: {e}"
        print(f"[FAILED] {name}: {report['qnn_int8']}")

    (args.out / "ead_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
