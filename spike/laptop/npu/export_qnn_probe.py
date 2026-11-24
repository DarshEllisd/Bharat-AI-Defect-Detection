"""Compile NPU probe models with ExecuTorch's Qualcomm (QNN/HTP) backend.

SM7550 (Snapdragon 7 Gen 3) isn't in ExecuTorch's QcomChipset enum, so we compile the same models for listed
chipsets covering every Hexagon (HTP) generation. QNN context binaries are tied to the
HTP architecture, so a chipset with the same generation as the phone is the most likely
stand-in. The phone run then shows which (if any) actually executes on the NPU.

Outputs in --out:
  <name>.pte, <name>.input.raw (float32), <name>.ref.raw (float32 CPU reference)
  manifest.txt   one line per model: name pte input ref arch soc variant
  export_report.json   compile errors per model, if any
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)

# chipset -> HTP generation, from ExecuTorch v1.5.0 backends/qualcomm/serialization/qc_schema.py.
# The phone's vendor libs stop at HtpV73, so V73 gets two stand-ins: SM8550 (8 MB VTCM) and
# SSG2115P (2 MB VTCM). A context compiled for less VTCM is the safer bet on a mid-range chip.
# (SM7675, the closest mid-range V73 part, only appears in ExecuTorch after v1.5.0.)
CANDIDATES = {
    "SM8350": "v68",
    "SM8450": "v69",
    "SM8550": "v73",
    "SSG2115P": "v73",
    "SM8650": "v75",
    "SM8750": "v79",
    "SM8845": "v81",
}


class ProbeNet(nn.Module):
    """Small conv net: enough ops to exercise the HTP, compiles in seconds."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


def compile_spec(soc: str, fp16: bool):
    return generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, soc),
        backend_options=generate_htp_compiler_spec(use_fp16=fp16),
    )


def quantize_int8(model: nn.Module, example: tuple, soc: str) -> nn.Module:
    """PT2E post-training quantization with the QNN quantizer (default 8-bit)."""
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    try:  # newer releases take backend + soc; older ones take no arguments
        from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType

        quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=getattr(QcomChipset, soc))
    except TypeError:
        quantizer = QnnQuantizer()
    # strict=False: tolerates YOLO's Python-side control flow (the probe net exports either way)
    gm = torch.export.export(model, example, strict=False).module()
    gm = prepare_pt2e(gm, quantizer)
    for _ in range(8):  # calibration on random data is fine: this measures "does it run", not accuracy
        gm(torch.rand_like(example[0]))
    gm(*example)
    return convert_pt2e(gm)


def yolo_module() -> nn.Module:
    """Stock YOLO11n in export mode (single tensor output), as a realistic vision workload."""
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt").model.fuse().eval()
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            m.export = True
            m.format = "executorch"
            m.dynamic = False
    # Like Ultralytics' exporter: one dry run caches the anchors/strides for this input size, so
    # export doesn't trace the data-dependent anchor generation
    with torch.no_grad():
        model(torch.rand(1, 3, 320, 320))
    return model


def write_case(out: Path, name: str, model: nn.Module, example: tuple, soc: str, fp16: bool) -> None:
    with torch.no_grad():
        ref = model(*example)
    pte = to_edge_transform_and_lower_to_qnn(model, example, compile_spec(soc, fp16)).to_executorch()
    (out / f"{name}.pte").write_bytes(pte.buffer)
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    ref.detach().float().numpy().astype(np.float32).tofile(out / f"{name}.ref.raw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--socs", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--no-yolo", action="store_true", help="skip the YOLO11n fp16 variant")
    parser.add_argument("--yolo-int8", action="store_true",
                        help="also export YOLO11n int8 (random-data calibration: valid for latency, not accuracy)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    probe = ProbeNet().eval()
    probe_in = (torch.rand(1, 3, 224, 224),)
    probe_in[0].numpy().astype(np.float32).tofile(out / "probe.input.raw")

    yolo, yolo_in = None, None
    if not args.no_yolo:
        try:
            yolo = yolo_module()
            yolo_in = (torch.rand(1, 3, 320, 320),)
            yolo_in[0].numpy().astype(np.float32).tofile(out / "yolo320.input.raw")
        except Exception:
            print("YOLO unavailable, continuing with probe net only:\n" + traceback.format_exc())

    manifest, report = [], {}
    for soc in args.socs:
        if not hasattr(QcomChipset, soc):  # chip lists differ between ExecuTorch versions
            report[soc] = "chipset not in this ExecuTorch version"
            print(f"[skip]   {soc}: not in this ExecuTorch version's QcomChipset")
            continue
        arch = CANDIDATES[soc]
        # (variant, model factory, example input, fp16, input file); int8 models are quantized per chipset
        cases = [("probe_fp16", lambda: probe, probe_in, True, "probe.input.raw"),
                 ("probe_int8", lambda s=soc: quantize_int8(ProbeNet().eval().requires_grad_(False), probe_in, s),
                  probe_in, False, "probe.input.raw")]
        if yolo is not None:
            cases.append(("yolo320_fp16", lambda: yolo, yolo_in, True, "yolo320.input.raw"))
            if args.yolo_int8:
                cases.append(("yolo320_int8", lambda s=soc: quantize_int8(yolo_module(), yolo_in, s),
                              yolo_in, False, "yolo320.input.raw"))
        for variant, make_model, example, fp16, input_file in cases:
            name = f"{variant}_{soc}"
            try:
                write_case(out, name, make_model(), example, soc, fp16)
                manifest.append(f"{name} {name}.pte {input_file} {name}.ref.raw {arch} {soc} {variant}")
                report[name] = "ok"
                print(f"[ok]     {name}")
            except Exception as e:
                report[name] = f"{type(e).__name__}: {e}"
                print(f"[FAILED] {name}: {report[name]}")

    (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"{len(manifest)} models written to {out}")


if __name__ == "__main__":
    main()
