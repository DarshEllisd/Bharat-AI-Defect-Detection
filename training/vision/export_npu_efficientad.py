"""W3-E - int8 export of a trained EfficientAD-S model for the phone's NPU, with accuracy check.

    bash training/vision/runs/export_npu.sh efficientad mvtec_wood [--quant 16a8w]

Runs in the et-venv (CPU torch + ExecuTorch + QNN SDK env). For model <name>:
  1. PT2E quantisation with the QNN quantizer (8a8w by default, 16a8w fallback; per-channel conv
     weights), calibrated on --calib real TRAINING good images (never val/test).
  2. Accuracy of the quantised graph, simulated on the CPU, on the same val/test data and protocol as
     evaluate_efficientad.py (threshold = 99th pct of val good, re-derived for int8) -> delta vs the
     fp32 eval.json. Acceptance (ADR-019): image AUROC drop <= 0.02.
  3. Lowering: QNN int8 .pte for SSG2115P (V73 stand-in, ADR-012) + fp32 XNNPACK .pte (CPU fallback).
  4. Phone check files: one good + one defect TEST image (preprocessed float32 .raw) with the simulated
     int8 output as reference, and manifest.txt for spike/phone/07_npu_probe.sh.
Output: ~/bharatai-build/models/npu/<name>_<quant>/
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training/vision"))
sys.path.insert(0, str(REPO / "spike/laptop/npu"))
from efficientad_model import EfficientADS, preprocess  # noqa: E402
from export_qnn_probe import compile_spec  # noqa: E402

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype  # noqa: E402
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset, QnnExecuTorchBackendType  # noqa: E402
from executorch.backends.qualcomm.utils.utils import to_edge_transform_and_lower_to_qnn  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

SOC = "SSG2115P"
B = Path.home() / "bharatai-build"
EXTS = {".png", ".jpg", ".jpeg"}
IMG = 256


def images(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).glob("*") if p.suffix.lower() in EXTS)


def load(p: Path) -> torch.Tensor:
    return preprocess(torch.from_numpy(np.array(Image.open(p).convert("RGB"))), IMG)


def load_tiles(p: Path, t: int, synth: "random.Random | None" = None, pen: list[Path] | None = None) -> list[torch.Tensor]:
    """Model inputs for one image: t*t tiles of 1x3x256x256 (t=1: the whole image), as in training (P-57).
    synth: add synthetic defects first (calibration only, P-59)."""
    rgb = np.array(Image.open(p).convert("RGB").resize((IMG * t, IMG * t), Image.BILINEAR))
    if synth is not None:
        rgb = synthetic_defects(rgb, synth, pen or [])
    x = preprocess(torch.from_numpy(rgb), IMG * t)
    return [x[:, :, r * IMG:(r + 1) * IMG, c * IMG:(c + 1) * IMG] for r in range(t) for c in range(t)]


def synthetic_defects(rgb: np.ndarray, rng: random.Random, pen: list[Path]) -> np.ndarray:
    """CALIBRATION ONLY (P-59): paste 1-3 patches from ImageNet-like images and draw 1-2 random strokes,
    so the quantiser's activation ranges cover defect-sized responses. Without this, ranges calibrated on
    good images clip every defect to the same value. Never used for training or evaluation."""
    import cv2
    out = rgb.copy(); h, w = out.shape[:2]
    for _ in range(rng.randint(1, 3)):
        if pen:
            src = np.array(Image.open(rng.choice(pen)).convert("RGB"))
            s = rng.randint(h // 20, h // 5)
            y, x = rng.randint(0, src.shape[0] - s), rng.randint(0, src.shape[1] - s)
            ty, tx = rng.randint(0, h - s), rng.randint(0, w - s)
            out[ty:ty + s, tx:tx + s] = src[y:y + s, x:x + s]
    for _ in range(rng.randint(1, 2)):
        p1 = (rng.randint(0, w - 1), rng.randint(0, h - 1)); p2 = (rng.randint(0, w - 1), rng.randint(0, h - 1))
        col = tuple(int(rng.randint(0, 255)) for _ in range(3))
        cv2.line(out, p1, p2, col, rng.randint(1, max(2, h // 60)))
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count half) - avoids needing scikit-learn in the build venv."""
    pos, neg = s[y == 1], s[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def score_set(fn, val: list[Path], test: Path, t: int = 1) -> dict:
    def score(p: Path) -> float:  # image score = max over the (stitched) tile maps
        return max(float(fn(x).max()) for x in load_tiles(p, t))
    with torch.no_grad():
        v = np.array([score(p) for p in val])
        thr = float(np.percentile(v, 99))
        y, s = [], []
        for d in sorted(x for x in test.iterdir() if x.is_dir()):
            for p in images(d):
                y.append(int(d.name != "good")); s.append(score(p))
    y, s = np.array(y), np.array(s)
    return {"image_auroc": round(auroc(y, s), 4), "threshold": round(thr, 5),
            "false_alarm_rate": round(float((s[y == 0] >= thr).mean()), 4),
            "recall": round(float((s[y == 1] >= thr).mean()), 4), "_scores": s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--quant", choices=["8a8w", "16a8w"], default="8a8w")
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--synth-frac", type=float, default=0.5,
                    help="share of calibration inputs with synthetic defects (activation ranges, P-59)")
    ap.add_argument("--penalty-dir", type=Path, default=Path.home() / "bharatai-build/data/imagenette_flat")
    ap.add_argument("--widen", type=float, default=1.0,
                    help="widen 16-bit activation ranges after calibration; OFF by default: it matched fp32 in "
                         "simulation but gave all-zero outputs on the phone's NPU (linked ranges, P-74)")
    ap.add_argument("--boost-frac", type=float, default=0.25,
                    help="extra calibration tiles (share of --calib) = good tiles with contrast boosted (P-74)")
    ap.add_argument("--boost-gain", type=float, default=6.0, help="contrast gain of the boosted tiles")
    args = ap.parse_args()
    torch.manual_seed(0)

    mdir = B / "models/efficientad" / args.name
    cfg = json.loads((mdir / "config.json").read_text())
    fp32_eval = json.loads((mdir / "eval.json").read_text())
    out = B / "models/npu" / f"{args.name}_{args.quant}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"name": args.name, "soc": SOC, "quant": args.quant, "calibration": None}

    model = EfficientADS(IMG)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    T = int(cfg.get("tiles", 1))
    report["tiles"] = T
    train = images(Path(cfg["train"]))
    rng = random.Random(0)
    rng.shuffle(train)
    # calibration inputs = what the model sees at run time: whole images, or tiles of training images
    pen = sorted(args.penalty_dir.glob("*"))[:3000] if args.penalty_dir.exists() else []
    n_img = max(1, args.calib // (T * T)) * 2
    n_syn = int(round(n_img * args.synth_frac))
    pool_good = [x for p in train[: n_img - n_syn] for x in load_tiles(p, T)]
    pool_syn = [x for p in [rng.choice(train) for _ in range(max(1, n_syn))] for x in load_tiles(p, T, rng, pen)]
    rng.shuffle(pool_good); rng.shuffle(pool_syn)
    k_syn = int(round(args.calib * args.synth_frac))
    calib = pool_good[: args.calib - k_syn] + pool_syn[:k_syn]
    # + contrast-boosted good tiles: they push every layer to defect-like magnitudes, so activation ranges cover
    # real small defects instead of clipping them (P-74: cut scores were compressed 2-4x on the NPU). Unlike
    # editing observer ranges, this goes through normal calibration, so linked (shared) ranges stay consistent.
    n_boost = int(round(args.calib * args.boost_frac))
    calib += [t.mean() + args.boost_gain * (t - t.mean()) for t in pool_good[:n_boost]]
    report["calibration"] = (f"{len(calib)} {'tiles of ' if T > 1 else ''}training good images (seed 0): "
                             f"{k_syn} with synthetic defects (P-59), {n_boost} contrast-boosted x{args.boost_gain:g} (P-74)")
    example = (calib[0],)

    t0 = time.time()
    quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=getattr(QcomChipset, SOC))
    quantizer.set_default_quant_config(
        quant_dtype=QuantDtype.use_16a8w if args.quant == "16a8w" else QuantDtype.use_8a8w,
        is_conv_per_channel=True, is_linear_per_channel=True)
    gm = prepare_pt2e(torch.export.export(model, example, strict=False).module(), quantizer)
    with torch.no_grad():
        for x in calib:
            gm(x)
    # Widen the 16-bit ACTIVATION ranges after calibration (P-74): real small defects push some layers (e.g. the
    # normalised teacher features: 8.8 on a live cut vs a calibrated max of 6.06) beyond the calibrated range, and
    # clipping there shrank cut scores 2-4x on the NPU. 65,536 steps leave plenty of resolution at 3x the range.
    # Only per-tensor 16-bit activation observers are touched; 8-bit per-channel weight observers are not.
    n_wide = 0
    if args.widen > 1.0:
        for mod in gm.modules():
            if (hasattr(mod, "min_val") and hasattr(mod, "max_val") and getattr(mod, "quant_max", 255) > 255
                    and mod.min_val.dim() == 0 and torch.isfinite(mod.min_val) and torch.isfinite(mod.max_val)):
                c, h = (mod.max_val + mod.min_val) / 2, (mod.max_val - mod.min_val) / 2 * args.widen
                mod.min_val.copy_(c - h); mod.max_val.copy_(c + h); n_wide += 1
    report["activation_range_widen"] = f"x{args.widen:g} on {n_wide} 16-bit activation observers (P-74)"
    qgm = convert_pt2e(gm)
    report["quantise_seconds"] = round(time.time() - t0)

    # --- accuracy of the quantised graph vs fp32 (same data, same protocol) ---
    if not args.skip_eval:
        test_dir = Path(fp32_eval["test_dir"])
        val = images(Path(cfg["val"]))
        fp = score_set(model, val, test_dir, T)
        q = score_set(qgm, val, test_dir, T)
        r = float(np.corrcoef(fp.pop("_scores"), q.pop("_scores"))[0, 1])
        report["fp32"] = fp
        report[args.quant] = q
        report["delta_auroc"] = round(q["image_auroc"] - fp["image_auroc"], 4)
        report["score_correlation_fp32_vs_quant"] = round(r, 4)
        report["accept_auroc_drop_le_0.02"] = report["delta_auroc"] >= -0.02
        print(json.dumps(report, indent=2))

    # --- phone check inputs: first good and first defect test image ---
    test_dir = Path(fp32_eval["test_dir"])
    picks = {"good": images(test_dir / "good")[0]}
    defect_dir = next(d for d in sorted(test_dir.iterdir()) if d.is_dir() and d.name != "good")
    picks["defect"] = images(defect_dir)[0]
    lines = []
    pte = f"efficientad_{args.name}_{args.quant}_{SOC}.pte"
    for tag, p in picks.items():
        with torch.no_grad():  # tiled: send the tile with the highest fp32 score (the defect tile)
            x = max(load_tiles(p, T), key=lambda tl: float(model(tl).max()))
        x.numpy().astype(np.float32).tofile(out / f"{tag}.input.raw")
        with torch.no_grad():
            qgm(x).numpy().astype(np.float32).tofile(out / f"{tag}.ref.raw")
            model(x).numpy().astype(np.float32).tofile(out / f"{tag}.fp32.raw")
        lines.append(f"{args.name}_{args.quant}_{tag} {pte} {tag}.input.raw {tag}.ref.raw v73 {SOC} ead_{args.quant}")
        report[f"phone_input_{tag}"] = f"{p.parent.name}/{p.name}"

    # --- lowering ---
    try:
        edge = to_edge_transform_and_lower_to_qnn(qgm, example, compile_spec(SOC, fp16=False))
        try:
            from executorch.devtools.backend_debug import get_delegation_info
            info = get_delegation_info(edge.exported_program().graph_module)
            report["delegation"] = {"delegated": info.num_delegated_nodes, "not_delegated": info.num_non_delegated_nodes}
        except Exception as e:  # noqa: BLE001
            report["delegation"] = f"unavailable: {e}"
        (out / pte).write_bytes(edge.to_executorch().buffer)
        (out / "manifest.txt").write_text("\n".join(lines) + "\n")
        report["qnn"] = f"ok ({(out / pte).stat().st_size / 1e6:.1f} MB)"
    except Exception as e:  # noqa: BLE001
        report["qnn"] = f"{type(e).__name__}: {e}"
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        ep = torch.export.export(model, example, strict=False)
        (out / f"efficientad_{args.name}_fp32_xnnpack.pte").write_bytes(
            to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch().buffer)
        report["xnnpack_fp32"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["xnnpack_fp32"] = f"{type(e).__name__}: {e}"

    (out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("fp32", args.quant)}, indent=2))


if __name__ == "__main__":
    main()
