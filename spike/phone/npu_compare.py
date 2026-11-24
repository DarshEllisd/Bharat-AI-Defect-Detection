"""Compare NPU outputs from 07_npu_probe.sh with the CPU reference and give a verdict.

    python npu_compare.py <probe OUT_DIR> <models dir with manifest.txt>

Works on the phone (Termux) or on the laptop after pulling an adb-shell run.
"""
import re
import sys
from pathlib import Path

import numpy as np

# fp16 should track fp32 closely; int8 with random calibration is looser
MIN_COSINE = {"fp16": 0.99, "int8": 0.95,
              # W3: the reference is the CPU simulation of the same quantised graph, so agreement should be close
              "16a8w": 0.99, "8a8w": 0.97}


def avg_ms(log: Path) -> str:
    """Per-inference time. qnn_executor_runner prints "<N> inference took X ms, avg Y ms" per
    input line (Y = per run) and then "Total 1 inference took X ms, avg X ms" (the line's total)."""
    if not log.exists():
        return ""
    text = log.read_text(errors="replace")
    hits = re.findall(r"^(?!.*Total).*?\d+ inference took [0-9.]+ ms, avg ([0-9.]+) ms", text, re.M)
    return f"{float(hits[0]):.3f}" if hits else ""


def compare(out_file: Path, ref_file: Path):
    if not out_file.exists():
        return None, None, "no output"
    out = np.fromfile(out_file, dtype=np.float32)
    ref = np.fromfile(ref_file, dtype=np.float32)
    if out.size != ref.size:
        return None, None, f"size mismatch {out.size} vs {ref.size}"
    cos = float(np.dot(out, ref) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-12))
    return cos, float(np.max(np.abs(out - ref))), ""


def main() -> None:
    out_dir, models_dir = Path(sys.argv[1]), Path(sys.argv[2])
    print("name,soc,htp,exit_code,avg_ms,cosine,max_abs_diff,verdict")
    working = []
    for line in (models_dir / "manifest.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _pte, _input, ref, arch, soc, variant = line.split()
        case_dir = out_dir / "outputs" / name
        rc_file = case_dir / "exit_code"
        rc = rc_file.read_text().strip() if rc_file.exists() else "not run"
        ms = avg_ms(out_dir / "logs" / f"{name}.log")
        cos, diff, problem = compare(case_dir / "output_0_0.raw", models_dir / ref)
        precision = variant.rsplit("_", 1)[-1]
        if rc != "0":
            verdict = "FAILED_TO_RUN"
        elif cos is None:
            verdict = f"RAN_NO_VALID_OUTPUT ({problem})"
        elif cos >= MIN_COSINE.get(precision, 0.99):
            verdict = "PASS"
            working.append((name, ms))
        else:
            verdict = "RAN_WRONG_OUTPUT"
        print(f"{name},{soc},{arch},{rc},{ms},"
              f"{'' if cos is None else f'{cos:.4f}'},{'' if diff is None else f'{diff:.4g}'},{verdict}")

    print()
    if working:
        print("VERDICT: the NPU is usable through ExecuTorch on this phone with these stand-in chipsets:")
        for name, ms in working:
            print(f"  {name}  avg {ms} ms")
    else:
        print("VERDICT: no QNN-delegated model ran correctly; treat the NPU as not usable via ExecuTorch here.")


if __name__ == "__main__":
    main()
