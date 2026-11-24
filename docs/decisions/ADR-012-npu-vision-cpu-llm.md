# ADR-012: Vision runs on the Hexagon V73 NPU (int8, SSG2115P stand-in); Llama runs on the CPU

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
The brief's reference design runs vision on an NPU (Hailo via HailoRT) and Llama on the Arm CPU (ExecuTorch), documented as two separate pipelines. Our phone has a Qualcomm Hexagon NPU. ExecuTorch's Qualcomm (QNN) backend only officially supports chips in its `QcomChipset` list, and **SM7550 is not on it**. The first conclusion, "not supported", came from reading that list. **The user asked for proof on the actual device instead.**

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Trust the support list → CPU only | No work | Might leave an NPU unused for no reason |
| **Test on the device**: Qualcomm's `qnn-platform-validator`, plus ExecuTorch QNN models compiled for stand-in chips of each Hexagon generation, run from `adb shell` and from inside Termux, with outputs checked against a CPU reference | Definitive | A Qualcomm toolkit download (2.97 GB) and a backend build |

## What the test proved
- **Hexagon V73**, per Qualcomm's validator ("Core Version … Hexagon Architecture V73 … Unit Test: Passed").
- **VTCM (on-chip NPU memory) ≤ 2 MB.** Contexts compiled for 4 MB (SM8350) or 8 MB (SM8450/SM8550) are rejected: `Request feature vtcm size … unsupported`. V75+ contexts are rejected: `arch … unsupported`.
- **Only contexts compiled for SSG2115P (V73, 2 MB VTCM) load**, and their outputs match the CPU reference.

**Context matters:**

| Where the process runs | Result |
|---|---|
| `adb shell` | ✅ works |
| `run-as com.termux` remote shell | ❌ can't load the vendor `libcdsprpc.so`; with copies, `createUnsignedPD … not supported` |
| **Inside the Termux app** | ✅ works with `/vendor/lib64` on `LD_LIBRARY_PATH` (or with copies of the vendor FastRPC libs) |

## Decision
- **The vision pipeline runs on the Hexagon NPU:** ExecuTorch QNN backend, **int8** models, contexts compiled for **SSG2115P**, processes launched from the Termux app with `/vendor/lib64` on `LD_LIBRARY_PATH`.
- **Llama stays on the CPU** (ExecuTorch XNNPACK).
- **CPU/XNNPACK vision stays as a fallback.**
- Build pins: ExecuTorch v1.5.0 + **QAIRT 2.37.0.250724**.

## Why
| YOLO11n, 320 px | Where | Per image | Output check |
|---|---|---|---|
| fp32, XNNPACK | CPU, 4 threads | 32.2 ms | — |
| **int8, QNN** | **NPU (V73)** | **1.81 ms** | cosine 0.9934 vs the int8 CPU simulation |

- About **18× faster**, which frees the CPU for Llama and the server.
- It reproduces the brief's two-pipeline split exactly, with the Hexagon NPU in place of the Hailo chip.

## Trade-offs and consequences
- **int8 only:** fp16 builds fail for SSG2115P (YOLO context generation fails; `adaptive_avg_pool2d` isn't delegated). All vision models need int8 and QNN-compatible ops; **check EfficientAD early**.
- The 1.81 ms used **random-data calibration**: it's a speed figure, not an accuracy figure. Calibrate on real images and measure int8 vs fp32 accuracy in week 3.
- The SSG2115P stand-in is an assumption that works. A newer ExecuTorch may add a closer chip; re-test when upgrading.
- NPU processes must start from the Termux app (not `run-as`). Automated NPU tests are typed into Termux with `adb shell input text` ([ADR-015](ADR-015-remote-phone-workflow.md)).
- The native Android app (later) can declare `libcdsprpc.so` in its manifest, which is the clean long-term route.

## Update 2026-11-01 (week 2): the anomaly model confirmed on the NPU
EfficientAD-S (own module, int8, calibrated on real MVTec wood images) compiles for SSG2115P with 123 of 124 nodes delegated. It runs in **15.3 ms on the NPU** vs **1,419 ms on the CPU** (XNNPACK fp32), with output matching the int8 reference (cosine 0.961). That makes the NPU **required**, not just faster, for the anomaly half of the hybrid pipeline ([ADR-005](ADR-005-hybrid-vision.md)). Details: `docs/weekly/week-02-data.md` → E.


## Update 2026-11-22 (week 3, trained models)
- **Anomaly models (EfficientAD-S) run on the NPU at 16a8w** (16-bit activations, 8-bit weights), not 8a8w.
  - 8-bit activations can't represent the tiny student-teacher differences of normal images.
  - Activation ranges are calibrated on good images plus synthetic defects (P-59).
  - All four compile with 146/147 nodes on the NPU. AUROC Δ vs fp32: own_wood 0.000, own_fabric_t2 −0.001, mvtec_carpet +0.001, mvtec_wood −0.030.
- **Fabric uses 2×2 tiles** (P-57): 4 NPU runs per frame for fabric, 1 for wood.
- **YOLO11n runs fp32 on the CPU (XNNPACK, ~32 ms at 320 px)**. Its class scores collapse under QNN quantisation at every precision tried (P-62). YOLO is light enough for the CPU; the anomaly model is the one that needs the NPU.
- On-phone NPU latency and agreement for the trained models: week-3 phone run (results in `docs/weekly/week-03-training.md`).

## Evidence / references
- `spike/results/W1_RESULTS.md` → NPU probe
- `spike/results/npu_adb/`, `spike/results/phone_live/npu_termux_app/`
- `spike/laptop/npu/` (build, export, preflight), `spike/phone/07_npu_probe.sh`, `npu_compare.py`, `npu_app_test.sh`
