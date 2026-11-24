# Edge Quality Inspector: on-device defect detection and QC advice on an Android phone

AIBharat Bharat AI-SoC Student Challenge, **Problem Statement 5** (industrial defect detection and quality-control assistant).

A phone on a fixed stand inspects a product when the operator presses **Inspect**. Everything runs **on the phone, offline**:

```
Chrome page (camera, dashboard)
   │  full-resolution frame (HTTP, localhost)
   ▼
Python server (aiohttp, Termux) ── frame check ── preprocessing (bit-identical to training)
   │  Unix socket
   ▼
vision_server (C++, ExecuTorch v1.5.0, models resident)
   ├─ EfficientAD-S anomaly model ── Hexagon V73 NPU (QNN, 16a8w)   ~28.5 ms per 256 px tile
   └─ YOLO11n defect classifier    ── CPU (XNNPACK, fp32)            ~22 ms at 320 px
   ▼
fusion (anomaly regions + YOLO boxes) → rule-based severity → verdict PASS / REWORK / HOLD / SCRAP
   ▼
SQLite log + image, heatmap, crops ── CSV/JSON export
   ▼
Llama 3.2 1B Instruct (ExecuTorch, SpinQuant INT4, CPU) → advice grounded in a knowledge base
```

**Llama and ExecuTorch** are used for both pipelines: ExecuTorch runs the vision models (QNN backend on the NPU, XNNPACK on the CPU) and Llama 3.2 1B (XNNPACK on the CPU).

## Results at a glance
| | |
|---|---|
| Device | OnePlus CPH2613 (Snapdragon 7 Gen 3, Hexagon V73 NPU, 8 GB), Android 14, Termux |
| Anomaly model on the NPU | 28.5 ms per run, 146/147 graph nodes on the NPU; ~50× faster than the same model on the CPU (1,419 ms) |
| Quantisation loss (16a8w vs fp32) | AUROC −0.001 (fabric), 0.000 (wood) |
| Inspection, server time | 0.33 s (fabric, 4 tiles) with the camera page closed; 1.1 s p50 while Chrome's camera runs |
| Llama advice | valid, KB-grounded JSON in 22/22 live answers; 11 s idle, 40-90 s alongside the live camera |
| Tests | 44 unit tests (`pytest tests/`) |

Full write-up: **[docs/technical_report.md](docs/technical_report.md)**.

## Repository layout
| Path | What |
|---|---|
| `edge/` | The phone application: `vision_server/` (C++), `app/` (Python server, pipeline, advisor, storage), `config/` (products, severity rules, knowledge base), `web/` (operator dashboard), `scripts/`. See [edge/README.md](edge/README.md). |
| `training/` | Dataset preparation, EfficientAD-S and YOLO11n training, evaluation, NPU export (`training/vision/`), Label Studio helpers (`training/data/`). |
| `spike/` | Week-1 de-risking: ExecuTorch builds for Android, NPU probes, benchmarks, laptop↔phone helpers. Results in `spike/results/`. |
| `tests/` | Unit tests for the edge pipeline. |
| `docs/` | Decisions (ADR-001..022), weekly logs, problems and fixes (P-01..P-74), model and data cards, technical report. Index: [docs/README.md](docs/README.md). |
| `PLAN.md`, `PROBLEM_STATEMENT.md` | Project plan and the challenge brief. |

## Quick start
1. **Build** (laptop, WSL, Android NDK): ExecuTorch v1.5.0 with the QNN backend (QAIRT 2.37) and `edge/vision_server/build_android.sh`. See `spike/README.md`.
2. **Models**: train and export with `training/vision/` (see `docs/models/MODELS.md`). Model files (`.pte`), Qualcomm runtime libraries and the Llama weights are **not** in this repository (size and licences); they are produced by the scripts or downloaded from their sources.
3. **Deploy** to the phone (Termux) with `spike/laptop/termux_sync.sh`, then in the Termux app: `sh ~/bharatai/edge/scripts/start_inspector.sh`. Chrome opens on `http://localhost:8000`.
4. **Use**: choose the product, **Calibrate** (20 good frames), then **Inspect**.

## Licences of third-party components
MVTec AD (CC BY-NC-SA 4.0, non-commercial), Llama 3.2 (Llama 3.2 Community Licence), Ultralytics YOLO11 (AGPL-3.0), ExecuTorch (BSD), Qualcomm QAIRT (Qualcomm licence), anomalib teacher weights (Apache-2.0). Details in `docs/models/MODELS.md` §7.
