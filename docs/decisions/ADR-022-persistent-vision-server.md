# ADR-022: A persistent native vision server, called from Python over a Unix socket

- **Status:** Accepted (validated on the phone 2026-11-22)
- **Date:** 2026-11-22
- **Week:** 4 (planning)

## Context
- The Python orchestrator in Termux (ADR-010, Route B) can't load ExecuTorch directly: there is no bionic Python wheel, and the proot route was 1.8-4.6× slower.
- Weeks 1-3 ran models with `qnn_executor_runner`, which **loads the model on every invocation**. That's fine for benchmarks, but too slow per inspection: each call pays model and context loading (~0.2-0.4 s) on top of ~28 ms of inference.
- The NPU is only reachable from processes started inside the Termux app.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Small C++ server (NDK, ExecuTorch QNN + XNNPACK), models resident, Unix domain socket** (chosen) | Load once; per-call cost ≈ inference + a few ms of IPC; one process owns the NPU; easy to restart | Custom C++ to build and maintain |
| Spawn `qnn_executor_runner` per inspection | No new code | +0.2-0.4 s per call, temp files, no YOLO in the same call |
| Python ExecuTorch bindings (pybind) built for Termux | Simplest calling code | Building ExecuTorch's Python extension against Termux's Python/bionic is uncharted and heavy |
| proot Python (Route A) | Python only | 1.8-4.6× slower (W1), and NPU access from proot is unproven |

## Decision
Build **`vision_server`** from ExecuTorch v1.5.0 with the Android NDK, reusing W1's `qnn_executor_runner` build setup.
- **Startup:** loads the selected product's anomaly `.pte` (QNN, NPU) and YOLO (XNNPACK, CPU), and keeps them resident.
- **Protocol:** listens on a Unix domain socket in Termux's home. Request = model id + float32 tensor; reply = output tensors + per-call timings.
- **Switching:** the product is switched by a load/unload request.
- **Lifecycle:** Python starts it, health-checks it and restarts it if it crashes.

## Why
- Inspection latency is dominated by model loading unless models stay resident.
- A single owner process for the NPU avoids repeated context setup.
- Unix sockets are available in Termux, fast, and keep the vision pipeline a separate process, matching the brief's separate vision/advisory pipelines.

## Trade-offs and consequences
- More native code: build scripts go in the repo, and a parity check against `qnn_executor_runner` outputs is required.
- If the build proves too hard in week 4, the fallback is a long-running loop variant of `qnn_executor_runner` reading inputs from a FIFO.

## Validation (2026-11-22, phone, from the Termux app; `spike/results/vision_server_test.json`)
- **Outputs:** **identical** to `qnn_executor_runner` on all 4 week-3 anomaly check inputs (max abs diff 0.0). YOLO (XNNPACK) matches laptop fp32 within 0.0009.
- **Residency:** own_wood + own_fabric_t2 (NPU) + YOLO (CPU) are loaded **at the same time**. Load times: first QNN model 222 ms (NPU init), second 16 ms, YOLO 27 ms. Switching products needs no reload.
- **Latency (100 calls):** NPU compute 28.4 ms p50 / 28.9 p95; YOLO compute 22 ms; Python round trip 34 ms (anomaly) / 27 ms (YOLO). IPC overhead ~6 ms after making the client zero-copy (it was 14 ms).
- **Memory:** server RSS 29-37 MB, unchanged over 1,000 calls (no leak).
- **Build notes:** P-64 (link setup), P-65 (YOLO export via Ultralytics).

## Evidence / references
- ADR-010, ADR-012; W1 build `spike/laptop/npu/build_qnn.sh`; week-3 phone run `spike/results/npu_w3/`
