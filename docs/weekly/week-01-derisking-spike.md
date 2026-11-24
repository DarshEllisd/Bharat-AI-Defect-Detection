# Week 01: De-risking tests on the real phone

- **Dates:** 2026-11-01 (the spike completed in one long session)
- **Plan reference:** PLAN.md → W1
- **Runbook:** [`spike/README.md`](../../spike/README.md) · **Raw results:** [`spike/results/W1_RESULTS.md`](../../spike/results/W1_RESULTS.md)

## Goals
Before building anything, prove on the actual phone that:
1. ExecuTorch runs in Termux, for both Llama (advisory) and a vision model.
2. It's fast enough.
3. The camera → backend path works.
4. Which runtime route to use (native vs Python).
5. Whether the NPU is usable, **tested on the device rather than assumed from support lists**.

## Work done (in order)
1. **Laptop setup:** WSL memory raised, adb installed; uv Python 3.12 added.
2. **Phone setup** in Termux: Python 3.14, numpy, Pillow, SciPy, OpenCV (x11-repo + dbus), aiohttp. Device profile confirmed SM7550, 4×A510 + 4×A715, `i8mm`/`bf16`, Android 14.
3. **Built** ExecuTorch v1.5.0 Android runners (`llama_main`, `executor_runner`) with NDK r28c. The first attempt stalled on a slow network; root cause found and fixed with WSL mirrored mode (IPv6).
4. **Built remote-control helpers** to run commands and copy files into Termux over adb.
5. **Llama 3.2 1B** (pre-exported SpinQuant INT4) downloaded, verified, copied; thread-sweep benchmark run.
6. **YOLO11n** exported to `.pte` (320 and 640 px); CPU benchmark run.
7. **Camera link** test: Chrome → WebSocket → Termux at 320/640/1280 px.
8. **Route A** (Python runtime in proot Debian) benchmarked against native, then rejected; container removed.
9. **NPU:**
   - Built ExecuTorch's Qualcomm backend (QAIRT 2.37).
   - Compiled test models for stand-in chips of every Hexagon generation.
   - Ran them via adb and **inside the Termux app**.
   - Then compiled YOLO11n int8 for the working stand-in (SSG2115P) and benchmarked it.
10. Wrote up the results, decisions and problems (this `docs/` folder).

## Results
| Metric | Target | Result |
|---|---|---|
| ExecuTorch runs in Termux | must pass | ✅ native NDK binaries |
| Llama 1B decode | ≥ 10 tok/s | ✅ **13.3 tok/s** (4 threads) |
| Llama 1B TTFT (354-token prompt) | ≤ 3 s | ⚠️ **3.13 s** (prefill-bound) |
| Llama peak RAM | ≤ 2 GB | ✅ **1.38 GB** |
| Advice grounded in the knowledge base | yes | ✅ actions copied from the KB; JSON needs repair |
| YOLO11n 320 px, CPU (fp32) | ≤ 40 ms | ✅ **32.2 ms** |
| YOLO11n 320 px, **NPU (int8)** | — | ✅ **1.81 ms**, output matches |
| Camera link 640 px | ≥ 15 FPS, p95 ≤ 60 ms | ✅ **29 FPS, 19.9 ms** |
| Route A within 15% of native | nice to have | ❌ 1.8-4.6× slower → rejected |

## Decisions made
- [ADR-009](../decisions/ADR-009-laptop-toolchain.md) Laptop toolchain
- [ADR-010](../decisions/ADR-010-runtime-route-native.md) Native runtime (route B)
- [ADR-011](../decisions/ADR-011-web-stack-aiohttp.md) aiohttp + OpenCV from x11-repo
- [ADR-012](../decisions/ADR-012-npu-vision-cpu-llm.md) **Vision on the Hexagon NPU, Llama on the CPU**
- [ADR-013](../decisions/ADR-013-vision-input-resolution.md) 320 px input, cropped frames
- [ADR-014](../decisions/ADR-014-wsl-networking.md) WSL mirrored networking
- [ADR-015](../decisions/ADR-015-remote-phone-workflow.md) Remote phone workflow
- ADR-001 and ADR-007 were confirmed with measurements.

## Problems hit
P-01 to P-30 in [`problems/PROBLEMS_AND_FIXES.md`](../problems/PROBLEMS_AND_FIXES.md). The ones that cost the most time:
- **P-10:** slow WSL downloads. The real cause was the ISP's IPv4 routes; the fix was IPv6 via mirrored mode.
- **P-13 to P-15:** bugs in the copy helper (corrupted text and binaries; adb crash on 1.1 GB).
- **P-20:** wrong Qualcomm toolkit version and chip list from development docs, caught by a preflight check before a 40-minute build.
- **P-21 / P-23:** the NPU only works from inside the Termux app, and only with 2 MB-VTCM (SSG2115P) contexts.

## Open items / carried over
- **P-28:** LLM JSON repair and validation (W5).
- Reduce TTFT: shorter system prompt or prefill caching (W5).
- Check **EfficientAD** compiles to int8 QNN for SSG2115P (early W3).
- Real int8 calibration plus an accuracy check against fp32 for every NPU model (W3).
- Persistent C++ runner processes for vision (NPU) and LLM (W4).
- Measure CPU contention between Llama and the rest of the pipeline (W4).

## Next week
**W2: Data.**
- Prepare MVTec for the 9 categories (masks → YOLO boxes, splits).
- Build the capture rig.
- Collect good and self-made defect photos of the real objects available.
