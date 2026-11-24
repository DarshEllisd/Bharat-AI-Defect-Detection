# Week 1 Spike: Results

Running log of measured results. Raw logs are under `spike/results/phone_live/`.

## Device (confirmed on the phone, 2026-11-01)

| Item | Value |
|---|---|
| Phone | OnePlus CPH2613, Android 14 (SDK 34) |
| SoC | Qualcomm SM7550 (Snapdragon 7 Gen 3), platform `crow`, soc_id 608 |
| CPU | 4× Cortex-A510 @ 1.8 GHz (cpu0-3), 3× Cortex-A715 @ 2.4 GHz (cpu4-6), 1× A715 @ 2.63 GHz (cpu7) |
| ISA | `asimddp`, `i8mm`, `bf16`, `asimdhp`; no SVE/SVE2 |
| RAM | 7.07 GiB visible; ~2.8 GB free with apps open; 5.5 GB zram swap |
| Storage | ~13-15 GB free (tight) |
| NPU evidence | OnePlus ships QNN + SNPE in `/odm/lib64` (`libQnnHtp.so`, stubs up to `libQnnHtpV73Stub.so`), so the HTP is very likely **Hexagon V73**. `libcdsprpc.so` is in `public.libraries.txt`; `/dev/adsprpc-smd` is world-readable |

## Termux environment

| Item | Result |
|---|---|
| Python | 3.14.8 |
| numpy / Pillow / SciPy | 2.5.4 / 13.0.0 / 1.18.1 (Termux packages) |
| OpenCV | 5.0.0 via x11-repo `python-opencv-python`. Needs `dbus` and `dpkg --configure -a`; JPEG encode and `connectedComponents` verified |
| Web stack | **aiohttp 3.14.5**. FastAPI rejected: `pydantic-core` needs a Rust build that pip can't bootstrap on Android |

## Laptop build (A1)

| Item | Result |
|---|---|
| ExecuTorch | v1.5.0, editable install, Python 3.12.16 (uv), torch 2.14.0+cpu |
| NDK | r28c, API 23, arm64-v8a |
| `llama_main` | 8.1 MB, statically linked C++, needs only Android system libs |
| `executor_runner` | 6.7 MB after `llvm-strip` (155 MB with debug info) |
| Runs in Termux | **Yes, both** (route B confirmed) |

**Network finding:** this connection's IPv4 route to Fastly (PyPI), CloudFront (pytorch.org), GitHub and AWS runs at ~10-40 KB/s, on Windows too, while IPv6 runs at 11-14 MB/s. WSL NAT mode is IPv4-only, so the first build crawled. WSL **mirrored networking** gives it IPv6 (~12 MB/s). GitHub and Qualcomm are IPv4-only and stay slow.

## Advisory pipeline: Llama 3.2 1B Instruct (B3)

Model `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` (1.14 GB, sha256 `8715cdba…cda3b`), pre-exported by executorch-community; it loads in the v1.5.0 runtime. Prompt: the 354-token defect-advisory prompt (`spike/phone/prompts/advisory_prompt.txt`); 107 tokens generated; temperature 0; native in Termux.

| Threads | Load (s) | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Peak RSS (MB) |
|---|---|---|---|---|---|
| 2 | 2.96 | 5.65 | 62.7 | 12.4 | 1377 |
| **4** | 3.44 | **3.13** | 113.0 | **13.3** | 1376 |
| 6 | 2.86 | 3.31 | 107.0 | 12.6 | 1377 |
| 8 | 2.87 | 2.78 | 127.4 | 12.5 | 1377 |
| auto (-1) | 2.87 | 3.15 | 112.2 | 13.4 | 1377 |

- Against targets: decode ≥ 10 tok/s ✅, RSS ≤ 2 GB ✅, TTFT ≤ 3 s ⚠️ (borderline, dominated by the 354-token prefill).
- **Choice: 4 threads** (one per A715 core). More threads barely help, and four leaves cores for vision.
- **Output quality:** actions and cause were taken verbatim from the knowledge-base notes; severity untouched. But it prefixed "Here is the response:" and **omitted the closing `}`**, so the W5 parser must extract and repair JSON, with a template fallback.
- **Design implications:** keep a persistent LLM process (3 s load per start). Shorten the static system prompt or cache its prefill to cut TTFT.

## Vision pipeline: YOLO11n via XNNPACK (B4)

Stock YOLO11n (2.6 M params), exported by Ultralytics 8.4.197 with ExecuTorch 1.5.0, **fp32, not quantized** (~10.6 MB each). `executor_runner`, native in Termux, 50 runs, input of ones. Per-inference time = (T₅₀ − T₁) / 49 wall clock; it agrees with the runner's own per-iteration log (median 31.6 ms at 320, ~138 ms at 640).

| Input | Threads | Per inference (ms) | FPS | Load + 1 run (ms) |
|---|---|---|---|---|
| **320×320** | **4** | **32.2** | **31.1** | 154 |
| 320×320 | 2 | 53.8 | 18.6 | 170 |
| 320×320 | auto | 32.1 | 31.2 | 151 |
| 640×640 | 4 | 140.7 | 7.1 | 281 |
| 640×640 | 2 | 224.3 | 4.5 | 375 |
| 640×640 | auto | 140.1 | 7.1 | 273 |

- Against target (YOLO11n at 320 ≤ 40 ms): **32 ms ✅**, which leaves budget for the anomaly model in the hybrid pipeline.
- **640 px is too slow** (7 FPS, ~4.4× the cost of 320, which tracks the 4× pixel count). Stay at 256-320 px as planned.
- Auto threads picks the 4 performance cores, the same as `--cpu_threads 4`.
- Expected upside: int8 quantization (W3) should roughly halve latency on the `i8mm`/dot-product cores. Not measured yet.
- Open question for W4: vision and Llama both want the 4 A715 cores. Measure contention when an advisory runs during inspection.

## Camera link: Chrome → WebSocket → Termux (B5)

Chrome `getUserMedia` (back camera) → canvas → JPEG q=0.8 → WebSocket → aiohttp server in Termux, which decodes each frame with OpenCV and replies. One frame in flight at a time (maximum sustainable loop rate), 30 s per run. The camera delivered **720×1280 (portrait)**, so frame height follows that aspect, and the 1280-wide run is **upscaled** from 720 (stress test only).

| Width sent | Frame | FPS | Browser encode avg / p95 (ms) | Round trip avg / p95 (ms) | Server decode (ms) | KB/frame |
|---|---|---|---|---|---|---|
| 320 | 320×569 | **48.5** | 13.7 / 22.0 | 6.9 / 10.2 | 2.1 | 30.6 |
| **640** | 640×1138 | **29.0** | 19.7 / 27.6 | 14.8 / 19.9 | 6.0 | 82.9 |
| 1280 (upscaled) | 1280×2276 | 10.8 | 55.1 / 64.3 | 37.5 / 45.9 | 18.3 | 236.2 |

- Against target (640 px ≥ 15 FPS, RTT p95 ≤ 60 ms): **29 FPS, 19.9 ms ✅**.
- Browser-side JPEG encoding costs more than the transfer. The model input is 320 px, so the real app should send ~320-480 px crops of the inspection area, not full frames.
- The phone was held in portrait. For the station, lock the orientation or crop a square region of interest, so frames aren't mostly background.

## Route A: ExecuTorch Python runtime in proot Debian (B6)

proot-distro 5.x installed the official `debian` image from Docker Hub (Python 3.13.5). pip `executorch==1.5.0` (cp313 aarch64 wheel) + `torch==2.14.0+cpu` from the PyTorch CPU index (the executorch wheel does **not** pull in torch). The Python LLM-runner bindings are present in the wheel. Same `.pte` files as B4; random input, 5 warm-up + 50 timed runs. Container size on disk: **2.1 GB**.

| Input | Threads | Route A p50 (ms) | Route B native (ms) | Route A / B |
|---|---|---|---|---|
| 320 | 2 | 101.1 | 53.8 | 1.9× |
| 320 | 4 | 149.8 | 32.2 | **4.6×** |
| 320 | 8 | 180.1 | — | — |
| 640 | 2 | 269.5 | 224.3 | 1.2× |
| 640 | 4 | 256.6 | 140.7 | 1.8× |
| 640 | 8 | 332.5 | — | — |

- Route A works, but it is **much slower than native, and gets slower with more threads**. The likely cause (unverified) is proot's syscall interception, which makes threadpool synchronisation expensive.
- The container was removed afterwards to free storage. `05_route_a_proot.sh` reinstalls it if ever needed.
- **Decision: Route B for both pipelines.** Native ExecuTorch C++ processes (vision runner, LLM runner) run in Termux, driven by the Python orchestrator (Termux's own Python 3.14) over stdin or a Unix socket. This is also the path that carries over to the native Android app (JNI) later.
- `executor_runner` already has a `--server_mode` (one inference per stdin line). It's good enough to prototype; W4 replaces it with a small persistent vision server that exchanges raw tensors.

## NPU probe: Hexagon HTP via ExecuTorch Qualcomm backend (A4/A6/B6b)

Built from ExecuTorch v1.5.0 + Qualcomm AI Runtime (QAIRT) **2.37.0.250724** (the version v1.5.0 pins; 2.51 is for newer ExecuTorch). SM7550 isn't in ExecuTorch's chip list, so models were compiled for listed stand-in chips of each Hexagon generation and run on the phone; outputs were compared with the CPU reference. Reports: `spike/results/npu_adb/`, `spike/results/phone_live/npu_termux_app/`.

**Hardware facts (proven, not inferred):**
- `qnn-platform-validator` (adb shell): "Core Version of the backend DSP: **Hexagon Architecture V73** … Unit Test: **Passed**".
- **VTCM ≤ 2 MB.** Contexts compiled for 4 MB (SM8350) or 8 MB (SM8450/SM8550) fail with `Request feature vtcm size with value … unsupported`. V75+ contexts fail with `arch … unsupported`. **Only SSG2115P (V73, 2 MB VTCM) contexts load.**

**Results by context:**

| Context (SELinux domain) | Result |
|---|---|
| `adb shell` (`shell`) | ✅ SSG2115P models run; outputs match |
| Remote `run-as` shell (`runas_app`) | ❌ Can't load `libcdsprpc.so` (vendor lib, apps need a manifest declaration); with copied libs → `createUnsignedPD … not supported` |
| **Inside the Termux app (`untrusted_app_27`)** | ✅ **Works** when `/vendor/lib64` is on `LD_LIBRARY_PATH` (or with copies of `libcdsprpc.so`, `vendor.qti.hardware.dsp@1.0.so`, `libvmmem.so`) |

**Latency inside the Termux app** (`qnn_executor_runner`, 3 warm-up + 20 runs, per-run average):

| Model | Precision | NPU (HTP) | CPU (XNNPACK, 4 threads) | Output check |
|---|---|---|---|---|
| Probe conv net, 224 px | int8 | 0.26 ms | — | cosine 1.0000 ✅ |
| **YOLO11n, 320 px** | **int8** | **1.81 ms** | 32.2 ms (fp32) | cosine 0.9934 vs the int8 CPU simulation ✅ |

- **~18× faster than the CPU** (note: int8 NPU vs fp32 CPU; int8 XNNPACK on the CPU isn't measured yet).
- The int8 calibration used random data, so this measures **speed, not detection accuracy**. Real calibration and accuracy come in W3.
- YOLO fp16 doesn't compile for SSG2115P, and the probe fp16 build fails on `adaptive_avg_pool2d`. **The NPU path is int8.**
- YOLO int8 export needed Ultralytics-style prep: set Detect `export/dynamic=False` and run one dry forward pass to cache the anchors, then `torch.export(strict=False)`.
- Correction: earlier reports said "6.28 ms". That was the runner's total for 20 runs, not one. The per-run figure is 0.31 ms (probe); the parsers now read the per-run line.

**Decision:** move the **vision pipeline to the Hexagon NPU** (ExecuTorch QNN backend, int8, SSG2115P-compiled context, launched from the Termux app with `/vendor/lib64` on `LD_LIBRARY_PATH`). **Llama stays on the CPU** (ExecuTorch XNNPACK). That is exactly the brief's two-pipeline split (Hailo NPU for vision, Arm CPU for Llama), here with a Hexagon NPU instead. Keep the CPU/XNNPACK vision path as a fallback.

## Week 1 summary

| Question | Answer |
|---|---|
| Does ExecuTorch run in Termux? | Yes, native NDK binaries (route B) |
| Llama 3.2 1B on the phone | 13.3 tok/s, ~3.1 s TTFT, 1.4 GB RAM; grounded advice, imperfect JSON |
| Vision on the CPU | YOLO11n fp32 at 320 px: 32 ms |
| **Vision on the NPU** | **YOLO11n int8 at 320 px: 1.81 ms, works from inside the Termux app** |
| Camera link | 29 FPS at 640 px, 20 ms p95 round trip |
| Python runtime (proot) | Works but 1.8-4.6× slower, so rejected |
| Web stack | aiohttp (FastAPI needs Rust on Android) |
