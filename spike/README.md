# Week 1: De-risking Spike

Goal: before building the real system, prove on the **actual phone (OnePlus CPH2613, Snapdragon 7 Gen 3 / SM7550, 8 GB)** that:

1. **ExecuTorch runs in Termux.** Llama for the advisory pipeline, plus a YOLO `.pte` as a proxy for the vision pipeline.
2. **It's fast enough.** We measure LLM time-to-first-token and tokens/s, vision latency, and peak memory.
3. **The camera path works.** Chrome camera → WebSocket → Termux backend, at a usable frame rate.
4. **We know which runtime route to use** for the rest of the project:
   - **Route B (native):** ExecuTorch C++ runners cross-compiled with the Android NDK, run directly in Termux. This is the officially documented Android path.
   - **Route A (Python):** the ExecuTorch Python runtime (PyPI aarch64 wheel) inside a `proot-distro` Debian userland. Termux itself can't load manylinux wheels (it uses bionic libc, not glibc).

5. **Whether the NPU (Hexagon HTP) is actually usable.** ExecuTorch's Qualcomm backend doesn't list SM7550, so instead of trusting the list we **test on the device**:
   - Qualcomm's `qnn-platform-validator` checks whether the HTP is reachable and reports its Hexagon version.
   - ExecuTorch QNN models are compiled for listed chipsets of every Hexagon generation (v68–v81). We run them on the phone and check their outputs against a CPU reference.
   - Both run from `adb shell` and from Termux, because Android may allow DSP access from one and block the other.

```
spike/
  laptop/                    run in WSL2 Ubuntu on the laptop
    build_android_runners.sh   NDK + ExecuTorch v1.5.0 → llama_main, executor_runner (arm64)
    export_yolo_pte.py         YOLO11n → .pte at 320/640 px (vision latency proxy)
    get_llama_pte.sh           Llama 3.2 1B Instruct .pte + tokenizer (download or export)
    push_to_phone.sh           adb push scripts + artifacts → phone Download/bharatai/
    npu/build_qnn.sh           QNN SDK + ExecuTorch Qualcomm backend + probe models → artifacts/npu/
    npu/export_qnn_probe.py    probe net (fp16 + int8) and YOLO11n compiled per Hexagon generation
    npu/run_npu_adb.sh         run the NPU probe via adb shell and pull the results
  phone/                     run in Termux on the phone (copied to ~/bharatai/spike)
    00_setup_termux.sh         packages (numpy, Pillow, SciPy, OpenCV via x11-repo, aiohttp)
    01_device_profile.sh       SoC, cores, ISA flags (i8mm/dotprod), RAM
    02_install_artifacts.sh    copy binaries/models into Termux home, chmod +x
    03_bench_llama.sh          Llama thread sweep with a real defect-advisory prompt
    04_bench_vision.sh         YOLO .pte latency via executor_runner (route B)
    05_route_a_proot.sh        ExecuTorch Python runtime in proot Debian (route A)
    ws_echo/                   camera → WebSocket round-trip test page + server
    07_npu_probe.sh            NPU test: device facts, QNN validator, ExecuTorch QNN models (POSIX sh)
    npu_compare.py             NPU output vs CPU reference → PASS/FAIL verdict
    06_collect_results.sh      tarball of all results for the laptop
  artifacts/                 build outputs (binaries, .pte models), created by laptop scripts
  results/                   put the phone's results tarball here
```

---

## Part A: Laptop (WSL2 Ubuntu)

**Prerequisites** (checked on this laptop on 2026-11-01)

| Item | Status | Action |
|---|---|---|
| WSL2 + Ubuntu 22.04, 235 GB free, RTX 4050 visible | ✅ installed | none |
| Python in Ubuntu is 3.10; ExecuTorch 1.5 needs ≥ 3.11 | handled | the build script installs Python 3.12 with `uv`; system Python is untouched |
| WSL memory (was capped at 4 GB; host has 16 GB) | ✅ raised by user | The ExecuTorch build and the Llama export need roughly 8–12 GB |
| `adb` (Android platform-tools) | ✅ installed | Add the **platform-tools folder** to PATH, enable **USB debugging** on the phone, and accept the prompt when you run `adb devices` |
| Hugging Face account with the **Llama 3.2 licence** accepted | ? | Run `hf auth login` after A1, inside the venv |

To raise WSL memory, edit `%UserProfile%\.wslconfig` so it reads:
```ini
[wsl2]
memory=10GB
processors=6
swap=16GB
```
Then run `wsl --shutdown` in PowerShell and reopen Ubuntu. Docker Desktop shares this limit. Exporting the **3B** model yourself would need about 24 GB or more, so for 3B use only a pre-exported file.

**Slow downloads: keep WSL in mirrored networking mode.**
Root cause, measured on 2026-11-01: this internet connection's **IPv4** route to some CDNs is very slow (Fastly/PyPI ~40 KB/s, CloudFront/PyTorch ~15 KB/s, GitHub ~8 KB/s), on Windows too, while **IPv6** to the same servers runs at 11–14 MB/s. WSL's default NAT mode is IPv4-only, so everything crawled. With `networkingMode=Mirrored` (WSL Settings → Networking), WSL gets IPv6, and Python/pip/curl prefer it automatically (~12 MB/s).
- Check any time: `bash spike/laptop/check_hosts.sh` (speed and IP version per build host).
- Still slow: **GitHub and Qualcomm**, which are IPv4-only. If a large download from them stalls, use a different network for it (e.g. phone hotspot on the second Wi-Fi adapter).

Run from the project root inside WSL (`cd /mnt/c/<path-to>/BharatAI-SOC`):

```bash
# A1. Build the Android runners (first run ~1-2 h: NDK download, ExecuTorch install + build)
bash spike/laptop/build_android_runners.sh

# A2. Export YOLO11n .pte files with the same ExecuTorch version
source ~/bharatai-build/et-venv/bin/activate
pip install "ultralytics>=8.3.220"
python spike/laptop/export_yolo_pte.py --sizes 320 640

# A3. Get Llama 3.2 1B Instruct (pre-exported SpinQuant INT4, ~1.1 GB)
hf auth login
bash spike/laptop/get_llama_pte.sh download
#    optional: also export it ourselves, and a 3B for comparison
#    bash spike/laptop/get_llama_pte.sh export
#    SIZE=3B bash spike/laptop/get_llama_pte.sh export

# A4. NPU test bundle: QNN SDK + ExecuTorch Qualcomm backend + probe models (~30-60 min)
#     If the SDK download asks for a login, download the Qualcomm AI Runtime zip manually
#     from softwarecenter.qualcomm.com and run:  QNN_ZIP=/path/to/zip bash spike/laptop/npu/build_qnn.sh
bash spike/laptop/npu/build_qnn.sh

# A5. Phone connected over USB → push everything to Download/bharatai/
bash spike/laptop/push_to_phone.sh

# A6. NPU test via adb shell (Qualcomm's documented developer route); results → spike/results/npu_adb/
bash spike/laptop/npu/run_npu_adb.sh
```

## Part B: Phone (Termux)

**B0.** Install **Termux from F-Droid or its GitHub releases**. The Play Store build is outdated. Keep the phone **on the charger with battery saver off** for all benchmarks, and close other apps.

**B1.** Open Termux and grant storage access:
```bash
termux-setup-storage
```

**B2.** Bootstrap the scripts, install packages and artifacts, and record the device profile:
```bash
mkdir -p ~/bharatai/spike && cp -r ~/storage/downloads/bharatai/phone/. ~/bharatai/spike/ && sed -i 's/\r$//' ~/bharatai/spike/*.sh
bash ~/bharatai/spike/00_setup_termux.sh
bash ~/bharatai/spike/02_install_artifacts.sh
bash ~/bharatai/spike/01_device_profile.sh
```

**B3.** Llama benchmark for the advisory pipeline (about 5 minutes):
```bash
bash ~/bharatai/spike/03_bench_llama.sh
# 3B or our own export:  MODEL=~/bharatai/models/llama/<file>.pte bash ~/bharatai/spike/03_bench_llama.sh
```

**B4.** Vision benchmark (route B):
```bash
bash ~/bharatai/spike/04_bench_vision.sh
```

**B5.** Camera → WebSocket test:
```bash
python ~/bharatai/spike/ws_echo/server.py
```
Then switch to **Chrome** and open **http://localhost:8000**, allow the camera, and run once each at width **320, 640 and 1280** (30 s each). Results save automatically. Stop the server with Ctrl+C.

**B6.** Route A, the Python runtime in proot. It downloads about 1–2 GB (Debian, torch, executorch):
```bash
bash ~/bharatai/spike/05_route_a_proot.sh
```

**B6b.** NPU test from inside Termux, the context our app runs in. Compare it with A6:
```bash
sh ~/bharatai/spike/07_npu_probe.sh
```

**B7.** Collect the results and bring them back:
```bash
bash ~/bharatai/spike/06_collect_results.sh
```
On the laptop: `adb pull /sdcard/Download/bharatai/w1_results_<timestamp>.tar.gz spike/results/`, then share it with Claude so we can analyse it together.

---

## What we're looking for

The targets below are working estimates for a smooth operator experience, not measured facts. W1 replaces them with real numbers.

| Check | Working target | Why it matters |
|---|---|---|
| `llama_main` runs inside Termux | must pass | Llama + ExecuTorch is mandatory |
| Llama 1B decode speed | ≥ 10 tok/s | A ~100-token advice message in ≤ 10 s |
| Llama 1B time-to-first-token (~400-token prompt) | ≤ 3 s | Advice starts appearing quickly after the alert |
| Llama peak RSS | ≤ 2 GB | Leaves room for Chrome, the camera and the vision models |
| YOLO11n @ 320 px per inference | ≤ 40 ms | Leaves budget for the anomaly model in the hybrid pipeline |
| Camera loop @ 640 px | ≥ 15 FPS, RTT p95 ≤ 60 ms | Live overlay feels real-time |
| OpenCV imports in Termux (x11-repo) | nice to have | Otherwise Pillow + SciPy cover decode, resize and region labelling |
| Route A works, latency within ~15% of route B | nice to have | Decides the vision runtime |
| NPU probe: any `PASS` in `npu_compare.py` | informative | Decides whether vision can move to the NPU |

**NPU decision rules**

| adb shell result | Termux result | Conclusion |
|---|---|---|
| PASS | PASS | NPU works through ExecuTorch with a stand-in chipset. The vision pipeline can run on the HTP (compare its ms against XNNPACK from B4) while Llama stays on the CPU, which mirrors the brief's Hailo split. |
| PASS | FAIL | The NPU works, but Android blocks it from Termux. Use the CPU for phase 1; the native Android app (later) can use the NPU. |
| validator passes, ExecuTorch FAILs | — | The HTP is reachable, but ExecuTorch can't target this SoC. CPU-only, with proof for the report. |
| validator FAILs too | — | No user-space HTP access on this phone. CPU-only, with proof for the report. |

The device facts in `npu_probe.txt` (`soc0/soc_id`, vendor `libQnnHtpV**` library names) also show the real Hexagon version, which tells us which stand-in chipset is the fair match.

**Decision rules**
- **Route A works and is close to route B in speed:** run the vision pipeline in Python (`executorch.runtime`), and run Llama as a separate persistent native process. This keeps the two pipelines cleanly split.
- **Route A fails or is much slower:** use route B for both, with small C++ sidecar processes driven by the Python orchestrator over stdin or a Unix socket.
- **Llama 1B is too slow:** try fewer or more threads, a shorter prompt, and fewer `max_new_tokens`. 1B stays the model; 3B is only an upgrade if it's clearly fast enough.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `llama_main` errors loading the pre-exported `.pte` (version / flatbuffer error) | It was exported with an older ExecuTorch. Run `get_llama_pte.sh export` on the laptop, re-push, re-run B3. |
| `Permission denied` running a binary | Binaries must run from `~/bharatai/bin`, not shared storage. Re-run `02_install_artifacts.sh`. |
| Termux processes get killed ("signal 9") | Run `termux-wake-lock`. On Android 14+ enable *Developer options → Disable child process restrictions*. On Android 12–13 run `adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent; /system/bin/device_config put activity_manager max_phantom_processes 2147483647"`. |
| Camera error in Chrome | Use `http://localhost:8000` (a secure context), not the phone's LAN IP. Check that Chrome has camera permission. |
| Need a reference run outside Termux | `adb push` the binary and model to `/data/local/tmp/llama/` and run via `adb shell` (the official ExecuTorch Android flow) to compare. |
| WSL build is very slow | Builds live in `~/bharatai-build` on the Linux filesystem; don't move them to `/mnt/c`. |
