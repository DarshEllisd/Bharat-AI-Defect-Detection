# Plan: Edge Quality Inspector (AIBharat PS5)

## Context

AIBharat Problem Statement 5 (see `PROBLEM_STATEMENT.md`) asks for an **on-device, real-time visual inspection system** that must:
1. detect defects;
2. classify their type and severity;
3. alert the operator with actionable advice;
4. log every inspection;
5. run fully at the edge.

**Llama + ExecuTorch are mandatory** for the advisory reasoning. The project folder currently holds only the brief, so this is a greenfield build.

**Documentation:** [`docs/`](docs/README.md) records every decision (ADRs with options, why and trade-offs), a weekly log, and all problems with their fixes. Update it whenever a decision is made or a problem is solved.

**Decisions made with the user**

| Topic | Decision |
|---|---|
| Target device | **OnePlus CPH2613**, **Qualcomm Snapdragon 7 Gen 3 (SM7550)**, 8 GB RAM (see the device profile below) |
| Runtime host (phase 1) | Python backend inside **Termux** on the phone, serving a **local web dashboard** at `localhost`. Native Android app is a later TODO. |
| Camera | Dashboard page (Chrome) uses `getUserMedia` → JPEG frames → **WebSocket** → backend |
| Domain / data | **MVTec AD**, 9 categories: screw, metal_nut, transistor, cable, carpet, leather, tile, wood, grid |
| Product selection | **Operator selects the product** in the UI, which loads that category's models |
| Vision approach | **Hybrid**: supervised YOLO (named defect classes) + anomaly model (catches unseen defects) |
| Severity | **Rule-based**, deterministic and configurable; Llama explains it but does not decide it |
| LLM | **Llama 3.2 1B Instruct**, 4-bit, exported to ExecuTorch `.pte` (XNNPACK) |
| Language | English only |
| Logging | SQLite + CSV/JSON export + saved defect crops and heatmaps |
| Training | Laptop with NVIDIA GPU (Windows/WSL2). Nothing is trained or inferred in the cloud at runtime. |
| Demo | **Real physical objects** in front of the phone camera |
| Team / time | Solo, comfortable with Python/ML; 1–2 months (~8 weeks) |

### Target device profile: OnePlus CPH2613, Snapdragon 7 Gen 3, 8 GB RAM

| Aspect | Implication for this project |
|---|---|
| **SoC**: SM7550, platform "crow", soc_id 608; Android 14 (SDK 34). **Confirmed on device 2026-11-01** | — |
| **CPU**: 4× Cortex-A510 @ 1.8 GHz (cpu0-3) + 3× Cortex-A715 @ 2.4 GHz (cpu4-6) + 1× A715 @ 2.63 GHz (cpu7). ISA includes `asimddp`, `i8mm`, `bf16`, no SVE. **Confirmed on device** | Llama runs here via ExecuTorch XNNPACK (4 threads on the A715 cores, 13.3 tok/s measured). Vision falls back to this CPU path (YOLO11n fp32 at 320 px: 32 ms). The efficiency cores stay free for the server and UI. |
| **GPU**: Adreno 720 | Possible later speed-up for vision via ExecuTorch's Vulkan delegate. Not needed for the baseline. |
| **NPU**: Qualcomm Hexagon | Phone equivalent of the brief's Hailo NPU. ExecuTorch's Qualcomm (QNN) chipset list (`backends/qualcomm/serialization/qc_schema.py`, checked Nov 2026) has no SM7550; its only 7-series entry is SM7675 (7+ Gen 3). **This is tested empirically in W1** (`spike/phone/07_npu_probe.sh`): the QNN platform validator plus ExecuTorch QNN models compiled for stand-in chipsets of each Hexagon generation, run from both adb shell and Termux. **W1 result (2026-11-01):**
- The QNN validator confirms **Hexagon V73**; **VTCM ≤ 2 MB**. Only contexts compiled for **SSG2115P** (V73, 2 MB VTCM) load; 4/8 MB contexts are rejected.
- **YOLO11n int8 at 320 px: 1.81 ms on the NPU** vs 32 ms fp32 on the CPU, output matches.
- It works **from inside the Termux app** (`untrusted_app`) with `/vendor/lib64` on `LD_LIBRARY_PATH`. It is blocked for the `run-as` remote shell.
- **Decision: vision on the Hexagon NPU (int8), Llama on the Arm CPU**, the brief's two-pipeline split. CPU/XNNPACK vision is kept as a fallback. |
| **RAM**: 8 GB (7.07 GiB visible to Android; only ~2.7 GB was free with apps open) | Close background apps before benchmarks and the demo. Memory budget: Llama 3.2 1B 4-bit ≈ 1–1.5 GB, vision models < 100 MB (one category loaded at a time), Chrome and the camera ≈ 0.5–1 GB. Comfortable headroom. **Llama 3.2 3B (4-bit, ≈ 2.5–3 GB) is feasible as an optional quality upgrade**; benchmark it in W1 against 1B, which stays the default. |
| **Thermals** | Mid-range phones throttle under sustained load. Measure over a 10-minute run (W7); keep vision input at 256–320 px and cap LLM output tokens. |

---

## Architecture

```
┌──────────────────────── Android phone (all on-device, airplane-mode capable) ────────────────────────┐
│                                                                                                       │
│  Chrome tab: http://localhost:8000 (Operator UI)                                                      │
│   ├─ getUserMedia camera → canvas → JPEG ──WebSocket /ws/frames──┐                                   │
│   └─ live overlay, verdict banner, alerts (sound + vibrate), advice, log, stats ◄── /ws/events ──┐   │
│                                                                   ▼                              │   │
│  Termux: Python orchestrator (aiohttp)                                                           │   │
│   1 Pre-process (OpenCV: crop ROI, resize, stability/trigger detection)                          │   │
│   2 VISION PIPELINE  ── ExecuTorch QNN backend, int8, Hexagon V73 NPU (CPU/XNNPACK fallback) ──  │   │
│        • YOLO11n (defect classes)          • EfficientAD-S per category (anomaly map + score)    │   │
│   3 Fusion → Defect record (type, severity, bbox, area, anomaly score, confidence)               │   │
│   4 Rule-based severity + PASS/FAIL disposition ──► immediate alert ────────────────────────────►│   │
│   5 ADVISORY PIPELINE ── separate process: Llama 3.2 1B via ExecuTorch, Arm CPU ──               │   │
│        structured record + knowledge-base snippet → JSON {explanation, cause, actions, log_note} ─┘   │
│   6 Storage: SQLite + crops/heatmaps on disk + CSV/JSON export                                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**The two pipelines are documented separately**, mirroring the brief's Hailo split:
- The **vision pipeline** handles perception. Its own process, models and runtime config sit behind a `VisionRuntime` interface. It runs on the **Hexagon V73 NPU** via ExecuTorch's Qualcomm (QNN) backend: int8 models, contexts compiled for the SSG2115P stand-in (V73, 2 MB VTCM), and `/vendor/lib64` on `LD_LIBRARY_PATH` when launched from the Termux app. YOLO11n at 320 px takes 1.81 ms there. CPU/XNNPACK is the fallback.
- The **advisory pipeline** runs Llama via ExecuTorch on the Arm CPU in a separate worker. It is fed from an async queue so it **never blocks the vision loop**.

### Key design choices
- **Item-level inspection ("station mode").** Frames stream continuously for a live overlay. When an item is placed and the image is stable (frame-difference trigger), the system runs one full inspection and writes **one log entry per item**, not one per frame.
- **Hybrid fusion.** The anomaly map is thresholded into connected regions, and each region is matched (IoU) to YOLO boxes:
  - region with a YOLO match → classified defect;
  - region with no match → `unknown_anomaly`;
  - YOLO box with high confidence but a low anomaly score → kept, flagged `detector_only`.
- **Severity** is computed from config (`severity.yaml`) as `class criticality × area fraction × anomaly score × zone weight`. It maps to LOW / MEDIUM / HIGH / CRITICAL and a disposition (PASS / REWORK / HOLD / SCRAP). It is deterministic and unit-tested.
- **Grounded advice.** Each category has a hand-written knowledge base (`kb/<category>.yaml`: defect → likely causes → corrective actions). Llama receives the JSON record plus the matching KB entries and must return JSON whose actions are drawn from the KB. If the output is invalid JSON or ungrounded, a templated fallback is used, so an alert is never empty.
- **Domain gap for real objects.** The anomaly model needs **only good samples**, so the plan captures 100–200 phone photos of good real objects per demo product in a fixed rig (phone stand, plain background, steady lighting). YOLO is fine-tuned on MVTec defects plus some self-made defects (scratched screw, cut cable, torn fabric). MVTec test sets are kept for objective metrics through a **replay mode**.

---

## Repository layout (to create)

```
BharatAI-SOC/
  PROBLEM_STATEMENT.md          (exists)
  README.md                     setup, phone deploy, run instructions
  training/                     laptop-side (GPU)
    data/prepare_mvtec.py       9 categories; masks → YOLO boxes; splits
    data/capture_tool/          helper to collect own phone images
    vision/train_yolo.py        Ultralytics YOLO11n, all defect classes
    vision/train_anomaly.py     anomalib EfficientAD-S per category
    vision/export_vision.py     torch.export → ExecuTorch .pte (XNNPACK, int8) + parity check
    vision/evaluate.py          AUROC, pixel-AUROC, mAP, miss/false-alarm at threshold
    llm/export_llama.sh         Llama 3.2 1B Instruct → .pte (4-bit, XNNPACK) + tokenizer
    llm/eval_advice.py          groundedness and format checks on sample records
  edge/                         runs in Termux on the phone
    app/main.py                 server, WebSocket endpoints, static files
    app/pipeline/preprocess.py  ROI, resize, stability trigger
    app/pipeline/vision.py      VisionRuntime (ExecuTorch), YOLO + anomaly wrappers
    app/pipeline/fusion.py      region extraction, IoU matching
    app/pipeline/severity.py    rule engine
    app/pipeline/record.py      DefectRecord / InspectionRecord dataclasses (JSON schema)
    app/advisor/llm_worker.py   separate process, ExecuTorch Llama runner, queue
    app/advisor/prompt.py       prompt + JSON output parsing/validation + fallback
    app/storage/db.py           SQLite schema (inspections, defects, advice, acks)
    app/storage/export.py       CSV/JSON export
    config/categories.yaml      per-category model paths, thresholds, ROI
    config/severity.yaml
    config/kb/<category>.yaml
    web/                        index.html, app.js, styles.css (no build step)
    runtimes/                   NDK build scripts for C++ runners (if fallback B is needed)
  scripts/deploy_to_phone.sh    adb push models/config; Termux setup
  scripts/benchmark.py          on-device FPS, per-stage latency, LLM TTFT + tok/s, RAM
  models/                       .pte artifacts (gitignored)
  docs/technical_report.md      design, both runtimes, results, limits
  tests/                        unit tests for severity, fusion, record, prompt parsing
```

---

## Milestones (~8 weeks)

**W1: De-risking spike (most important)**
- Set up Termux on the phone and decide how ExecuTorch runs there:
  - **Route A:** `proot-distro` Debian (glibc) plus the ExecuTorch aarch64 Python wheel, so everything stays in Python.
  - **Route B:** small C++ runners (`vision_runner`, ExecuTorch's `llama_main`) cross-compiled with the Android NDK, driven from Python over stdin or a Unix socket.
- Record the device profile: CPU core layout, CPU feature flags (`asimddp`, `i8mm`, `sve2`), available RAM.
- Run the stock Llama 3.2 1B `.pte` on the phone and measure time-to-first-token and tok/s with 4 vs. 8 threads; optionally compare 3B.
- **Empirical NPU test**: QNN `qnn-platform-validator`, plus ExecuTorch QNN probe models (fp16, int8, YOLO11n) compiled for a stand-in chipset of each Hexagon generation. Run from adb shell and from Termux, with outputs checked against a CPU reference. This decides whether the vision pipeline uses the NPU or the CPU.
- Raise the WSL memory limit (currently 4 GB) to about 10 GB plus swap before building; install adb.
- **W1 outcome (2026-11-01, details in `spike/results/W1_RESULTS.md`):**
  - Llama 3.2 1B SpinQuant runs natively in Termux: 13.3 tok/s decode, ~3.1 s TTFT on a 354-token prompt, 1.4 GB RSS, 4 threads.
  - YOLO11n fp32 at 320 px takes 32 ms. Camera → WebSocket reaches 29 FPS at 640 px.
  - Route A (Python in proot) is 1.8-4.6× slower, so **Route B is chosen**: native ExecuTorch processes, Python orchestrator in Termux, aiohttp.
  - **NPU works**: Hexagon V73 with VTCM ≤ 2 MB. SSG2115P-compiled contexts run YOLO11n int8 at 320 px in 1.81 ms from inside the Termux app. Vision moves to the NPU; Llama stays on the CPU.
- **Implications for W3-W4:**
  - Vision models must be **int8 with QNN-compatible ops**: fp16 fails for the 2 MB-VTCM target, and EfficientAD needs checking.
  - Calibrate on real MVTec/own images.
  - Measure int8 accuracy against fp32.
  - Keep the XNNPACK fallback.
- Runbook and scripts: `spike/README.md`.
- Export a toy YOLO11n `.pte` and run it on the phone.
- Prove browser camera → WebSocket → backend echo on `localhost`.
- Web stack: **aiohttp** (FastAPI tested 2026-11-01 and rejected: pydantic-core needs a Rust build on Android). OpenCV comes from x11-repo (`python-opencv-python`).
- Download MVTec AD.

**W2: Data** (detailed plan: `docs/weekly/week-02-data.md`; decisions ADR-016, ADR-017)
- Download MVTec for the 9 categories (official per-category archives, ~3.3 GB, ~15 min; CC BY-NC-SA 4.0).
- `prepare_mvtec.py`: masks → YOLO boxes; `<category>_<defect>` classes; 60/20/20 stratified splits; per-category anomaly layout; box QA grids.
- Phone **capture tool** (Termux web page: product, label, defect instance ID, square guide, burst).
- **Live-demo products: wood + fabric** (fabric ≈ MVTec carpet). Improvised rig. ~120 good + ~40 self-made defect photos each (~2.5 h).
- Label own defects in **local Label Studio**. Split by defect instance; hold out 20% as the real-object test set.
- Stretch: EfficientAD int8 on the NPU (SSG2115P) feasibility check.
- Data card `docs/data/DATASET.md`.

**W3: Vision models** (detailed plan: `docs/weekly/week-03-training.md`; ADR-018, ADR-019)
- A separate CUDA `train-venv` (torch 2.14) for training; `et-venv` (CPU torch) stays for export.
- EfficientAD-S (own module, anomalib pretrained teacher) for **own wood, own fabric, MVTec wood, MVTec carpet**.
- YOLO11n on the combined MVTec + own dataset (48 classes, 320 px, standard augmentation).
- Evaluation per ADR-019:
  - anomaly: image AUROC, MVTec pixel AUROC, own box hit rate, false-alarm rate + recall at a threshold set on val;
  - YOLO: mAP50 per class on MVTec test + own test.
- int8 QNN (SSG2115P) export calibrated on real images; int8-vs-fp32 deltas (AUROC ≤ 0.02, mAP50 ≤ 5 points; 16a8w fallback); on-phone NPU latency + agreement.
- Model card `docs/models/MODELS.md`.

**W4: Edge pipeline** (detailed plan: `docs/weekly/week-04-edge-pipeline.md`; ADR-020/021/022)
- Fixed stand + lamp; **Inspect button** grabs a full-resolution frame through the training capture path (ADR-020; no auto-trigger).
- **Per-product calibration** on the phone from ~20 good frames, saved until redone (ADR-021).
- **Persistent native `vision_server`** (ExecuTorch QNN + XNNPACK, Unix socket), with models resident (ADR-022).
- Python pipeline: shared preprocessing (parity test), VisionRuntime, fusion, severity (Claude drafts `severity.yaml`, user reviews), records, SQLite + images/heatmaps/crops, CSV/JSON export.
- Minimal operator page (the full dashboard is W6).
- Replay-mode parity on the phone; benchmark (button-to-verdict p50/p95, RAM); user test session.
- Estimated ~1 calendar week: Claude ~8-12 h, user ~1.5-2 h.

**W5: Advisory**
- Write the KB YAMLs and the prompt with JSON output.
- Validation, templated fallback and the async worker process.
- Advice evaluation (grounded-action rate, valid-JSON rate, latency).

**W6: Operator dashboard**
- Live view with overlays and heatmap toggle; product selector.
- Verdict banner (green PASS / amber / red FAIL by severity), sound and vibration alerts, an operator "acknowledge" button.
- Advice panel (shows the alert immediately, fills in the advice when ready).
- Log table with filters, CSV/JSON export, stats (pass rate, defect breakdown), and a performance panel (FPS, per-stage ms, tok/s).

**W7: Integration and tuning**
- End-to-end runs on real objects; tune per-category thresholds for misses vs. false alarms.
- Performance work: input resolution, thread counts, quantisation, thermal behaviour over a 10-minute run.

**W8: Report, video, cleanup**
- `docs/technical_report.md` with **separate sections for the vision and advisory pipelines** (runtime, models, quantisation, measured latency for each), plus results and limitations.
- Demo video showing detection, alerts and logging.
- README and deploy script.

**Later TODO:** native Android app (CameraX + the ExecuTorch Android AAR), reusing the same `.pte` models, `severity.yaml` and KB YAMLs.

---

## Verification

- **Unit tests** (`pytest tests/`): severity rules (boundary cases), fusion matching, record JSON schema, prompt-output parsing and fallback, SQLite round-trip.
- **Detection quality** (`training/vision/evaluate.py` and replay mode on the phone): image AUROC, pixel AUROC, and miss rate vs. false-alarm rate at the chosen threshold per category. Laptop and phone results must agree (parity).
- **Real-time** (`scripts/benchmark.py` on the phone): end-to-end FPS, p50/p95 latency per stage, inspection-to-alert time, LLM time-to-first-token and tok/s, peak RAM, plus a 10-minute thermal run.
- **Advice quality** (`training/llm/eval_advice.py`): % valid JSON, % of actions grounded in the KB, plus a manual 1–5 usefulness rating on about 30 records.
- **Edge proof:** run the full demo with the phone in **airplane mode**.
- **End-to-end:** place good and defective real objects; confirm each gives one log row, the right verdict, an alert, advice, saved crops, and working CSV/JSON export.

## Key risks
- ExecuTorch on Termux (bionic libc). Mitigated by the W1 spike with routes A and B.
- MVTec → real-object domain gap. Mitigated by own good-sample capture and a fixed rig.
- Thin MVTec defect data for YOLO. Mitigated by heavy augmentation, self-made defects, and the anomaly branch as a safety net.
- Chrome stops the camera when backgrounded. The app has to stay in the foreground during inspection (documented as a limitation; the native app fixes this).
- 1B LLM quality. Mitigated by KB grounding, JSON validation and the templated fallback.
