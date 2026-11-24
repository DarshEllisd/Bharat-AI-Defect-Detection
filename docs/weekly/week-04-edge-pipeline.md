# Week 04: Edge pipeline on the phone (plan)

- **Status:** In progress (started 2026-11-22)
- **Decisions:**
  - [ADR-020](../decisions/ADR-020-button-trigger-fixed-rig.md): button-triggered full-resolution capture, fixed stand.
  - [ADR-021](../decisions/ADR-021-per-product-calibration.md): per-product calibration.
  - [ADR-022](../decisions/ADR-022-persistent-vision-server.md): persistent native vision server.
- **Estimated effort:** ~1 calendar week.
  - Claude ~8-12 h, over 2-4 sessions.
  - User ~1.5-2 h: build the stand + lamp (~30-45 min), one test session (~30 min), start the server in Termux when asked.

## Goal
On the phone, fully offline, the operator presses **Inspect** and gets back, within about a second:
- a verdict (**PASS / REWORK / HOLD / SCRAP**);
- a severity;
- the defect list (type, location, size, score);
- a heatmap overlay.

Every inspection is **logged** (SQLite + saved image, heatmap and crops; CSV/JSON export). Llama advice (week 5) and the full operator dashboard (week 6) plug into this later.

## Inputs (from weeks 1-3)
- **Models** (`docs/models/MODELS.md`):
  - anomaly: `own_wood` (1 NPU run per frame) and `own_fabric_t2` (4 tiles) at 16a8w on the NPU, ~28.5 ms per run;
  - YOLO11n fp32 on the CPU (XNNPACK, ~32 ms);
  - MVTec wood/carpet models for replay mode.
- **Phone runtime:**
  - ExecuTorch v1.5.0 + QAIRT 2.37 libraries in `~/bharatai/npu`.
  - The NPU works only from inside the Termux app, with `/vendor/lib64` on the library path.
  - aiohttp + OpenCV in Termux. The capture tool already grabs 3072×4096 frames.
- **Lessons that shape the design:**
  - Thresholds must come from the deployed model's own outputs under current conditions (P-56, phone run).
  - Preprocessing must match training (P-60).
  - Camera distance must match training (P-51).

## Tasks

### A. Persistent native vision server (Claude; the largest task) - ADR-022
- **`vision_server`:** a small C++ program built with the Android NDK against ExecuTorch v1.5.0 (QNN + XNNPACK backends), like W1's `qnn_executor_runner` build.
  - Loads each model **once** and keeps it resident: the anomaly model per product on the NPU, YOLO on the CPU.
  - Serves inference requests from Python over a **Unix domain socket**: `{model, tensor}` in, output tensors out, timings included.
- **Why:** starting `qnn_executor_runner` per frame reloads the model (~0.2-0.4 s per call). A resident process makes the call cost ≈ inference time.
- **Checks:**
  - outputs bit-identical to `qnn_executor_runner` on the week-3 check inputs;
  - per-call latency and IPC overhead on the phone;
  - survives 1,000 calls without leaking memory.

### B. Python inspection pipeline (`edge/app/`, Claude)
- `pipeline/preprocess.py` (**one shared preprocessing**, P-60): capture square → antialiased resize (OpenCV `INTER_AREA`) → ImageNet normalisation. Fabric: 512 px → 4 tiles. YOLO: 320 px letterbox. A parity test against the training code is required.
- `pipeline/vision.py`:
  - a `VisionRuntime` client for the server (CPU/XNNPACK fallback if the NPU isn't available);
  - stitches the tile maps;
  - decodes YOLO (normalised boxes ×320, NMS, filter to the selected product's classes).
- `pipeline/fusion.py`: threshold the anomaly map into connected regions, match regions to YOLO boxes by IoU:
  - matched → typed defect;
  - unmatched region → `unknown_anomaly`;
  - confident YOLO box without anomaly support → `detector_only`.
- `pipeline/severity.py` + `config/severity.yaml`: `criticality(type) × area fraction × anomaly score` → LOW / MEDIUM / HIGH / CRITICAL → disposition. **Claude drafts the table with reasoning; the user reviews it.**
- `pipeline/record.py`: `InspectionRecord` / `DefectRecord` dataclasses with a JSON schema. Week 5's advisor consumes these.
- `pipeline/calibrate.py` (ADR-021): ~20 good frames → deployed-model image scores → threshold = 99th percentile, plus a safety margin from the score spread. Saved per product in `state/calibration.json` with the date, frame count and score stats. Shown in the UI; re-calibrate any time.
- `storage/db.py` + `storage/export.py`: SQLite (inspections, defects, calibrations); full image, heatmap PNG and defect crops on disk; CSV/JSON export.
- `main.py`: aiohttp app with endpoints for frames, `inspect`, `calibrate`, `products`, `log` and `export`.

### C. Minimal operator page (Claude; the full dashboard is week 6)
- Live preview with the **framing square** (as in the capture tool) and a product selector (wood / fabric / MVTec replay).
- **Calibrate** (guided: "place a good item, press 20 times / auto-burst") and **Inspect** buttons.
- Result card: verdict colour, severity, defect list, heatmap overlay toggle, inspection time.
- Last-10 log table.

### D. Tests (laptop, `pytest tests/`)
- severity boundary cases; fusion matching; record schema round-trip; SQLite round-trip; calibration maths;
- **preprocessing parity**: the edge pipeline vs the training `preprocess()` on the same images, max abs diff below a tolerance.

### E. On-phone validation (Claude, phone connected)
- **Replay mode** (no camera): run the MVTec and own **test** images through the full phone pipeline. Image scores, AUROC and decisions must match the laptop (week-3) numbers within tolerance.
- **Benchmark** (`scripts/benchmark.py`): per-stage p50/p95 (capture → preprocess → NPU → YOLO → fusion → log), button-to-verdict time, peak RAM, for wood (1 NPU run) and fabric (4 runs). 30 inspections each.

### F. User test session (~30 min, after A-E work)
1. Fixed stand at one height and the same lamp; mark the item spot; take a frame to check that the framing matches the training photos (Claude compares loop/grain scale, P-51).
2. Calibrate wood and fabric.
3. Inspect ~10 good + ~10 defective placements per product.
4. Claude checks that each gave one log row, verdict, heatmap and crops, and that CSV/JSON export works, then reports the live false alarms and misses.

### G. Documentation (standing requirement)
- Week log results.
- ADR-020..022 (and updates if anything changes).
- Problems log entries; data/model card updates if needed.
- Edge README: how to start the server in Termux, how to calibrate and inspect.

## Acceptance criteria
- [ ] `vision_server` runs from the Termux app: models resident, outputs identical to `qnn_executor_runner`, per-call overhead measured
- [ ] Shared preprocessing passes the parity test against the training code
- [ ] Replay mode on the phone reproduces the laptop results (AUROC within 0.01, same decisions on ≥ 95% of images)
- [ ] Calibrate + Inspect work end to end for wood and fabric; one log row per inspection with image, heatmap, crops; CSV/JSON export
- [ ] Severity table drafted and reviewed by the user; unit tests pass
- [ ] Benchmark: button-to-verdict p50/p95 per product and peak RAM reported (target ≤ 1 s, ideally ≤ 0.5 s)
- [ ] Week log, ADRs and problems log updated

## Risks
| Risk | Mitigation |
|---|---|
| Building a custom ExecuTorch C++ program for Android (link flags, backend registration, QNN libraries) | Reuse W1's working `qnn_executor_runner` build tree and flags; start from its source; fallback: one long-running `qnn_executor_runner`-style loop over a FIFO |
| Full-resolution frame grab is slow in Chrome (3072×4096 canvas + JPEG) | Crop the square in the browser and send only the square; measure; fall back to a smaller square (the models use ≤ 512 px anyway) |
| Live frames differ from training photos (lighting, distance, camera auto-exposure) | Fixed stand + lamp; calibration on live good frames; a scale check of the first frame against training statistics |
| Memory with Llama later (1.4 GB) + vision server + Chrome | Measure peak RAM now; the vision server holds only the selected product's anomaly model + YOLO |
| Phone Wi-Fi debugging drops / the Termux session already runs the capture server | Commands are typed into a **new** Termux session (Ctrl+Alt+C); the capture server is left alone |

## Results

### Progress log
- **2026-11-22 - A done: `vision_server` works on the phone** (ADR-022 accepted).
  - **Build** (`edge/vision_server/build_android.sh`): ExecuTorch v1.5.0 for Android arm64 with QNN + XNNPACK in one tree (install `$WORK/et-android-vision`), then the 6 MB `vision_server` against it. Link issues fixed (P-64).
  - **YOLO:** must be exported with Ultralytics' ExecuTorch exporter; our wrapper export failed at run time (P-65).
  - **Phone test** (`edge/scripts/test_vision_server.py`, run from the Termux app; `spike/results/vision_server_test.json`):
    - outputs **identical** to `qnn_executor_runner` (4/4, diff 0.0); YOLO = laptop fp32 (diff 0.0009);
    - all three models resident at once (switching products needs no reload); load 222 ms first NPU model, then 16-27 ms;
    - NPU 28.4 ms p50; YOLO 22 ms; round trip incl. IPC 34 / 27 ms (IPC ~6 ms after zero-copy, was 14);
    - RSS ~37 MB, flat over 1,000 calls.
  - **Model time per inspection:** wood ~61 ms, fabric ~165 ms (4 tiles), well under the 0.5 s target.
- **2026-11-22 - B started: shared preprocessing** (`edge/app/pipeline/preprocess.py`).
  - The parity check first failed for fabric (OpenCV `INTER_AREA` vs PIL bilinear: AUROC 0.877 vs 0.918, P-63).
  - With Pillow (same function as training), inputs are **bit-identical** and AUROC is identical for both products (`edge/scripts/parity_preprocess.json`).
- **2026-11-22/23 - B, C, D done; E (replay) done.**
  - **Pipeline** (`edge/app/`): vision runtime over `vision_server`, fusion (anomaly regions + YOLO, P-67 fixes), rule-based severity from `config/severity.toml` (**draft awaiting user review**), records, per-product calibration (ADR-021 rule with a variety prompt), SQLite + images/heatmaps/crops, CSV/JSON export, aiohttp server, operator page `edge/web/index.html`.
  - **Configs are TOML:** Python's built-in `tomllib`, because PyYAML isn't installed in Termux.
  - **Tests:** 29 unit tests pass (`tests/test_edge_pipeline.py`): severity boundaries, fusion, YOLO decode/NMS, calibration, record/SQLite round trips, inspector end to end with a fake vision runtime.
  - **HTTP end to end on the phone** (laptop → adb forward → phone server; full 3072×4096 frames + ROI, like the browser):
    - calibrate (12 good wood frames → threshold 0.169), inspect, log, CSV/JSON export, heatmap/crop files all work;
    - verdicts: good PASS ×2 (+1 YOLO-only false "liquid" → REWORK), hole SCRAP, liquid HOLD, colour REWORK, scratch HOLD;
    - server time 180-260 ms per inspection.
  - **Replay on the phone** (`edge/scripts/replay.py`, `spike/results/replay_w4.json`; full pipeline on the own val/test photos):

    | | Wood | Fabric |
    |---|---|---|
    | AUROC phone (laptop 16a8w) | **1.000** (1.000) | **0.914** (0.917), within 0.01 ✅ |
    | Phone-calibrated threshold (val good) | 0.111 (5 near-identical frames, too tight) | 0.258 (20 frames) |
    | False alarms / recall at that threshold | 7/15* / **51/51** | **1/20** / 64/85 (thread 10/30) |
    | Server time p50 / p95 | **185 / 207 ms** | **324 / 339 ms** |
    | of which NPU / YOLO (CPU) / file saving | 29 / 36 / 58 ms | 116 / 41 / 51 ms |

    \*The good test scores are 0.053-0.153. With the varied 12-frame calibration from the HTTP test (0.169), all 15 pass, which is why the Calibrate button now asks the operator to shift the item every 5 frames (P-67).
  - **Problems fixed along the way:** P-63 (resize parity), P-64 (link), P-65 (YOLO export), P-66 (incomplete pushes → `termux_sync.sh`), P-67 (region inflation, strong bump, calibration variety).
  - **Severity reviewed by the user (2026-11-24):** colour marks capped at REWORK; YOLO-only detections are notes, not verdicts (ADR-006 update). 31 tests pass.
  - **Live fabric session 1 (2026-11-24, user):** 14 inspections, 3 calibrations (P-68).
    - Camera distance varied 3× (loop period 13-47 px vs 12.6-34.1 in training); a black frame was judged; Termux processing ran 2-6× slower while Chrome had the camera (server 0.72-1.1 s instead of ~0.33 s).
    - The colour mark was found correctly.
    - **Frame check added:** too dark / fabric too close or too far → "retake", not judged, not logged. 34 tests pass.
  - **Still to do:** fabric session 2 at a fixed height; wood session; A/B speed check (page open vs closed); final week summary.
