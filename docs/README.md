# Project Documentation

Engineering record for the **Edge Quality Inspector** (AIBharat Problem Statement 5). It records what we decided and why, what each week produced, and what went wrong along the way and how it was fixed. It feeds the final technical report.

| Folder / file | What it holds |
|---|---|
| [`decisions/`](decisions/) | **Decision records (ADRs)**: one file per decision, with context, options considered, the choice, why, trade-offs and evidence |
| [`weekly/`](weekly/) | **Weekly log**: goals, work done, results, open items for each week |
| [`models/MODELS.md`](models/MODELS.md) | **Model card**: deployed vision models, training, metrics (fp32, quantised, on-phone NPU), limitations, licences |
| [`problems/PROBLEMS_AND_FIXES.md`](problems/PROBLEMS_AND_FIXES.md) | **Problems and fixes**: symptom, root cause, fix, and where the fix lives |
| [`data/DATASET.md`](data/DATASET.md) | **Data card**: sources, licence, counts, splits, quality checks, caveats |

Related files elsewhere in the repo:
- [`../PROBLEM_STATEMENT.md`](../PROBLEM_STATEMENT.md): the brief
- [`../PLAN.md`](../PLAN.md): the overall 8-week plan (kept in sync with these decisions)
- [`../spike/README.md`](../spike/README.md): week-1 runbook; [`../spike/results/W1_RESULTS.md`](../spike/results/W1_RESULTS.md): raw week-1 measurements

## Decision index

| ID | Decision | Status |
|---|---|---|
| [ADR-001](decisions/ADR-001-target-device.md) | Target device: Android phone (OnePlus CPH2613, Snapdragon 7 Gen 3) | Accepted |
| [ADR-002](decisions/ADR-002-phase1-host-termux-web-ui.md) | Phase 1 runs in Termux with a local web dashboard; native Android app later | Accepted |
| [ADR-003](decisions/ADR-003-camera-ingest.md) | Camera frames: Chrome `getUserMedia` → WebSocket | Accepted |
| [ADR-004](decisions/ADR-004-dataset-and-product-selection.md) | Data: MVTec AD, 9 categories; operator selects the product | Accepted |
| [ADR-005](decisions/ADR-005-hybrid-vision.md) | Vision: hybrid YOLO detector + anomaly model | Accepted |
| [ADR-006](decisions/ADR-006-rule-based-severity.md) | Severity: deterministic rules; Llama explains, doesn't decide | Accepted |
| [ADR-007](decisions/ADR-007-llm-and-advice.md) | LLM: Llama 3.2 1B SpinQuant INT4 via ExecuTorch; grounded JSON advice | Accepted |
| [ADR-008](decisions/ADR-008-logging.md) | Logging: SQLite + CSV/JSON export + defect crops | Accepted |
| [ADR-009](decisions/ADR-009-laptop-toolchain.md) | Laptop toolchain: WSL2, uv Python 3.12, ExecuTorch v1.5.0, NDK r28c | Accepted |
| [ADR-010](decisions/ADR-010-runtime-route-native.md) | ExecuTorch on the phone: native binaries (route B), not Python in proot | Accepted |
| [ADR-011](decisions/ADR-011-web-stack-aiohttp.md) | Web stack in Termux: aiohttp, not FastAPI | Accepted |
| [ADR-012](decisions/ADR-012-npu-vision-cpu-llm.md) | Vision on the Hexagon NPU (int8, SSG2115P stand-in); Llama on the CPU | Accepted |
| [ADR-013](decisions/ADR-013-vision-input-resolution.md) | Vision input 320 px; send cropped frames | Accepted |
| [ADR-014](decisions/ADR-014-wsl-networking.md) | WSL mirrored networking (IPv6) instead of a download proxy | Accepted |
| [ADR-015](decisions/ADR-015-remote-phone-workflow.md) | Drive the phone from the laptop over adb | Accepted |
| [ADR-016](decisions/ADR-016-demo-products-wood-fabric.md) | Live-demo products: wood and fabric | Accepted |
| [ADR-017](decisions/ADR-017-dataset-structure-and-splits.md) | Dataset structure, class names, splits, local labelling | Accepted |
| [ADR-018](decisions/ADR-018-training-setup.md) | Training setup: separate GPU venv, anomalib teacher, 4 anomaly models, YOLO on combined data | Accepted |
| [ADR-019](decisions/ADR-019-evaluation-protocol.md) | Evaluation protocol and operating thresholds (val decides, test reports) | Accepted |
| [ADR-020](decisions/ADR-020-button-trigger-fixed-rig.md) | Button-triggered full-resolution capture on a fixed stand | Accepted |
| [ADR-021](decisions/ADR-021-per-product-calibration.md) | Per-product calibration on the phone, saved until redone | Accepted |
| [ADR-022](decisions/ADR-022-persistent-vision-server.md) | Persistent native vision server, called from Python over a Unix socket | Proposed |

## Weekly log

| Week | Topic | Status |
|---|---|---|
| [00](weekly/week-00-planning.md) | Problem statement and planning | Done |
| [01](weekly/week-01-derisking-spike.md) | De-risking tests on the real phone | Done |
| [02](weekly/week-02-data.md) | Data: MVTec prep, capture tool, own photos, labelling | Done |
| [03](weekly/week-03-training.md) | Model training, evaluation, int8 NPU export | Done |
| [04](weekly/week-04-edge-pipeline.md) | Edge pipeline on the phone (vision server, inspect/calibrate, logging, benchmark) | Mostly done |
| [05](weekly/week-05-advisory.md) | Advisory pipeline: on-device Llama advice grounded in a knowledge base | In progress |
| [06](weekly/week-06-dashboard.md) | Operator dashboard (banner, sound, acknowledge, log, stats, speed) + one-tap start | In progress |

## Conventions

- **New decision:** copy [`decisions/_TEMPLATE.md`](decisions/_TEMPLATE.md) to the next `ADR-NNN-short-name.md` and add it to the index above. Never rewrite history: if a decision changes, set the old one to `Superseded by ADR-NNN` and write a new one.
- **New week:** copy [`weekly/_TEMPLATE.md`](weekly/_TEMPLATE.md) to `week-NN-topic.md`.
- **New problem:** append to `problems/PROBLEMS_AND_FIXES.md` with the next `P-NN` number.
- Keep these docs free of personal or account details (usernames, tokens, network names, device serials, home-directory paths).
