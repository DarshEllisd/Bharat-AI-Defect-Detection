# Week 00: Problem statement and planning

- **Dates:** 2026-11-01
- **Plan reference:** PLAN.md (whole document)

## Goals
- Turn the AIBharat PS5 brief (PDF transcript + slides) into a clear problem statement.
- Agree on the target hardware, scope and architecture before writing any code.

## Work done
- Wrote `PROBLEM_STATEMENT.md` from the PDF: the five capabilities (detect, classify, alert, log, run at the edge), domains and hardware, the reference architecture, the two-pipeline Hailo note, required/optional tools and starter datasets, deliverables, and the evaluation criteria.
- Ran a structured Q&A to settle the open choices, then wrote the 8-week plan (`PLAN.md`).

## Decisions made
| Topic | Decision | Record |
|---|---|---|
| Hardware | Android phone (later identified as OnePlus CPH2613, SD 7 Gen 3) | [ADR-001](../decisions/ADR-001-target-device.md) |
| Runtime host | Termux + local web dashboard; native Android app later | [ADR-002](../decisions/ADR-002-phase1-host-termux-web-ui.md) |
| Camera | Chrome `getUserMedia` → WebSocket | [ADR-003](../decisions/ADR-003-camera-ingest.md) |
| Data | MVTec AD, 9 categories; operator selects product; real objects for the demo | [ADR-004](../decisions/ADR-004-dataset-and-product-selection.md) |
| Vision | Hybrid YOLO + anomaly | [ADR-005](../decisions/ADR-005-hybrid-vision.md) |
| Severity | Rule-based | [ADR-006](../decisions/ADR-006-rule-based-severity.md) |
| LLM | Llama 3.2 1B, 4-bit, ExecuTorch; English; grounded JSON | [ADR-007](../decisions/ADR-007-llm-and-advice.md) |
| Logging | SQLite + CSV/JSON + crops | [ADR-008](../decisions/ADR-008-logging.md) |
| Training | Laptop NVIDIA GPU (RTX 4050) | [ADR-009](../decisions/ADR-009-laptop-toolchain.md) |
| Team / time | Solo, comfortable with Python/ML; ~8 weeks | — |

## Key risks identified
- ExecuTorch on Termux (bionic libc) → week-1 test of two routes.
- Domain gap from MVTec to real objects → own good-sample capture plus a fixed rig.
- Thin MVTec defect data for YOLO → augmentation, self-made defects, the anomaly safety net.
- 1B LLM quality → knowledge-base grounding, validation, a template fallback.

## Next week
- The week-1 de-risking tests on the real phone (`spike/`).
