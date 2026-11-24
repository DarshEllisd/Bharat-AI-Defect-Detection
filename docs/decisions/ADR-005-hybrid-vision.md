# ADR-005: Hybrid vision: YOLO detector for named defects + anomaly model for unseen ones

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0

## Context
The brief requires detecting defects and **classifying their type**. MVTec has few defect images, and real defects can look unlike anything seen in training.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Supervised detector only (YOLO11n) | Gives class, box and confidence directly; fast | Misses defect types it hasn't seen; needs labelled defects |
| Anomaly detection only (PatchCore / EfficientAD) | Needs only good samples; catches novel defects | Gives a heatmap and score, no defect class names |
| **Hybrid** | Named classes *and* a safety net for unknown defects | More models, plus fusion logic |

## Decision
Run both:
- **YOLO11n** for the known defect classes.
- **EfficientAD-S** (one per category) for an anomaly map and score.

**Fusion:** threshold the anomaly map into regions and match them to YOLO boxes by IoU (overlap).
- Region with a matching box → classified defect.
- Region without a box → `unknown_anomaly`.
- Confident box without an anomaly region → kept, flagged `detector_only`.

## Why
Best coverage for the "detection quality: misses vs false alarms" criterion, and it degrades gracefully: unseen defects still get flagged.

## Trade-offs and consequences
- Two models per frame. The latency budget was validated in week 1: YOLO11n at 320 px takes 32 ms on the CPU and 1.81 ms on the NPU (int8).
- **The NPU path is int8-only and limited to supported ops** ([ADR-012](ADR-012-npu-vision-cpu-llm.md)). EfficientAD must be checked for QNN compatibility early in week 3.
- More tuning (two thresholds plus the IoU rule). This is planned for week 7.

## Evidence / references
- `PLAN.md` → Hybrid fusion
- `spike/results/W1_RESULTS.md` → vision and NPU sections
