# ADR-013: Vision models use 320 px input; the browser sends cropped frames

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
Input resolution drives latency, both on the device and in the browser's encoding step.

## Options considered
320 px vs 640 px model input. Full camera frames vs a region-of-interest crop.

## Decision
- **320×320 model input** (up to 416 if accuracy needs it).
- The browser sends a **~320-480 px crop of the inspection area** rather than the full portrait frame.

## Why
Measured in week 1:

| | 320 px | 640 px |
|---|---|---|
| YOLO11n fp32, CPU, 4 threads | **32.2 ms (31 FPS)** | 140.7 ms (7 FPS) |
| Camera link round trip p95 | 10.2 ms | 19.9 ms |
| Browser JPEG encode | 13.7 ms | 19.7 ms |

640 px costs about 4.4× more, tracking the 4× pixel count. On the NPU, 320 px int8 takes 1.81 ms.

## Trade-offs and consequences
- Small defects may need a tighter crop rather than higher resolution, so a fixed rig and region of interest matter ([ADR-004](ADR-004-dataset-and-product-selection.md)).
- Revisit if week-3 accuracy at 320 px is too low.

## Evidence / references
- `spike/results/W1_RESULTS.md` → vision, camera link
