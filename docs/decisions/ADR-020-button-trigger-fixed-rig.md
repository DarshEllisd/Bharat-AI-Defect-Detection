# ADR-020: Button-triggered full-resolution capture on a fixed stand

- **Status:** Accepted
- **Date:** 2026-11-22
- **Week:** 4 (planning)

## Context
- Week 3 showed the models are sensitive to how the image is taken:
  - fabric defect photos taken ~1.6× closer than the good photos confounded the test (P-51);
  - thresholds set on validation data didn't transfer to new conditions (P-56).
- The plan originally proposed "station mode": an automatic inspection when the frame is stable, with live frames streamed over a WebSocket (ADR-003).
- The training photos are **full-resolution 3072×4096 frames** from the capture tool, cropped to a 2458 px square.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Fixed stand + Inspect button** (chosen) | Fully predictable, exactly one record per press; each press grabs a full-resolution frame through the **same code path as the training photos**; easy to explain in the demo | Operator must press a button per item |
| Auto-trigger when stable (station mode) | Hands-free | Needs a stability detector; risk of double or missed records; inspects video-resolution frames unless a full-resolution grab is added anyway |
| Continuous per-frame inspection | Live feedback | Many near-duplicate results; hard to keep one record per item; more power and heat |
| Handheld phone | Flexible | Distance and angle vary, which week 3 showed causes misses and false alarms |

## Decision
- **Rig:** the phone sits in a **fixed stand at one height** with the **same lamp** each time, and the item spot is marked.
- **Trigger:** inspections are triggered by an **Inspect button**.
- **Capture:** each press grabs a **full-resolution frame** with the capture tool's method (`getUserMedia` at ideal 4096 px, crop to the same square ROI). Only the square is sent to the backend.
- **Preview:** the live preview stays (framing square shown), but nothing is inspected or logged without a press.

## Why
- These are the user's choices (2026-11-22).
- Using the exact capture path of the training data removes a whole class of train/live mismatch: resolution, ROI and camera processing.
- One press = one record keeps logging and severity simple and testable.

## Trade-offs and consequences
- No hands-free mode for now. An auto-trigger could be added later on top of the same pipeline.
- The stand's height must reproduce the training scale. The first test frame is checked against training statistics (texture scale, P-51).
- ADR-003's streaming path is still used for the preview; the inspected image comes from the full-resolution grab.

## Evidence / references
- P-51, P-56 (`docs/problems/PROBLEMS_AND_FIXES.md`); capture tool `training/data/capture_tool/index.html`
- `docs/weekly/week-04-edge-pipeline.md`
