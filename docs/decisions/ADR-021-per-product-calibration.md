# ADR-021: Per-product calibration on the phone, saved until redone

- **Status:** Accepted
- **Date:** 2026-11-22
- **Week:** 4 (planning)

## Context
Thresholds chosen on laptop validation data don't reliably carry over to live use:
- **MVTec carpet:** AUROC 0.999 but 12/28 false alarms at the val threshold, because val came from the training distribution (P-56).
- **NPU vs laptop:** the phone's NPU scores run ~15-20% below the laptop's simulation of the same quantised model (week-3 phone run).
- **Conditions:** live lighting and placement will differ from the capture sessions.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Calibrate per product, save, re-calibrate on demand** (chosen) | Threshold comes from the deployed model's own outputs under real conditions; ~30 s once per product | Stale if lighting changes and nobody re-calibrates |
| Calibrate at every session start | Always current | ~30 s of setup every start |
| Laptop thresholds by default, calibrate only when needed | No setup | Known to be wrong on the NPU (score offset) and under new conditions |

## Decision
- The operator **calibrates each product once**: shows ~20 good frames.
- The threshold is set from the **deployed (NPU) model's** image scores with the same rule as ADR-019 (99th percentile), plus a small safety margin from the score spread.
- It is saved per product (`state/calibration.json`: threshold, date, number of frames, score statistics) and reused until the operator presses **Re-calibrate**.
- The UI shows the calibration's age and frame count. Inspecting a product that has never been calibrated isn't allowed (or it is clearly marked "uncalibrated").

## Why
- This is the user's choice (2026-11-22).
- It applies ADR-019's rule ("thresholds from good data, never from test") on in-condition data from the actual deployed model, which addresses both P-56 and the NPU score offset.

## Trade-offs and consequences
- With only ~20 frames the threshold is noisy. The margin and score statistics are logged, and the week-4 test session measures live false alarms and misses.
- Calibration frames must be truly good items. The UI asks for a good item and shows the frames.
- Each calibration is a logged event, so an inspection can be traced to the threshold in force.

## Update 2026-11-23 (implementation and phone replay)
- **Rule as implemented:** threshold = p99 of the good-frame scores + max(1 × std, 10% × p99). The 10% floor exists because near-identical frames on a fixed stand give std ≈ 0, and the threshold would then equal the highest calibration score (found by a unit test).
- **Frames must vary.** 5 near-identical good frames (one burst) gave a wood threshold of 0.111 and 7/15 false alarms. 12 varied frames gave 0.169 and 0/15. The Calibrate button therefore pauses every 5 frames and asks the operator to shift or rotate the good item (20 frames, ~26 s).
- Minimum frames: 10 for the button (it takes 20). Replay only: 5 (wood has just 5 val photos).

## Evidence / references
- P-56; `docs/models/MODELS.md` §4 (on-phone NPU) and §6; ADR-019
