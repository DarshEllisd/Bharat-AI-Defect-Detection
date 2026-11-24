# ADR-004: MVTec AD (9 categories), operator selects the product, demo on real objects

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0

## Context
The brief allows any manufacturing domain and suggests MVTec AD, NEU surface defects or KolektorSDD. The demo has to show detection on something tangible.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Steel surfaces (NEU-DET) | 6 labelled classes with boxes; a natural fit for YOLO | Steel samples are hard to find for a live demo |
| **MVTec AD** | 15 object/texture categories, mostly "good" images plus labelled defects with masks | Few defect images per category; a domain gap to real objects |
| PCB defects | Visually striking | Needs PCBs for a live demo |
| KolektorSDD | Clean crack segmentation | Narrow and binary |

Product identification: **the operator selects the product** vs automatic classification vs both.

## Decision
- **MVTec AD, 9 categories:** screw, metal_nut, transistor, cable, carpet, leather, tile, wood, grid.
- **The operator selects the product** in the UI, which loads that category's models.
- **The demo uses real physical objects**, with MVTec test sets replayed for objective metrics.

## Why
- MVTec's "good-only" training data suits anomaly detection, and its masks can be turned into boxes for YOLO.
- Selecting the product matches real production lines (one station inspects one product type). It needs less memory (one category loaded at a time), gives better accuracy, and avoids an extra classifier that could fail.

## Trade-offs and consequences
- **Domain gap:** MVTec photos don't look like the phone camera's view of real objects. Mitigation: collect 100-200 phone photos of good real objects per demo product in a fixed rig (stand, plain background, steady light) to train the anomaly models, plus some self-made defects for YOLO.
- Products without physical samples are validated through MVTec replay only.

## Evidence / references
- `PLAN.md` → Key design choices, W2
