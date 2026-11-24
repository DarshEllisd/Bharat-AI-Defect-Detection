# ADR-016: Live-demo products are wood and fabric (two textures)

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 2

## Context
The demo uses real physical objects ([ADR-004](ADR-004-dataset-and-product-selection.md)). The user has screws/nuts, cables and textures (fabric, leather, wood, tile) available, but only **2-3 hours** of capture time, with an **improvised rig** (books/box). That's enough to do about two products properly.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Screw + wood | One object + one texture; screw defects are easy to make | Screws must be placed in the same orientation every time; MVTec's hardest category |
| Screw + fabric | Same, with fabric | Same screw issues |
| Metal nut + wood | Nut is rotation-symmetric (easier placement) | Real nuts differ from MVTec's nut type |
| Screw + cable | Two objects | Cables are hard to place identically, so more false alarms |
| **Wood + fabric** (user's choice) | Textures fill the frame, so framing errors from an improvised rig matter little; anomaly detection is strongest on textures; defects are easy to make and photograph | No discrete object in the live demo |

## Decision
Live-demo products: **wood** (MVTec `wood`) and **fabric** (closest MVTec category: `carpet`). The other 7 categories are covered through MVTec images (replay and metrics) only.

## Why
- The user's choice, and well matched to the constraints. With textures, the region of interest is simply "the material", so small camera or position shifts don't create false anomalies the way a shifted screw would.
- Self-made defects map directly onto MVTec defect types:
  - **wood:** scratch, hole, liquid, color;
  - **fabric:** cut, hole, color, thread (MVTec carpet's names).

## Trade-offs and consequences
- The demo shows texture inspection only. Object inspection (screw, nut, cable, transistor) is shown through MVTec replay.
- Fabric must lie flat: folds and wrinkles look like defects. Use a stiff backing (cardboard) and tape the edges.
- Real fabric differs from MVTec carpet, so the fabric anomaly model is trained mainly on our own good photos. MVTec carpet adds defect examples for YOLO.
- Revisit if time allows: an object product (metal nut) could be added in week 7.

## Update 2026-11-20: material choices after the first capture sessions
- **Demo board: the light wood board.** Photos of a grey and a dark board are excluded, so "normal" stays precise.
- **Fabric: plain (single-colour) fabrics only.** A busy printed fabric was tried and **dropped**: the anomaly model would have to learn every part of the print as normal, which adds noise and hides small stains or holes. MVTec carpet is a uniform weave too.
- Exclusions are documented per photo in `training/data/reports/own_exclusions.csv` (`training/data/curate_own.py`).

## Evidence / references
- `docs/weekly/week-02-data.md` → shot list
