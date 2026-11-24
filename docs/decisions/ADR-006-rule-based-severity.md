# ADR-006: Severity is computed by deterministic rules; Llama explains it but doesn't decide it

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0

## Context
The brief requires classifying defect **severity**. MVTec has no severity labels.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Rule-based scoring** | Deterministic, explainable, unit-testable, instant | Hand-tuned weights |
| Let Llama decide | Flexible | A 1B model is inconsistent; slow; hard to test |
| Learned severity model | Data-driven | No labels exist; we'd have to invent them |

## Decision
`severity = class criticality × area fraction × anomaly score × zone weight` (weights in `config/severity.yaml`), mapped to **LOW / MEDIUM / HIGH / CRITICAL** and a disposition (**PASS / REWORK / HOLD / SCRAP**).

## Why
- The alert can be raised immediately, without waiting for the LLM.
- The same input always gives the same verdict, which the operator can trust and the tests can check.
- Week 1 showed that Llama copies the given severity rather than changing it, which suits this split.

## Trade-offs and consequences
- The weights need tuning per category (for example, a scratch on a screw's thread is worse than one on the head).
- Llama receives severity as input and is told not to change it.


## Update 2026-11-24: rule table implemented and reviewed (week 4)
- **Implemented** in `edge/config/severity.toml` + `edge/app/pipeline/severity.py` (31 unit tests):
  - level = defect type's base level (1 LOW … 4 CRITICAL) + 1 if large (≥ 2% of the square) + 1 if strong (peak ≥ 10× the calibrated threshold), capped at 4;
  - the item takes its worst defect;
  - disposition: LOW/MEDIUM → REWORK, HIGH → HOLD, CRITICAL → SCRAP, none → PASS.
  - Every level carries its reasons, which are shown in the UI and stored in the log.
- **User decisions:**
  1. **Colour marks are capped at MEDIUM (→ REWORK)** on wood and fabric. They're cleanable, so never HOLD/SCRAP on their own. Before this, every large dye mark on fabric ended up HOLD.
  2. **YOLO-only detections are notes.** They're shown with severity "NOTE" and their reason, but never change the verdict. In testing, YOLO alone flagged a good wood board as "liquid".
- The strong-bump ratio was raised from 3× to 10× after the phone replay showed median defect peaks of 15-44× the threshold (P-67).

## Evidence / references
- `PLAN.md` → Key design choices
- `spike/results/W1_RESULTS.md` → output quality
