# ADR-019: Evaluation protocol and operating thresholds

- **Status:** Accepted
- **Date:** 2026-11-21
- **Week:** 3

## Context
The brief scores **detection quality (misses vs false alarms)**. Our own real test sets are small, and the thresholds that turn scores into PASS/FAIL decide both numbers. Choosing thresholds on test data would inflate the results.

## Decision
**Data roles:**
- **train:** fit models.
- **val:** early stopping, anomaly-map quantiles, and **thresholds**.
- **test:** reported once, never tuned on.

**Anomaly models (EfficientAD-S):**
- **Image AUROC** (threshold-free) on test: good test vs defect test.
- **MVTec:** pixel AUROC from the masks.
- **Own:** **box hit rate**, the share of defect boxes containing an above-threshold anomaly region.
- **Operating threshold:** set on validation good photos, at the 99th percentile of their image scores (a ~1% target false-alarm rate on good parts). Reported on test as **false-alarm rate** (good flagged) and **recall** (defects caught).

**YOLO11n:**
- **mAP50 and mAP50-95** per class on the MVTec test split and on the own real test set, reported separately.
- **Operating confidence:** the F1-maximising value on validation. Precision and recall at that confidence are reported on test.

**int8 / NPU:**
- The same metrics recomputed with int8 outputs. **Acceptance:** AUROC drop ≤ 0.02, mAP50 drop ≤ 5 points.
- Plus on-phone latency and output agreement (cosine vs the CPU reference).

**Reporting:** every number states its dataset, split and sample size. Small own test sets are flagged as such.

## Why
- Separating validation (decisions) from test (reporting) is the minimum for honest numbers.
- Image AUROC is the standard anomaly metric; false-alarm rate and recall at a fixed threshold are what an operator actually experiences.
- A 99th-percentile threshold on good validation photos directly encodes "rarely bother the operator on good parts". It can be retuned in W7 with more data.

## Trade-offs and consequences
- Own validation sets are small (5 good photos per product). The 99th percentile then effectively equals the maximum. Validation good photos can grow in W7, and MVTec numbers give a sturdier reference.
- The own YOLO test covers only 4 defect types (wood color, hole, liquid; fabric hole).

## Evidence / references
- `docs/data/DATASET.md` (splits), `docs/weekly/week-03-training.md`
