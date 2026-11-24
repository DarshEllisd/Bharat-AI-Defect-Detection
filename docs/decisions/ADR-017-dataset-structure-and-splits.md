# ADR-017: Dataset structure, class names, splits and labelling

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 2

## Context
Two models are trained from two sources:
- **YOLO** needs boxes with classes; **EfficientAD** needs good images, with labelled test images for evaluation.
- **MVTec** has pixel masks for defects and many good images; **our own photos** need labelling.

Splits must give honest numbers, especially for the real-object demo.

## Options considered
**Class naming.** Product-prefixed (`wood_scratch`) vs a shared defect vocabulary across products (`scratch` everywhere).

**YOLO model scope.** One model for all 9 categories (filtered by the selected product at runtime) vs one model per product.

**Mask → box.** One box per connected component vs one box around the whole mask.

**Labelling own photos.** Local Label Studio vs a custom annotator vs semi-automatic labelling.

**Own-data split.** Random by photo vs by **physical defect instance**.

## Decision
- **Classes:** `<category>_<defect>` using MVTec's defect names; own wood/fabric photos use the same names (fabric uses carpet's names under the `fabric_` prefix).
  - MVTec's `combined` defect type (several defects mixed together) is **excluded from YOLO training**, but kept for anomaly evaluation.
- **One YOLO model for all categories**, filtered by the selected product at runtime. Revisit per-product models in W3 if accuracy or NPU size argues for it.
- **Mask → boxes:** one box per connected component of the mask; drop components below a minimum area.
- **MVTec splits:**
  - test defects → YOLO **60/20/20** train/val/test, stratified by category and defect type, fixed seed;
  - `test/good` → split between YOLO negatives and final evaluation;
  - `train/good` → anomaly training.
- **Own photos:** labelled in **local Label Studio** (nothing leaves the laptop). Split **by defect instance** (the capture tool records an instance ID), and **20% held out as the real-object test set**, never used for training or tuning.
- **Storage:** raw data and derived datasets live on the WSL filesystem (`~/bharatai-build/data`, `datasets/`), with symlinks rather than copies. They stay out of the repo; only scripts, stats and QA images live in the repo.

## Why
- Product-prefixed classes keep the runtime "filter by selected product" simple and avoid merging defects that look different across materials.
- One shared model lets small categories benefit from shared low-level features, and it's one model to deploy.
- Per-instance splitting prevents the same physical scratch, photographed three times, from appearing in both train and test, which would inflate the scores.
- Local labelling respects the confidentiality rules and needs no account.

## Trade-offs and consequences
- A shared model has more classes (~40), so per-class data is thin. Augmentation, plus the anomaly model as a safety net.
- Excluding `combined` loses some defect examples for YOLO.
- Instance IDs must be recorded correctly during capture ("new defect" button).

## Update 2026-11-22: stable splits when sessions are added
- Photos already merged keep their split (`reports/own_splits.csv`); a new capture session's groups are split among themselves with the same rules (good bursts 70/10/20; a defect class's new group becomes its held-out test group if it had none).
- Why: results before and after a new session stay comparable; no photo silently moves from train to test (or the reverse); no stale copies are left in old split folders (the merge doesn't delete).
- Session 6 (close-distance fabric, P-51): fabric good is now 65 train / 20 val / 20 test. The new photos split 30 / 15 / 10, so val and test both include close-distance good photos. The new thread defect (instance 2) is fabric thread's test group.

## Evidence / references
- `docs/weekly/week-02-data.md`
- MVTec AD licence: CC BY-NC-SA 4.0 (attribution in `docs/data/DATASET.md`)
