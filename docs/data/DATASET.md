# Data Card

Datasets used by the Edge Quality Inspector, how they were prepared, and what to watch out for. Decisions: [ADR-004](../decisions/ADR-004-dataset-and-product-selection.md), [ADR-016](../decisions/ADR-016-demo-products-wood-fabric.md), [ADR-017](../decisions/ADR-017-dataset-structure-and-splits.md).

## 1. MVTec AD (public)

| | |
|---|---|
| Source | MVTec Software GmbH, <https://www.mvtec.com/company/research/datasets/mvtec-ad> (official per-category archives) |
| Licence | **CC BY-NC-SA 4.0**: non-commercial use only, attribution required, derivatives under the same licence |
| Citation | Bergmann et al., *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019; and *The MVTec Anomaly Detection Dataset*, IJCV 2021 |
| Categories used | 9 of 15: cable, carpet, grid, leather, metal_nut, screw, tile, transistor, wood |
| Archives | 3.54 GB total; sha256 per archive in [`training/data/reports/mvtec_manifest.json`](../../training/data/reports/mvtec_manifest.json) |
| Downloaded | 2026-11-01 with `training/data/download_mvtec.sh` (archives deleted after verified extraction) |

### Contents (per category)

| Category | train/good | test/good | Defect types (test images) |
|---|---|---|---|
| cable | 224 | 58 | bent_wire 13, cable_swap 12, combined 11, cut_inner_insulation 14, cut_outer_insulation 10, missing_cable 12, missing_wire 10, poke_insulation 10 |
| carpet | 280 | 28 | color 19, cut 17, hole 17, metal_contamination 17, thread 19 |
| grid | 264 | 21 | bent 12, broken 12, glue 11, metal_contamination 11, thread 11 |
| leather | 245 | 32 | color 19, cut 19, fold 17, glue 19, poke 18 |
| metal_nut | 220 | 22 | bent 25, color 22, flip 23, scratch 23 |
| screw | 320 | 41 | manipulated_front 24, scratch_head 24, scratch_neck 25, thread_side 23, thread_top 23 |
| tile | 230 | 33 | crack 17, glue_strip 18, gray_stroke 16, oil 18, rough 15 |
| transistor | 213 | 60 | bent_lead 10, cut_lead 10, damaged_case 10, misplaced 10 |
| wood | 247 | 19 | color 8, combined 11, hole 10, liquid 10, scratch 21 |

### Prepared datasets (`training/data/prepare_mvtec.py`)

Stored on the WSL filesystem (`~/bharatai-build/datasets/`, on drive D:) as symlinks to the raw images. Full statistics: [`training/data/reports/data_stats.md`](../../training/data/reports/data_stats.md).

**YOLO detection (`yolo_mvtec/`)**
- **48 classes:** 44 MVTec `<category>_<defect>` classes plus 4 reserved for our own fabric photos (`fabric_color`, `fabric_cut`, `fabric_hole`, `fabric_thread`). Class ids are stable when own data is added.
- **Images:** train 811 / val 181 / test 296, including defect-free negatives.
- **Boxes:** 996 in total (train 599 / val 193 / test 204). Per class: 10 (transistor_misplaced) to 59 (grid_thread).
- **Splits:** defects 60/20/20 per category+type (seed 42); `test/good` half → final evaluation, half → negatives; plus up to 30 `train/good` per category as background images.
- **Excluded:** `combined` defect types (mixed defects). They're kept for anomaly evaluation only.
- **Mask → box:** one box per connected component; 1 speck below 0.005% of the image was dropped; 0 masks missing.

**Anomaly (`anomaly/<category>/`)**
- `train/good`: all MVTec training images.
- `test/good`: the evaluation half of `test/good`.
- `test/<defect>`: the same held-out defects as the YOLO test split, plus all `combined`, with matching `ground_truth/`.

### Quality checks
- Box overlays for every category: `training/data/reports/qa_boxes_<category>.jpg`, all reviewed. Boxes match the defects for textures, screws, cables and transistors.
- Ultralytics loaded every label with **0 corrupt files** and trained for one epoch (`training/data/sanity_yolo.py`).

### Known caveats
- **Long, thin defects** (wood scratches, carpet threads) get large boxes that are mostly background. The anomaly map captures the precise shape.
- **A gap in a mask splits one defect into several boxes** (e.g. carpet thread, cable cut_outer_insulation).
- **`transistor_misplaced`** where the part is missing marks the *empty* board area. That's correct per MVTec, but a different kind of target.
- **Per-class data is thin** (6-15 training images per defect class). Expect augmentation and the anomaly safety net to matter.
- **Domain gap:** MVTec's controlled studio photos differ from phone photos. The live demo relies on our own photos (section 2).

## 2. Own photos (wood, fabric)

| | |
|---|---|
| Products | **wood** (one light-coloured board; MVTec `wood` classes) and **fabric** (one plain looped/towel-like fabric; `fabric_*` classes using MVTec carpet's defect names), per [ADR-016](../decisions/ADR-016-demo-products-wood-fabric.md) and its 2026-11-20 update |
| Device | OnePlus CPH2613 rear camera via the capture tool (`training/data/capture_tool/`); 3072×4096 frames, square ROI 2458 px |
| Collected | 2026-11-20/21, 5 sessions, improvised book rig |
| Labelling | Local Label Studio 1.23, boxes drawn by the project member; exported and validated (`export_annotations.py`): 121/121 defect photos labelled, every box label = its folder |
| Privacy | Photos of materials only; no people or identifying information |

### Curation (all logged, nothing deleted)
- `training/data/reports/own_exclusions.csv`: **210 photos set aside**, each with a reason:
  - printed (busy) fabric; older white/grey fabrics; other wood boards (grey, dark);
  - pencil marks around holes; peeled patch; board-edge "scratch";
  - photos with shadows or background in frame.
- `training/data/reports/own_relabels.csv`: **5 photos relabelled** (a blue ink stroke saved as `liquid` → `color`, user-confirmed).
- Materials were told apart by capture time + ROI colour statistics (`photo_stats.py`); contact sheets are in `training/data/reports/own*.jpg`.

### Final counts (used photos)
| | good | scratch | hole | liquid | color | cut | thread |
|---|---|---|---|---|---|---|---|
| wood | 55 | 4 | 12 | 20 | 15 | — | — |
| fabric | 50 | — | 15 | — | 20 | 20 | 15 |

Boxes: wood 56 kept (10 lying outside the ROI were dropped; no photo lost all its boxes), fabric 70.

### Preparation and splits (`training/data/merge_own_data.py`)
- Each photo is **cropped to the capture square** (what the models see), resized to 1024 px, with boxes converted and clipped.
- **Physical-defect groups:** the capture tool's instance id, split further when shots are > 20 s apart. Good photos are grouped into position bursts (> 4 s gap). Splits are by group, so near-duplicate shots never straddle splits.

| | wood | fabric |
|---|---|---|
| Defect groups | color 3, hole 2, liquid 3, scratch 1 | color 1, cut 1, hole 2, thread 1 |
| YOLO real-object **test** groups | 1 color, 1 hole, 1 liquid | 1 hole |
| Good: train / val / test | 35 / 5 / 15 | 35 / 5 / 10 |

- Defect types with only one physical defect stay in YOLO training: fabric classes have no MVTec equivalent, and wood scratch also has MVTec examples.
- **The anomaly model never trains on defects, so every real defect photo is in its test set** (`anomaly_own/<product>/test/<defect>/`, boxes in `test_boxes.json`).
- Outputs:
  - `datasets/own/` (YOLO; good photos as backgrounds);
  - `datasets/yolo_combined/data.yaml` (MVTec + own, shared 48 class ids);
  - `datasets/anomaly_own/<product>/`.
- Checks: box overlays `training/data/reports/qa_own_{wood,fabric}.jpg` were reviewed and are accurate. The combined dataset loads in Ultralytics with **0 corrupt** labels and trains for one epoch. Per-photo split: `training/data/reports/own_splits.csv`.

### Session 6 (2026-11-22): close-distance fabric (P-51)
- **Why:** the earlier fabric cut/thread/hole photos were taken ~1.6× closer than the good photos (texture-period measure: loop period ~25-27 px vs 15-18 px in the 1024 crop). That confounded the anomaly test.
- **Captured:** 70 good + 15 thread (a second physical thread defect, instance 2), 12:45-12:50, same frame/ROI. Pulled and verified (170 files); phone cleared.
- **Checked:**
  - Loop period: new good median 22 px (13-33 after curation), new thread 21 px. So the defect distance is now covered by good photos.
  - Sharpness (variance of the Laplacian) flagged 3 bursts.
- **Curated (user decisions):**
  - **15 good set aside**: 3 bursts held closer than the camera's minimum focus distance (sharpness 31-74 vs 150-900).
  - **10 creased good kept**: the live fabric won't be perfectly flat; the cost to thread recall is measured.
- **Counts now:** fabric good 105, thread 30.
- **Splits (stable, ADR-017 update):** old photos keep their splits. New good 30 train / 15 val / 10 test; the new thread group is fabric thread's test group.
- **Labelled and merged (2026-11-22):** export validated (fabric 85/85 tasks, 30 thread boxes). Fabric anomaly set: train 65 / val 20 good; test 20 good + color 20, cut 20, hole 15, thread 30 (85 boxes). YOLO own: 199 train / 25 val / 72 test images.

### Known limitations
- **Few physical defects per type** (1-3), so YOLO's real-object test covers only wood color/hole/liquid and fabric hole. The other types are evaluated through the anomaly model and MVTec.
- Good photos come from a handful of positions (5-6 groups per product), so the anomaly model's "normal" covers a limited area of each material.
- Lighting varied between sessions (midday vs evening). Expect brightness augmentation to matter.
