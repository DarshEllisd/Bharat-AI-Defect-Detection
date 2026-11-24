# Week 02: Data (plan)

- **Status:** Done (2026-11-01 → 2026-11-21)
- **Plan reference:** PLAN.md → W2
- **Decisions:** [ADR-016](../decisions/ADR-016-demo-products-wood-fabric.md) (demo products), [ADR-017](../decisions/ADR-017-dataset-structure-and-splits.md) (dataset structure, classes, splits, labelling)

## Goals
1. **MVTec AD** for the 9 categories, downloaded, verified and converted into:
   - a **YOLO detection dataset** (defect masks → boxes);
   - an **anomaly dataset** per category (good images for training, labelled test set).
2. A **phone capture tool** to photograph real objects with labels and defect instance IDs.
3. **Own photos** of the two live-demo products, **wood** and **fabric**: good samples plus self-made defects.
4. Own defect photos **labelled** in local Label Studio and merged into the dataset, with a held-out real-object test set.
5. Stretch: **EfficientAD int8 on the NPU** feasibility check. This retires the biggest week-3 risk early.

## Inputs we already have (from week 1)
- MVTec per-category archives are on the official page; download speed is ~3.9 MB/s (laptop) and ~3.4 MB/s (phone). 9 categories ≈ 3.3 GB compressed, so ~15 min.
- Licence: **CC BY-NC-SA 4.0**. Non-commercial use only, with attribution; recorded in the data card.
- The NPU path is **int8 only** and needs QNN-compatible ops (ADR-012). Real images are needed for int8 calibration in W3, which our own photos and MVTec provide.

## Tasks

### A. MVTec download and preparation (laptop; Claude builds and runs)
| Step | Output |
|---|---|
| `training/data/download_mvtec.sh`: 9 per-category archives from the official links, sha256 recorded, extracted to `~/bharatai-build/data/mvtec/` (Linux filesystem, faster than `/mnt/c`) | Raw MVTec + `mvtec_manifest.json` (file, size, sha256) |
| `training/data/prepare_mvtec.py`: inventory every category (train good / test good / test per defect type / masks) | `data_stats.md` with counts |
| Masks → YOLO boxes: connected components per mask, drop specks below a minimum area, one box per component; class = `<category>_<defect>` | YOLO labels |
| Splits (ADR-017): MVTec test defects 60/20/20 → YOLO train/val/test, stratified per category+defect type, fixed seed. `test/good` split between YOLO negatives and final evaluation | `datasets/yolo_mvtec/{images,labels}/{train,val,test}` + `data.yaml` |
| Anomaly layout per category: `train/good` and labelled `test/` | `datasets/anomaly/<category>/` (symlinks, no copies) |
| QA: grid images of boxes drawn on samples per class, for a visual check of the mask → box conversion | `qa/boxes_<category>.jpg` |

### B. Phone capture tool (Claude builds; served from Termux)
A dashboard page (aiohttp, same pattern as the W1 camera test):
- Product selector (**wood**, **fabric**); label selector (`good` or a defect type); **defect instance ID** (a "new defect" button increments it, so photos of the same physical defect stay together).
- Live preview with a **square region-of-interest guide** to keep framing consistent.
- Single shot and **burst** (e.g. 5 frames), at the camera's highest resolution.
- Saves `~/bharatai/data/raw/<product>/<label>/<timestamp>.jpg`, plus a JSON sidecar (label, instance, ROI, resolution, timestamp). Shows live counts per label.
- `training/data/pull_captures.sh` copies everything to the laptop and checks it.

### C. Capture session (you; ~2.5 h)
**Rig (improvised):**
- Stack books so the phone lies **face down, camera ~20-30 cm above the table**, and doesn't move between shots.
- Use soft, even light (a room light plus a lamp from the side, no direct sun, no phone shadow on the sample). Keep the same light for the whole session.
- The texture should **fill the square guide**. Mark the sample position with tape.

**Shot list per product.** Always photograph good samples **first**, because making defects destroys the material.

| Shots | Wood | Fabric (MVTec "carpet" types) |
|---|---|---|
| **Good** ×120 (move/rotate the piece between shots so different areas are seen) | plain wood, various grain areas | flat fabric, no folds |
| Defect type 1, 3-4 instances × ~3 shots | `scratch` (key/knife) | `cut` (blade) |
| Defect type 2 | `hole` (nail/drill) | `hole` (punch/burn) |
| Defect type 3 | `liquid` (water/oil stain) | `color` (marker/ink stain) |
| Defect type 4 | `color` (marker/paint mark) | `thread` (pulled-out thread) |

That gives about **120 good + ~40 defect photos per product**. Class names match MVTec's so the datasets merge cleanly.

**Time budget:** rig 20 min · wood 60 min · fabric 60 min · buffer 20 min.

### D. Labelling (Claude sets up; you label, ~45-60 min)
- **Label Studio** running locally in WSL (its own venv; nothing leaves the laptop), one project per product, labels = the defect types above.
- Draw boxes on the ~80 defect photos, then export in YOLO format.
- `training/data/merge_own_data.py` merges the photos into the dataset with **instance-level splits** (all photos of one physical defect go to the same split) and holds out **20% of good and defect photos as the real-object test set**, which is never used in training.

### E. Stretch: EfficientAD int8 on the NPU (Claude; laptop + phone)
Export EfficientAD-S (anomalib), compile int8 for SSG2115P, and run it on the phone NPU from inside Termux. If it fails, pick an NPU-compatible anomaly model now (for example a PaDiM/PatchCore variant with a small backbone, or a convolutional autoencoder) rather than discovering it in W3.

### F. Documentation
- Data card `docs/data/DATASET.md`: sources, licence and attribution, counts, class list, splits.
- Update this week log with results; add a P-NN entry for every problem hit.

## Acceptance criteria (end of week 2)
- [ ] 9 MVTec categories downloaded, checksums recorded, counts in `data_stats.md`
- [ ] YOLO dataset + `data.yaml` loads in Ultralytics (1-epoch sanity run on a small fraction)
- [ ] Box QA grids reviewed; no systematic conversion errors
- [ ] Capture tool works on the phone (saves labelled photos + sidecars; pull script verified)
- [ ] Wood and fabric: ≥ 100 good + ≥ 30 defect photos each, labelled
- [ ] Instance-level splits; 20% real-object test set held out
- [ ] Data card written
- [ ] (Stretch) EfficientAD int8 runs on the NPU, or an alternative is chosen

## Risks
| Risk | Mitigation |
|---|---|
| The improvised rig moves between shots | Tape marks; check framing against the square guide; re-shoot if moved |
| Lighting changes during the session | One fixed light setup; capture each product in one sitting |
| Too few defect photos for YOLO | The anomaly model is the safety net; MVTec defects add volume; augmentation in W3 |
| Same defect leaking across splits | Instance IDs from the capture tool; split by instance |
| Phone storage | ~250 photos × ~1-2 MB ≈ 0.5 GB; pull to the laptop and clear after each session |

## Results

### A. MVTec download and preparation: ✅ done (2026-11-01)
- All 9 categories downloaded from the official links (~3.9 MB/s over IPv4), verified with `xz -t`, sha256 recorded (`training/data/reports/mvtec_manifest.json`), extracted; archives deleted.
- `prepare_mvtec.py` output:
  - **48 classes** (44 MVTec + 4 reserved `fabric_*`);
  - YOLO images train 811 / val 181 / test 296 (incl. negatives);
  - **996 boxes**; 0 missing masks; 1 speck dropped;
  - anomaly layout for all 9 categories.
- Box overlays reviewed for every category: correct. Caveats are in the [data card](../data/DATASET.md).
- Ultralytics loads every label (0 corrupt) and trains for one epoch: **acceptance criterion met**.
- **Problems hit:**
  - [P-31](../problems/PROBLEMS_AND_FIXES.md#p-31): the C: drive filled up mid-download and crashed WSL. Fixed by moving WSL to D: (with approval) and clearing temp files; the download script is now crash-safe.
  - [P-32](../problems/PROBLEMS_AND_FIXES.md#p-32): MVTec's archives have read-only directories, which broke re-extraction. Fixed with `chmod -R u+w`.

### B. Capture tool: ✅ done (2026-11-01)
- `training/data/capture_tool/` (aiohttp server + page). It covers product/label selection, defect instance IDs, a square guide, single shot and burst ×5, undo, per-label targets, screen wake-lock and a best-effort camera lock. Each photo gets a JSON sidecar.
- Tested on the phone: save (good and defect), input validation (no instance / wrong label / not a JPEG → 400), counts, undo.
- `pull_captures.sh` tested from WSL: tar over `adb exec-out`, per-file size verification, `--clear`.
- Guide for the session: [`training/data/README.md`](../../training/data/README.md).

### D. Label Studio: ✅ set up (2026-11-01); labelling pending photos
- **Label Studio 1.23.2** in its own venv (`~/bharatai-build/ls-venv`). Local-only credentials in `~/bharatai-build/label-studio/credentials.env` (mode 600, outside the repo). The server listens on 127.0.0.1:8080, reachable from Windows via mirrored networking.
- Projects **"wood defects"** and **"fabric defects"** with the defect labels and instructions. Each defect folder is a local-files storage (photos are served from disk, not copied). `setup_projects.py --sync` imports newly pulled photos.
- Self-test: a test photo was imported, served (HTTP 200, valid JPEG) and cleaned up.
- **Problems hit:** [P-33](../problems/PROBLEMS_AND_FIXES.md#p-33) (WSL came up with no network), [P-34](../problems/PROBLEMS_AND_FIXES.md#p-34) (half-created venv), [P-36](../problems/PROBLEMS_AND_FIXES.md#p-36) (slow first start), [P-37](../problems/PROBLEMS_AND_FIXES.md#p-37) (API token disabled → session login).

### E. EfficientAD-S int8 on the NPU: ✅ feasible (2026-11-01)
- Own export-friendly EfficientAD-S module (`training/vision/efficientad_model.py`, 8.06 M params; PDN-S teacher/student + autoencoder), the same graph W3 will train.
- Int8 PT2E quantisation **calibrated on 32 real MVTec wood images**. Int8 simulation vs fp32: cosine **0.9996**.
- QNN compile for SSG2115P: **123 of 124 nodes delegated** to the NPU (after the P-35 fix).
- Measured on the phone (input: MVTec wood `test/hole` image, 256 px):

| Runtime | Per image | Output check |
|---|---|---|
| CPU, XNNPACK fp32, 4 threads | **1,419 ms** (2 threads: 2,191 ms) | — |
| **NPU, QNN int8, inside the Termux app** | **15.3 ms** (~93× faster) | cosine 0.961 vs int8 CPU simulation ✅ |

- **Conclusion:** EfficientAD is **usable only on the NPU**; the CPU is far too slow for real time. Hybrid budget on the NPU: YOLO11n 1.8 ms + EfficientAD-S 15.3 ms ≈ **17 ms per frame** of model compute.
- **Caveats:** weights were untrained (this checks ops, compilation, execution and latency, not accuracy). The output agreement (0.961) is lower than YOLO's (0.993); untrained maps are nearly flat, which exaggerates small numeric differences. **Re-check with trained weights in W3**, and measure int8 vs fp32 anomaly AUROC.

### C. Capture session 1: ⚠️ partially usable (2026-11-20)
- **149 photos** (12 MP, 3072×4096, ~3.6 MB each) pulled and verified (298 files incl. sidecars). Problems on the way: [P-38](../problems/PROBLEMS_AND_FIXES.md#p-38), [P-39](../problems/PROBLEMS_AND_FIXES.md#p-39).
- Contact sheets of the region the models see: `training/data/reports/own_*.jpg` (`training/data/contact_sheet.py`).

| Set | Photos | Verdict |
|---|---|---|
| wood/good | 55 | ✅ usable (clean grain, fills the frame); many near-identical bursts |
| wood/color | 10 | ½ usable: reddish stain ✅; a large peeled/gouged patch is not "color" |
| wood/hole | 10 | single hole ✅; second set has pencil lines + shadow (confusable with color) |
| wood/scratch | 5 | ❌ shows a broken board edge and a gap between boards, not a scratch |
| wood/liquid | 0 | missing |
| fabric/good | 44 | ❌ white cloth with large folds + a second (grey) fabric mixed in |
| fabric/color | 5 | defect ✅, but a phone shadow and folds in frame |
| fabric/thread | 20 | threads barely visible; folds/shadows |
| fabric/cut, hole | 0 | missing |

- Defect instance IDs were not advanced between different physical defects; they will be regrouped by time/appearance.
- **Top-up session planned:** fabric redone on one fabric taped flat, side lighting, all 4 defect types × 3 instances; wood scratch/liquid/color/hole top-up. Existing photos are kept; unsuitable sets are excluded at dataset build time.

### C (continued). Capture session 2: wood nearly complete, fabric still weak (2026-11-20)
- **271 photos** in total on the laptop (542 files incl. sidecars, all verified). Contact sheets: `training/data/reports/own2_*.jpg` (now labelled with capture times, to separate physical defects).

| Label | Total | Verdict |
|---|---|---|
| wood/good | 95 | ✅ usable; the new ones include **two other boards** (grey, dark walnut-coloured). Pick one demo board |
| wood/liquid | 20 | ✅ 3 clear stains, correctly numbered |
| wood/hole | 22 | ✅ 2 clean holes; 2 sets with a pencil rectangle + shadow (⚠️ colour marks in frame) |
| wood/color | 10 | ⚠️ 1 good instance (reddish stain); need 2 more |
| wood/scratch | 5 | ❌ still only the board-edge photos; need 3 real scratches |
| fabric/good | 54 | ❌ three different fabrics mixed (white folded, grey, printed); folds, shadows, background at edges |
| fabric/color, cut, hole | 20 / 11 / 14 | ⚠️ defects present on the printed fabric, but among folds, shadows and a busy print |
| fabric/thread | 20 | ⚠️ from session 1 (barely visible) |

- **Open decisions (user):** which wood board is the demo board; which fabric (a plain fabric taped flat is strongly preferred over the busy print).
- **Still to capture:** wood scratch ×3, wood color ×2; fabric (one fabric, flat): ~100 good + cut/hole/color/thread ×3 instances each.
- Defect instance IDs: some cover two physical defects (e.g. hole i3); they will be split by capture-time gaps at dataset build.

### C (continued). Curation (2026-11-20)
- **User decisions:** demo board = **the light wood board**; fabric = **plain fabrics only** (the busy printed fabric is dropped, since it adds noise for the anomaly model). Recorded as an update in [ADR-016](../decisions/ADR-016-demo-products-wood-fabric.md).
- Materials were identified by **capture time + colour statistics** of the square ROI (`training/data/photo_stats.py`): light board (saturation 25-34), grey board (~13), dark board (50-78); white fabric (~13), grey fabric (~26), printed fabric (29-50, varying hue).
- `training/data/curate_own.py --apply` **moved 110 photos** to `raw/_excluded/` (nothing deleted). Each one is listed with its reason in `training/data/reports/own_exclusions.csv`:
  - 50 printed fabric; 40 wood good from other boards;
  - 10 wood holes with pencil marks; 5 peeled patch (not "color"); 5 board-edge "scratch".
- Phone cleared after re-verifying all 542 files on the laptop (`pull_captures.sh --clear`). Phone free space 7.3 → 8.2 GB.

**Usable own photos now:**

| | good | scratch | hole | liquid | color | cut | thread |
|---|---|---|---|---|---|---|---|
| wood (light board) | 55 | **0** | 12 (2 defects) | 20 (3) | 5 (1) | — | — |
| fabric (white + grey plain) | 44 (24 white + 20 grey) | — | **0** | — | 5 (1, white) | **0** | 20 (2: 1 white, 1 grey) |

**Still to capture:** wood scratch ×3, wood color ×2; fabric on **one** plain fabric (taped flat): good, cut ×3, hole ×3, color ×2, thread ×1-2.

### C (continued). Capture session 3: a new plain fabric (2026-11-21)
- **105 photos** of a **new, single plain looped (towel-like) fabric**: good 35, color 20 (blue ink stain), cut 20, hole 15, thread 15. It fills the square, with no folds, harsh shadows or background. All usable.
- Transfer: USB kept dropping under load ([P-41](../problems/PROBLEMS_AND_FIXES.md#p-41)), so the photos came over **wireless debugging**. A false "PULL OK" was found and fixed ([P-40](../problems/PROBLEMS_AND_FIXES.md#p-40)). 210 files verified (`check_pairs.sh`: 105 photos + 105 sidecars); phone cleared (14 GB free).
- Contact sheets: `training/data/reports/own3_fabric_a.jpg`, `own3_fabric_b.jpg`.
- **Pending user decision:** make this looped fabric the demo fabric and set aside the older white/grey fabric photos (64), to keep one fabric = one "normal".
- Notes: more good photos (80-100) would make "normal" more robust; each defect type has ~1-2 physical defects. That's fine for the anomaly model, which never trains on defects (all real defects can be test data).
- **Wood still needs:** scratch ×3, color ×2.

### C (continued). Curation decisions + session 4 (2026-11-21)
- **User decision:** demo fabric = **the looped fabric**. 69 older white/grey fabric photos were set aside (one fabric = one "normal").
- **Session 4** (50 photos, 18:21-18:27, evening light):
  - fabric good ×15 ✅;
  - wood good ×15 and scratch ×5 were on a **dark brown board** in low light ❌;
  - a blue ink stroke saved as "liquid" was **relabelled to color** (instance 3, user-approved; logged in `training/data/reports/own_relabels.csv`, with `relabeled_from` in each sidecar);
  - a faint mark kept as color (user);
  - a color burst with shadow + background strip ❌.
- `curate_own.py` gained date-specific rules ([P-42](../problems/PROBLEMS_AND_FIXES.md#p-42)) and logged relabelling. 204 photos are set aside in total; nothing is deleted.

**Usable own photos now:**

| | good | scratch | hole | liquid | color | cut | thread |
|---|---|---|---|---|---|---|---|
| wood (light board) | 55 | **0** | 12 | 20 | 15 (3 instances) | — | — |
| fabric (looped) | 50 | — | 15 | — | 20 | 20 | 15 |

- **Fabric: complete.** **Wood: only scratch is missing** (3 thin scraped lines on the light board, in good light).

### C (continued). Session 5: wood scratch; capture complete (2026-11-21)
- 10 scratch photos of **one scratch on the light board** (colour stats match). The user set aside the 6 with a hand/phone shadow at the edge and kept 4 clean frames. 1 physical scratch is accepted: YOLO also learns scratches from MVTec's 21 wood-scratch examples.
- **Final usable own photos:** wood good 55 / scratch 4 / hole 12 / liquid 20 / color 15; fabric good 50 / cut 20 / hole 15 / color 20 / thread 15. 210 photos set aside, all logged.

### D (continued). Labelling ready (2026-11-21)
- Label Studio running (`ensure_label_studio.sh`; [P-43](../problems/PROBLEMS_AND_FIXES.md#p-43)). Defect photos imported: **wood defects 51 tasks, fabric defects 70 tasks**, matching the photo counts; sample images are served (`check_tasks.py`).
- Next: the user draws boxes → export → merge into the dataset.

### D/F. Labelling, merge, data card: ✅ done (2026-11-21)
- **Labelling:** 121/121 defect photos labelled by the user in local Label Studio. Export validated: no unlabelled/skipped photos, every box label matches its folder, no degenerate boxes.
- **Merge** (`merge_own_data.py`): ROI crops at 1024 px; physical-defect groups from instance ids + time gaps; splits by group.
  - Own YOLO images: train 169 / val 10 / test 47.
  - Anomaly sets: wood train/val/test good 35/5/15, fabric 35/5/10, plus all real defects in test.
  - Combined MVTec + own `data.yaml`.
- **Checks:** box overlays accurate (`qa_own_*.jpg`); combined dataset loads with 0 corrupt labels and trains one epoch.
- Data card updated: [`docs/data/DATASET.md`](../data/DATASET.md) section 2.

## Week 2 summary
| Acceptance criterion | Result |
|---|---|
| MVTec 9 categories, checksums, counts | ✅ |
| YOLO dataset loads + trains (MVTec, and combined with own) | ✅ |
| Box QA reviewed (MVTec + own) | ✅ |
| Capture tool on the phone + verified pulls | ✅ |
| Wood + fabric photos, labelled | ✅ wood 55 good + 51 defects; fabric 50 good + 70 defects (below the ~120 good target; fewer physical defects than planned) |
| Instance-level splits; held-out real test set | ✅ |
| Data card | ✅ |
| Stretch: EfficientAD int8 on the NPU | ✅ 15.3 ms (vs 1,419 ms on the CPU) |

**Carried into week 3:**
- A separate CUDA training venv (the build venv has CPU torch).
- Calibrate int8 on real images.
- Re-check EfficientAD NPU agreement with trained weights.
- Brightness augmentation.
- More good photos would strengthen "normal" if time allows.

### Notes for week 3
- The build venv (`et-venv`) has **CPU** PyTorch (required by ExecuTorch). Training on the RTX 4050 needs a separate training venv with CUDA PyTorch.
