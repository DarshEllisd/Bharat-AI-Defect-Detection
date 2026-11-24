# Week 03: Model training, evaluation and int8 NPU export

- **Status:** Planned (2026-11-21)
- **Plan reference:** PLAN.md → W3
- **Decisions:** [ADR-018](../decisions/ADR-018-training-setup.md) (training setup), [ADR-019](../decisions/ADR-019-evaluation-protocol.md) (evaluation protocol and thresholds)

## Goals
1. A **GPU training environment** on the laptop (RTX 4050, 6 GB) that doesn't disturb the ExecuTorch build environment.
2. **EfficientAD-S anomaly models** for **own wood, own fabric, MVTec wood, MVTec carpet**.
3. **YOLO11n detector** trained on the **combined MVTec + own** dataset (48 classes) with standard augmentation.
4. **Honest evaluation** on held-out data (MVTec test splits + real-object test sets), with **operating thresholds** chosen on validation data only.
5. **int8 versions for the Hexagon NPU** (QNN, SSG2115P stand-in): calibrated on real images, run on the phone inside the Termux app, accuracy drop vs fp32 measured.

## Inputs (from weeks 1-2)
- Datasets (WSL `~/bharatai-build/datasets/`): `yolo_combined/data.yaml`, `anomaly/<mvtec category>/`, `anomaly_own/{wood,fabric}/` (train/val/test good + all real defects in test, boxes in `test_boxes.json`).
- Export-friendly EfficientAD module: `training/vision/efficientad_model.py`, already proven on the NPU (W2: 15.3 ms, untrained weights).
- NPU path: ExecuTorch v1.5.0 + QAIRT 2.37, int8, SSG2115P contexts, run from the Termux app ([ADR-012](../decisions/ADR-012-npu-vision-cpu-llm.md)).

## Tasks

### A. Training environment (Claude)
- New venv `~/bharatai-build/train-venv`: **PyTorch 2.14 with CUDA** (the same torch version as ExecuTorch's), Ultralytics, OpenCV, scikit-learn. The build venv `et-venv` keeps its CPU torch for export.
- Check: `torch.cuda.is_available()` sees the RTX 4050; a quick throughput test.
- Weights move between environments as plain `state_dict` files.

### B. EfficientAD-S training (Claude; ~30-60 min GPU per model)
- **Teacher:** anomalib's published pretrained **PDN-small** teacher weights, mapped onto our module's layer names. Channel mean/std are computed on each category's training images. Downloaded early, in the background (GitHub is slow on this connection).
- **Student + autoencoder:** the paper's losses (hard-feature student-teacher loss, autoencoder and student-autoencoder losses; optional ImageNet-style penalty), Adam lr 1e-4, batch 1, 256 px. Iterations shortened from the paper's 70k to fit the GPU budget; the validation curve decides.
- **Augmentation** for the small own sets (35 good training photos each): brightness/contrast jitter (lighting varied between sessions), small flips/rotations (textures are orientation-free).
- **Map normalisation:** quantiles from **validation good** images → `set_quantiles()`.
- Models: `own_wood`, `own_fabric`, `mvtec_wood`, `mvtec_carpet`.

### C. YOLO11n training (Claude; ~1-2 h GPU)
- `yolo_combined/data.yaml`, COCO-pretrained `yolo11n.pt`, **320 px**, standard Ultralytics augmentation with a slightly stronger brightness (HSV-v) range, early stopping on validation.
- One model, 48 classes, filtered by the selected product at runtime ([ADR-017](../decisions/ADR-017-dataset-structure-and-splits.md)).

### D. Evaluation + thresholds (Claude) - see [ADR-019](../decisions/ADR-019-evaluation-protocol.md)
- **Anomaly:**
  - image AUROC;
  - MVTec pixel AUROC (masks);
  - own sets: defect-box hit rate;
  - **false-alarm rate on good test photos** and **recall on defect test photos** at the chosen threshold.
- **YOLO:** mAP50 / mAP50-95 on the MVTec test split and the own real test set, per class; precision/recall at the chosen confidence.
- Thresholds are chosen on **validation** data only, never on test.

### E. int8 export + NPU validation (Claude; phone run typed into Termux)
- PT2E int8 with the QNN quantizer, calibrated on ~50-100 real training images per model; compiled for SSG2115P.
- Measure **int8 vs fp32** on the same test sets. Targets: anomaly AUROC drop ≤ 0.02; YOLO mAP50 drop ≤ 5 points. Fallback if exceeded: 16-bit activations (16a8w), which QNN supports.
- Run on the phone from inside the Termux app: per-image latency + output agreement vs the CPU reference (including the W2 EfficientAD agreement of 0.961, re-checked with trained weights).
- CPU/XNNPACK fp32 versions are kept as fallbacks.

### F. Documentation
- Model card `docs/models/MODELS.md`: data, training settings, metrics, thresholds, int8 deltas, latency, limitations.
- Week log results, problems (P-NN), ADR updates.

## Acceptance criteria
- [x] Training venv with CUDA working on the RTX 4050
- [x] EfficientAD-S × 4 trained (full 70k iterations); MVTec wood 0.976 / carpet 0.999 image AUROC; own wood 1.000, own fabric 0.920 (2×2 tiles) with false-alarm rate + recall at threshold
- [x] YOLO11n trained; mAP50 per class on the MVTec test split and the own test set reported
- [x] Thresholds chosen on validation only and recorded (and shown not to transfer reliably, hence the app's calibrate step, P-56)
- [x] Quantised QNN versions compiled (16a8w for the anomaly models; YOLO stays fp32 on the CPU, P-62), run on the phone NPU from the Termux app, latency + quantised-vs-fp32 deltas reported
- [x] Model card written: [docs/models/MODELS.md](../models/MODELS.md)

## Risks
| Risk | Mitigation |
|---|---|
| Teacher weights on slow GitHub | Start the download first, in the background; fallback: distil our own teacher from an ImageNet WideResNet (fast PyTorch download) |
| Only 35 good training photos per own product, so the student/autoencoder overfit | Augmentation, shorter training, validation-curve stopping; collect more good photos if results demand it |
| int8 hurts the anomaly maps (squared differences are sensitive) | Calibrate on real images; measure the delta; 16a8w fallback |
| 6 GB VRAM | EfficientAD-S batch 1; YOLO11n at 320 px batch ~32 fits |
| Few real defects for YOLO's real test (4 types) | Report it honestly; MVTec test covers the rest; the anomaly model covers all real defects |

## Results

### Progress log
- **2026-11-21:**
  - **A done.** `train-venv`: torch 2.14.0+cu130, torchvision 0.29.0, Ultralytics 8.4.216; the RTX 4050 is visible (6.0 GiB, 12.3 TFLOPS fp16 matmul).
  - **Teacher.** Verified (sha256) and loads strictly after the P-44 fix.
  - **Penalty set.** Imagenette (320 px, 9,469 train images) downloaded; a fixed pool of 3,000 is used as the ImageNet-style penalty set.
  - **MVTec val splits.** wood: 222 train / 25 val; carpet: 252 train / 28 val (`make_anomaly_val.py`, seed 42).
  - **End-to-end smoke test** (600 iterations, MVTec wood): image AUROC 0.948, so training, map normalisation, threshold and evaluation work together.
  - **Training time** was underestimated (P-45). bf16 autocast brings all four models at the paper's 70k iterations to ~4 h, so the plan's "shortened schedule" isn't needed. Instead, an overfitting monitor logs the mean student-teacher and student-autoencoder discrepancy on val-good vs a train subset every 5k iterations (`val_log.csv`).
  - **B started:** two GPU lanes (own_wood → mvtec_wood, own_fabric → mvtec_carpet), each evaluated automatically when done.
  - **C started:** YOLO11n, 320 px, hsv_v 0.5, up to 200 epochs, patience 40. Ultralytics analytics turned off (P-47).
  - **D scripts:** `evaluate_efficientad.py` and the evaluation part of `train_yolo.py` (F1-max confidence on val; test reported separately for MVTec and own).
  - **E scripts:** `export_npu_efficientad.py` and `export_npu_yolo.py`. Each does real-image calibration with per-channel weights; int8 vs fp32 is measured with the *same* evaluation code; output is a QNN .pte plus an XNNPACK fallback and phone check files.
    - A test on a 20-iteration (untrained) model compiled fine (145 of 146 nodes on the NPU, 8.4 MB).
    - It also showed a risk: the int8 val threshold collapsed to 0.0. The anomaly-map output range calibrated on good images only may be too coarse. To be checked on the trained models; fallbacks are 16a8w or a wider output range.
  - **Training on the GPU only (P-48):** the EfficientAD input pipeline moved into VRAM and the per-step CPU-GPU syncs were removed. That gives one process at ~21-23 it/s using about one CPU core. The four models now train one after another (~55 min each).
  - **C done - YOLO11n** (`models/yolo/yolo11n_combined/eval.json`):
    - Training: early-stopped after 107 epochs (best epoch 67), 20 min on the GPU.
    - Operating confidence **0.415**, chosen on val (F1-max).

    | Split | Images | mAP50 | mAP50-95 | P / R at conf 0.415 |
    |---|---|---|---|---|
    | val (MVTec + own) | 191 | 0.701 | 0.436 | 0.78 / 0.62 |
    | **MVTec test** | 296 | **0.709** | 0.453 | 0.78 / 0.64 |
    | **own real test** | 47 | **0.634** | 0.360 | 0.99 / 0.48 |

    - MVTec test by category (mean mAP50 over its defect classes): tile 0.90, wood 0.88, carpet 0.83, leather 0.81, cable 0.74, transistor 0.65, metal_nut 0.59, grid 0.56, screw 0.41.
    - Weakest MVTec classes: transistor_bent_lead 0.00, screw_thread_side 0.00, cable_poke_insulation 0.07, screw_manipulated_front 0.10. These are tiny, subtle or geometry-only defects at 320 px, which is why the anomaly model sits alongside YOLO ([ADR-005](../decisions/ADR-005-hybrid-vision.md)).
    - Own test per class (mAP50): wood_liquid 0.91, wood_hole 0.84, fabric_hole 0.80, **wood_color 0.00**. The faint colour marks are missed by YOLO. Only 4 defect classes appear in the own YOLO test split.
    - High precision, low recall on own photos: when YOLO flags something it is almost always right, but it misses about half the boxes. The demo relies on the anomaly model to catch "something is wrong" and on YOLO to name the type when it can.
  - **B, first results (before the map-alignment fix P-50; to be re-evaluated):**
    - **own_wood:** image AUROC 1.000; 0/15 false alarms; recall 50/51 (colour 14/15, hole 12/12, liquid 20/20, scratch 4/4); boxes hit 54/56. Optimistic: val/test good are near-identical burst shots, and some hole photos have glare (P-51).
    - **own_fabric:** image AUROC 0.853; 0/10 false alarms; recall 51/70 (colour 20/20, hole 15/15, cut 12/20, thread 4/15). **Confounded by camera distance**: cut/thread/hole photos were taken ~1.6× closer than the good photos (P-51). Retraining with scale augmentation (`own_fabric_s18`); the rig needs a fixed camera distance.
    - **mvtec_wood:** image AUROC **0.984** (published EfficientAD-S ~0.99, so the training reproduces it); 0/19 false alarms; recall 51/60 (liquid 2/10 is the weak type); pixel AUROC 0.873, low because of the P-50 misalignment.
  - **Paused at 23:35 (user request).** Done: own_wood, own_fabric, mvtec_wood (`model.pt` + `eval.json`), YOLO11n.
    - Remaining, in order, via `runs/resume_w3b.sh`: mvtec_carpet (from 0); re-evaluation of all four with the aligned heatmap (P-50); `own_fabric_s18` (P-51).
    - After that: int8 exports (W3-E), then the phone NPU run.
  - **2026-11-22 - session 6 merged, training resumed.**
    - 55 usable close-distance good + 15 thread fabric photos (data card, P-51), labelled and merged with stable splits (P-54).
    - Queue (`runs/resume_w3b.sh`):
      1. mvtec_carpet;
      2. re-evaluation of all four with the aligned heatmap (P-50). own_fabric is now scored on the *new* fabric val/test sets; its old-test results stay in `eval_nopad.json`;
      3. `own_fabric_s18` (scale aug 1.8) trained on the new 65 good;
      4. YOLO11n retrain on the merged data (first model kept as `models/yolo/yolo11n_combined_v1`).
    - Training checkpoints every 5k iterations (P-52).
  - **2026-11-22 - results after re-evaluation** (aligned heatmap; own_fabric on the new session-6 test set). Training was interrupted twice by Windows low-memory protection (P-55) and resumed from checkpoints.

    | Model | Image AUROC | Pixel AUROC | False alarms (test good) | Recall | Notes |
    |---|---|---|---|---|---|
    | own_wood | **1.000** | - | 0/15 | 50/51 | box hits 55/56 |
    | own_fabric (orig.) | 0.805 | - | 0/20 | 52/85 | thread 4/30, cut 13/20 |
    | own_fabric_s18 (zoom aug) | 0.729 | - | 0/20 | 48/85 | thread 0/30: worse (P-57) |
    | mvtec_wood | **0.985** | 0.859 | 0/19 | 51/60 | liquid 2/10 |
    | mvtec_carpet | **0.988** | **0.968** | **15/28** | 89/89 | val threshold doesn't transfer (P-56) |

    - MVTec results reproduce published EfficientAD-S quality (image AUROC ~0.99).
    - The heatmap pad (P-50) had a negligible effect; that hypothesis was not confirmed.
    - Fabric thread is a resolution limit, so 2×2 tiling is being tested (`own_fabric_t2`, P-57). A 20-iteration smoke model already reached AUROC 0.951.
    - The carpet false alarms motivate an operator "calibrate on good frames" step in the app (week 4).
  - **YOLO retrained (v2, after session 6):**
    - Early-stopped at 124 epochs. Operating confidence 0.116 (val F1-max).
    - val mAP50 0.707 (v1 0.701); MVTec test 0.728 (v1 0.709).
    - Own test, new set, both models scored: v2 0.376 vs v1 0.509. v2 drops on *unchanged* classes, so this is run-to-run variance on tiny test groups (P-58).
    - Chosen on val (a tie), so v2 is current and v1 is kept.
    - YOLO can't see fabric thread either (AP ~0).
  - **Tiled fabric model (`own_fabric_t2`, P-57):**
    - Image AUROC **0.879** (untiled 0.805); cut AUROC 1.00 (0.875), 19/20 caught; 0/20 false alarms.
    - Thread still 2/30 (AUROC 0.66): a pulled thread looks like normal loop texture to the model.
    - Fabric adopts 2×2 tiles (4 NPU runs per frame). Thread is an open limitation (options in P-57).
  - **User decision (2026-11-22):** fabric thread is **accepted as a limitation**: reported as not reliably detected (2/30); no raking-light recapture or 3×3 tiles. Next: int8 NPU export of own_wood, own_fabric_t2, mvtec_wood, mvtec_carpet and YOLO11n (`runs/export_all.sh`).
  - **Corrected results: preprocessing now matches training (P-60).** These are the numbers of record for the deployed anomaly models:

    | Model | Image AUROC | Pixel AUROC | False alarms | Recall | Per type |
    |---|---|---|---|---|---|
    | own_wood | **1.000** | - | 1/15 | **51/51** | all types caught; box hits 56/56 |
    | own_fabric_t2 (2×2 tiles) | **0.920** | - | 0/20 | 68/85 | colour 20/20, cut 20/20, hole 15/15, thread 13/30 |
    | mvtec_wood | **0.976** | 0.874 | 2/19 | 52/60 | liquid 4/10 |
    | mvtec_carpet | **0.999** | **0.986** | 12/28 | 89/89 | val threshold doesn't transfer (P-56) |

  - **int8/NPU (W3-E) in progress.** 8a8w and 16a8w both collapsed until calibration got synthetic defects for activation ranges (P-59). MVTec wood 16a8w: AUROC 0.945 vs fp32 0.975.
  - **Quantised NPU versions of the anomaly models (16a8w, SSG2115P, simulated on the CPU with the same data and protocol; `models/npu/<model>_16a8w/export_report.json`):**

    | Model | fp32 AUROC | 16a8w AUROC | Δ | Score corr. | False alarms fp32 → 16a8w | Recall fp32 → 16a8w | ≤ 0.02 target |
    |---|---|---|---|---|---|---|---|
    | own_wood | 1.000 | 1.000 | 0.000 | 0.992 | 1/15 → 2/15 | 51/51 → 51/51 | ✅ |
    | own_fabric_t2 (4 tiles) | 0.918 | 0.917 | −0.001 | 0.959 | 0/20 → 4/20 | 68/85 → 68/85 | ✅ |
    | mvtec_carpet | 0.998 | 0.9996 | +0.001 | 0.999 | 12/28 → 5/28 | 89/89 → 89/89 | ✅ |
    | mvtec_wood | 0.975 | 0.945 | −0.030 | 0.923 | 2/19 → 3/19 | 52/60 → 49/60 | ❌ (by 0.010) |

    - All four compile with 146 of 147 nodes on the NPU (8.7 MB each). XNNPACK fp32 fallbacks were exported too.
    - The two demo models (own_wood, own_fabric_t2) lose no ranking quality on the NPU.
    - Their val-derived thresholds give a few more false alarms. The planned on-site "calibrate" step (P-56) sets the threshold from the NPU model's own outputs, which also absorbs this.
    - mvtec_wood misses the target narrowly. It's a benchmark model; calibration settings are **not** tuned against test results (ADR-019), so this is reported as is.
    - The fp32 AUROCs in this table come from the export script's PIL loader and differ by ≤ 0.002 from `evaluate_efficientad.py` (both antialiased, P-60).
  - **YOLO11n on the NPU failed:** class scores collapse at 8a8w/16a8w/16a16w, even with post-processing on the CPU (P-62).
    - Decision (ADR-012 update): YOLO runs fp32 on the CPU (XNNPACK, ~32 ms); the anomaly models stay on the NPU.
    - Phone package `models/npu/phone_w3/` (4 anomaly models × good/defect input) is ready for the on-phone NPU run.
  - **Phone NPU run (2026-11-22, typed into a new Termux session; `spike/results/npu_w3/`):**
    - All 8 cases (4 models × good/defect) ran on the Hexagon NPU at **28.2-29.1 ms per run**.
    - Decisions agree with fp32 in 7/8. The 8th is a good fabric image that fp32, the simulation and the NPU all score just above the quantised threshold.
    - NPU scores run ~15-20% below the laptop simulation on defects, so thresholds must come from the NPU's own outputs (the app's calibrate step).
    - Frame budget: wood ~29 ms, fabric ~115 ms (4 tiles), plus YOLO ~32 ms on the CPU.
    - (The capture-tool server from week 2 was still running in Termux session 1. The probe ran in a new session, Ctrl+Alt+C, and the server was left untouched.)

## Summary
Week 3 is done.
- **Anomaly models:** all four trained at the paper's 70k iterations. MVTec results reproduce published EfficientAD-S quality.
- **Own wood:** 1.000 AUROC.
- **Own fabric:** 0.920, after three fixes:
  - discovering the camera-distance confound and recapturing (P-51);
  - 2×2 tiling (P-57);
  - matching preprocessing to training (P-60).
  - Thread (13/30) is an accepted limitation.
- **NPU:** quantised anomaly models run at ~28.5 ms. That needed 16-bit activations and synthetic-defect calibration (P-59).
- **YOLO** stays fp32 on the CPU, because its class scores don't survive QNN quantisation (P-62).
- **Carried into week 4:**
  - the operator **calibrate** step (thresholds from NPU outputs on in-condition good frames);
  - one shared preprocessing function;
  - fixed camera distance and lighting on the rig;
  - YOLO NPU quantisation (optional).
- Model card: [docs/models/MODELS.md](../models/MODELS.md).
