# Model Card: Edge Quality Inspector vision models (week 3)

The on-device vision pipeline for the AIBharat PS5 inspector. It has two complementary detectors per product ([ADR-005](../decisions/ADR-005-hybrid-vision.md)):
- an **anomaly model** (EfficientAD-S) that flags *anything unusual* and localises it with a heatmap;
- a **detector** (YOLO11n) that names the defect type when it recognises one.

All numbers below are on held-out test data, with thresholds chosen on validation data only ([ADR-019](../decisions/ADR-019-evaluation-protocol.md)).

## 1. Deployed models

| Model | Purpose | Input | Runs on (phone) | Latency per run (phone) | File |
|---|---|---|---|---|---|
| `own_wood` | anomaly, demo wood board | 256 px, whole capture square | NPU, 16a8w (QNN, SSG2115P context) | 28.5 ms | `efficientad_own_wood_16a8w_SSG2115P.pte` (8.7 MB) |
| `own_fabric_t2` | anomaly, demo looped fabric | 2×2 tiles of 256 px (512 px square) | NPU, 16a8w, 4 runs/frame | 4 × 28.5 ≈ 115 ms | `efficientad_own_fabric_t2_16a8w_SSG2115P.pte` |
| `mvtec_wood`, `mvtec_carpet` | anomaly, MVTec benchmark / replay mode | 256 px | NPU, 16a8w | 28.5 ms | `efficientad_mvtec_*_16a8w_SSG2115P.pte` |
| `yolo11n_combined` (v2) | defect type, 48 classes (MVTec 9 categories + wood/fabric) | 320 px letterbox | **CPU, fp32 (XNNPACK)** | 22 ms compute, 27 ms with IPC (vision_server, W4) | `yolo11n_combined_fp32_xnnpack.pte` (exported with Ultralytics' ExecuTorch exporter, P-65) |

- Files live on the laptop in `~/bharatai-build/models/npu/<model>_<quant>/`, with `export_report.json` next to each.
- An fp32 XNNPACK fallback `.pte` exists for every anomaly model.
- YOLO's int8/16-bit NPU versions are **not** used: their class scores collapse under QNN quantisation ([P-62](../problems/PROBLEMS_AND_FIXES.md#p-62)).

## 2. Architecture and training

**EfficientAD-S** (Batzner et al., WACV 2024), our own export-friendly module (`training/vision/efficientad_model.py`).
- **Networks:** teacher PDN-small (frozen), student PDN-small with 768 channels, convolutional autoencoder. The anomaly map is the mean of the quantile-normalised student-teacher and student-autoencoder maps, padded by 4 cells and upsampled to the input size.
- **Training** (`train_efficientad.py`):
  - 70,000 iterations (the paper's setting), batch 1, Adam (lr 1e-4, wd 1e-5), LR × 0.1 at 95%.
  - Hard-feature loss over the top 0.1% of values; ImageNet-style penalty on 3,000 Imagenette images.
  - Colour jitter for the autoencoder input; flips and 90° rotations (textures).
  - bf16 autocast, GPU-resident data pipeline ([P-45](../problems/PROBLEMS_AND_FIXES.md#p-45), [P-48](../problems/PROBLEMS_AND_FIXES.md#p-48)).
  - Map-normalisation quantiles (q 0.9 / 0.995) come from **validation-good** images.
  - Overfitting monitor: mean discrepancy on val-good vs a train subset every 5k iterations. No overfitting was seen on any model.
- **Tiling** (fabric only, [P-57](../problems/PROBLEMS_AND_FIXES.md#p-57)): the capture square is resized to 512 px and the model is trained on, and scores, 2×2 tiles of 256 px. The image score is the max of the stitched map. This was needed for thin defects (cut, thread).
- **Preprocessing:** antialiased resize, the same in training, evaluation and the app (PIL bilinear / `F.interpolate(antialias=True)` / OpenCV `INTER_AREA`), then ImageNet normalisation ([P-60](../problems/PROBLEMS_AND_FIXES.md#p-60)).

**YOLO11n** (Ultralytics 8.4.216), COCO-pretrained.
- 320 px, standard augmentation with `hsv_v` 0.5, early stopping (patience 40).
- v2: 124 epochs, chosen on val ([P-58](../problems/PROBLEMS_AND_FIXES.md#p-58)).
- Ultralytics analytics disabled ([P-47](../problems/PROBLEMS_AND_FIXES.md#p-47)).

## 3. Data
See the [data card](../data/DATASET.md).

| Model | Train (good only for anomaly) | Val (good) | Test |
|---|---|---|---|
| own_wood | 35 photos | 5 | 15 good + 51 defects (colour 15, hole 12, liquid 20, scratch 4) |
| own_fabric_t2 | 65 photos (260 tiles) | 20 | 20 good + 85 defects (colour 20, cut 20, hole 15, thread 30) |
| mvtec_wood / carpet | 222 / 252 (MVTec train, 90%) | 25 / 28 (MVTec train, 10%) | MVTec test: 19+60 / 28+89 |
| YOLO11n | 811 MVTec + 199 own images | 181 + 25 | MVTec test 296, own test 72 |

## 4. Results

### Anomaly models (fp32, antialiased preprocessing; threshold = 99th percentile of val-good scores)
| Model | Image AUROC | Pixel AUROC | False alarms (test good) | Recall (test defects) | Per type |
|---|---|---|---|---|---|
| own_wood | **1.000** | - | 1/15 | **51/51** | all caught; defect-box hits 56/56 |
| own_fabric_t2 | **0.920** | - | 0/20 | 68/85 | colour 20/20, cut 20/20, hole 15/15, **thread 13/30** |
| mvtec_wood | 0.976 | 0.874 | 2/19 | 52/60 | liquid 4/10 |
| mvtec_carpet | **0.999** | **0.986** | **12/28** | 89/89 | all types caught |

### Quantised for the NPU (16a8w; laptop simulation of the quantised graph, same data and protocol)
| Model | fp32 AUROC | 16a8w AUROC | Δ (target ≥ −0.02) | False alarms | Recall |
|---|---|---|---|---|---|
| own_wood | 1.000 | 1.000 | 0.000 ✅ | 2/15 | 51/51 |
| own_fabric_t2 | 0.918 | 0.917 | −0.001 ✅ | 4/20 | 68/85 |
| mvtec_carpet | 0.998 | 0.9996 | +0.001 ✅ | 5/28 | 89/89 |
| mvtec_wood | 0.975 | 0.945 | −0.030 ❌ | 3/19 | 49/60 |

- Quantisation: PT2E with the QNN quantizer, per-channel conv weights.
- Calibration: 64 training images (tiles for fabric), half of them with **synthetic defects used only to set activation ranges** ([P-59](../problems/PROBLEMS_AND_FIXES.md#p-59)). Without them the quantised output collapses to a constant.
- 8a8w can't represent normal-image differences, so the anomaly models use 16a8w.

### On the phone NPU (OnePlus CPH2613, Hexagon V73, from the Termux app)
- All 8 check runs (4 models × good/defect image) execute: **28.2-29.1 ms per run**, 146/147 graph nodes delegated.
- **Decision agreement with fp32: 7 of 8.** The 8th is a good fabric image that fp32, the simulation and the NPU all score just above the quantised threshold.
- NPU scores are ~15-20% lower than the laptop simulation on defects, so **the threshold must be calibrated from the NPU's own outputs** (see §6). Details: `spike/results/npu_w3/` (`npu_probe.txt`, `agreement.json`).

### YOLO11n (fp32; operating confidence 0.116 = F1-max on val)
| Split | mAP50 | mAP50-95 |
|---|---|---|
| val | 0.707 | 0.449 |
| MVTec test | **0.728** | 0.453 |
| own test (72 images) | 0.376 | 0.169 |

- MVTec categories (v1 run): tile 0.90, wood 0.88, carpet 0.83, leather 0.81, cable 0.74, transistor 0.65, metal_nut 0.59, grid 0.56, screw 0.41.
- Own-photo per-class AP swings a lot between runs: v1 scored 0.509 on the same own test set. Each own class is one physical defect with 5-30 photos ([P-58](../problems/PROBLEMS_AND_FIXES.md#p-58)).
- YOLO doesn't detect faint wood colour marks or fabric thread (AP ≈ 0). The anomaly model covers those.

## 5. Intended use and limitations
- **Intended use:** demo-scale visual inspection of one light wood board and one looped fabric (plus MVTec replay), operator in the loop, on one Android phone. Not validated for production QC.
- **Thread defects on fabric are only partly detected** (13/30). A pulled loop looks like normal loop texture. Accepted by the user as a known limitation; a raking side light at capture would likely help.
- **Thresholds from validation data don't always transfer** to new conditions (MVTec carpet: 12/28 false alarms at an excellent AUROC of 0.999, [P-56](../problems/PROBLEMS_AND_FIXES.md#p-56)). The quantised models also shift scores. Hence §6.
- **Small, single-item own data:** one board, one fabric, 1-2 physical defects per type, a few capture sessions, improvised lighting. Own-photo results (especially own_wood's 1.000) are optimistic for live conditions. Camera distance must match the capture setup ([P-51](../problems/PROBLEMS_AND_FIXES.md#p-51)).
- MVTec wood pixel AUROC (0.874) is below published values. This is an open item; the app's decision uses the image score.

## 6. Required in the app (from these results)
1. **"Calibrate" step:** before inspecting, the operator shows ~20 good frames of the product under the current setup. The threshold is set from the deployed (NPU) model's own scores with the same 99th-percentile rule. This fixes threshold transfer and the NPU score offset.
2. Identical preprocessing to training: capture square → antialiased resize (fabric: 512 px → 4 tiles), ImageNet normalisation.
3. YOLO on the CPU (XNNPACK fp32, Ultralytics export); output `(1, 52, 2100)`: rows 0-3 = cx, cy, w, h in **pixels** of the 320 px input; rows 4+ = class scores. Run NMS, then filter to the selected product's classes.
4. Fixed camera distance and lighting on the rig.

## 7. Provenance and licences
Check before any non-research use.

| Component | Source | Licence / terms |
|---|---|---|
| MVTec AD | MVTec Software GmbH | CC BY-NC-SA 4.0 (non-commercial) |
| Own wood/fabric photos | captured by the project member, materials only | project-internal |
| EfficientAD teacher weights | anomalib's published `efficientad_pretrained_weights` (sha256 of the zip `c09aeaa2…684a`) | anomalib project, Apache-2.0 |
| Imagenette (penalty set, quantisation calibration) | fast.ai subset of ImageNet | ImageNet terms (non-commercial research) |
| YOLO11n weights / Ultralytics | Ultralytics | AGPL-3.0 (or Ultralytics enterprise licence) |
| ExecuTorch, QNN backend | PyTorch / Qualcomm | BSD-style / Qualcomm QAIRT licence |
