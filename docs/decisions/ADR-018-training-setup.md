# ADR-018: Training setup: separate GPU venv, anomalib teacher, 4 anomaly models, YOLO on combined data

- **Status:** Accepted
- **Date:** 2026-11-21
- **Week:** 3

## Context
Week 3 trains the vision models. The ExecuTorch build venv must keep its **CPU** PyTorch 2.14, which ExecuTorch v1.5.0 pins. The laptop has an RTX 4050 (6 GB). EfficientAD needs a pretrained teacher network. The user chose the training scope.

## Options considered
| Topic | Options |
|---|---|
| Environment | Swap the build venv's torch for CUDA (risks breaking exports) / **a separate CUDA venv with the same torch version** |
| Teacher | **anomalib's published pretrained PDN-small weights** (standard; GitHub download, slow here) / distil our own from an ImageNet WideResNet (~1 h GPU, reproducible) |
| Anomaly scope | wood + fabric only / **wood + fabric + MVTec wood & carpet** / + all 9 MVTec |
| YOLO data | **combined MVTec + own, standard augmentation** / + copy-paste synthetic experiment |

## Decision
- **`train-venv`**: PyTorch 2.14 (CUDA) for training. Export stays in `et-venv` (CPU torch). Models move between them as `state_dict`s, so the graph that's exported is exactly `training/vision/efficientad_model.py`.
- **Teacher:** anomalib's pretrained PDN-small weights, key-mapped onto our module. Fallback: distil our own.
- **Anomaly models:** `own_wood`, `own_fabric`, `mvtec_wood`, `mvtec_carpet` (user's choice). The two MVTec categories closest to the demo give standard benchmark numbers for the report. The other 7 MVTec categories can be trained later for the replay mode.
- **YOLO11n:** one model on `yolo_combined` (48 classes), 320 px, standard augmentation with stronger brightness jitter (user's choice: no synthetic experiment).

## Why
- A separate venv keeps the proven export toolchain intact.
- The pretrained teacher is how EfficientAD is meant to be used. Distilling on ImageNet ourselves would cost hours and add little.
- MVTec wood/carpet results show whether our training matches published EfficientAD quality before we trust the own-data results.

## Trade-offs and consequences
- Two venvs to maintain. Torch versions must stay aligned (2.14) so state_dicts load identically.
- Teacher weights come from an external project: record their source and licence in the model card. If they can't be fetched, use the fallback.
- 7 MVTec categories have no anomaly model yet, so their replay mode will rely on YOLO until trained.

## Update 2026-11-21 (during training)
- **Schedule:** the paper's full **70,000 iterations** for every EfficientAD model, not a shortened run. bf16 autocast (P-45) and a GPU-resident input pipeline (P-48) bring this to ~55 min per model. A shortened run would have saved little and made the MVTec comparison with the paper weaker.
- **Overfitting monitor instead of early stopping:** with only 35 good own photos, `val_log.csv` records the mean student-teacher and student-autoencoder discrepancy on val-good vs a train subset every 5k iterations. Val has no defects, so there's no label-based stopping. A widening gap is the warning sign.
- **Training runs on the GPU only** (user request): images and penalty pool in VRAM, augmentation on GPU tensors, no per-step CPU-GPU sync; about one CPU core used. CPU-heavy work (int8 export and evaluation) is not run alongside training.
- **Ultralytics analytics off** (`sync=False`, P-47), in line with the organisation's no-external-transmission rule.

## Evidence / references
- `docs/weekly/week-03-training.md`
- EfficientAD: Batzner et al., WACV 2024; anomalib EfficientAd docs (pretrained teacher prepared automatically)
