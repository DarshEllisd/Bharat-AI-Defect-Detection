# Week 2 data tooling

| Script | Runs on | Purpose |
|---|---|---|
| `download_mvtec.sh` | laptop (WSL) | Downloads the 9 MVTec AD categories (official per-category archives), extracts them, records sha256 → `reports/mvtec_manifest.json` |
| `prepare_mvtec.py` | laptop (WSL, `et-venv`) | Builds the YOLO dataset (masks → boxes) and the per-category anomaly dataset; writes `reports/data_stats.md` and `reports/qa_boxes_*.jpg` |
| `capture_tool/` | phone (Termux) | Web page for photographing wood/fabric samples with labels and defect instance IDs |
| `pull_captures.sh` | laptop (WSL or Git Bash) | Copies captured photos from the phone and verifies every file (`--clear` deletes them from the phone afterwards) |

Design decisions: [ADR-016](../../docs/decisions/ADR-016-demo-products-wood-fabric.md) (wood + fabric), [ADR-017](../../docs/decisions/ADR-017-dataset-structure-and-splits.md) (classes, splits, labelling). Week plan: [`docs/weekly/week-02-data.md`](../../docs/weekly/week-02-data.md).

## Capture session (on the phone)

1. **Start the tool from the Termux app** (not over USB, so unplugging the cable won't stop it):
   ```bash
   python ~/bharatai/capture/server.py
   ```
   Then open **http://localhost:8001** in Chrome and allow the camera.
2. **Rig:** phone face down on stacked books, camera ~20-30 cm above the sample, soft even light, no shadows. Tape the sample's position. Don't move anything between shots.
3. **Fill the dashed square with the material.** Press **Lock camera** once the picture looks right; if the phone doesn't support it, just keep the light constant.
4. **Good photos first** (making defects destroys the material): select `good`, take ~120 shots, moving or rotating the piece a little between shots so different areas are covered. **Burst ×5** speeds this up.
5. **Defects:** select a defect type → make the defect → press **New defect** → take ~3 photos of it (move the piece slightly between them). Repeat for 3-4 defects per type. Button colours turn green when a target is reached.

   | Wood | Fabric |
   |---|---|
   | `scratch`: key or knife | `cut`: blade |
   | `hole`: nail or drill | `hole`: punch or burn |
   | `liquid`: water or oil stain | `color`: marker or ink stain |
   | `color`: marker or paint mark | `thread`: pulled-out thread |

6. **Undo** removes the last photo if a shot was bad (only within the current session).
7. Keep fabric flat on cardboard with taped edges: folds look like defects.

Afterwards, on the laptop: `bash training/data/pull_captures.sh` (add `--clear` once verified, to free phone storage).

Photos are stored on the phone at `~/bharatai/data/raw/<product>/<label>/`, each with a `.json` sidecar (label, instance ID, square region, frame size, camera settings, session, time).

## Labelling (on the laptop, local Label Studio)

One-time setup (already done): `bash training/data/label_studio/setup_label_studio.sh` creates its own venv and local-only credentials in `~/bharatai-build/label-studio/credentials.env` (never commit or share them).

1. **Start the server** (runs in the background; safe to run again):
   ```bash
   bash <repo>/training/data/label_studio/ensure_label_studio.sh
   ```
   The first start can take 2-3 minutes (P-36). It prints `healthy: http://localhost:8080` when ready.
2. **Import the pulled photos** (in a second Ubuntu window):
   ```bash
   ~/bharatai-build/ls-venv/bin/python <repo>/training/data/label_studio/setup_projects.py --sync
   ```
   Only defect photos are imported; `good` photos need no boxes.
3. **Log in** at `http://localhost:8080` with the user and password from `credentials.env` (`cat ~/bharatai-build/label-studio/credentials.env`).
4. Open **wood defects** → **Label All Tasks**. For each photo:
   - Pick the label matching the folder in the header (e.g. `.../scratch/...` → **scratch**), drag **one tight box per defect**, then **Submit**.
   - Blurry, or defect not visible? **Skip**; it gets excluded.
5. Repeat for **fabric defects**. Then tell Claude: the boxes are exported in YOLO format and merged into the dataset (instance-level splits, 20% held-out real test set).
