# Edge inspector (phone)

Runs fully on the phone (OnePlus CPH2613, Termux): Chrome page → Python server (aiohttp) → `vision_server` (ExecuTorch: anomaly models on the Hexagon NPU, YOLO on the CPU) → severity → SQLite log.

Design: [ADR-020](../docs/decisions/ADR-020-button-trigger-fixed-rig.md) (button + fixed stand), [ADR-021](../docs/decisions/ADR-021-per-product-calibration.md) (calibration), [ADR-022](../docs/decisions/ADR-022-persistent-vision-server.md) (vision server). Week log: [week 4](../docs/weekly/week-04-edge-pipeline.md).

## Layout
| Path | What |
|---|---|
| `vision_server/` | C++ server (models resident, Unix socket) + `build_android.sh` (WSL, NDK). Binary goes to `~/bharatai-build/edge-bin/` |
| `app/main.py` | HTTP server: `/api/inspect`, `/api/calibrate/{add,commit,reset}`, `/api/products`, `/api/log`, `/api/stats`, `/api/export.{csv,json}`, `/files/...` |
| `app/pipeline/` | `preprocess` (bit-identical to training, P-60/P-63), `vision_client` + `vision` (server client), `fusion` (anomaly regions + YOLO), `severity` (rules), `calibrate`, `record`, `inspector` |
| `app/storage/db.py` | SQLite (inspections, defects, calibrations) + CSV/JSON export |
| `config/categories.toml` | products, model files, tiles, default thresholds |
| `config/severity.toml` | severity rules (**draft for review**) |
| `web/index.html` | operator page (preview, Calibrate, Inspect, result, log) |
| `scripts/` | `test_vision_server.py`, `replay.py` (run on the phone), `parity_preprocess.py` (laptop) |

## Phone folders (`~/bharatai/`)
`edge/` (this folder, minus `vision_server/`), `edge-bin/` (vision_server + libqnn_executorch_backend.so), `npu/` (QNN runtime libs, W1), `models_w3/` (`.pte` files), `inspections/` (log DB + images), `state/calibration.json`.

## Deploy (laptop, WSL; phone on wireless debugging)
```bash
export ANDROID_SERIAL=<ip:port> WSLENV=ANDROID_SERIAL
bash spike/laptop/termux_sync.sh <staged edge folder> "~/bharatai/edge"     # only changed files are sent
```

## One-tap start (week 6)
In Termux: `sh ~/bharatai/edge/scripts/start_inspector.sh`. It starts the inspector in the background (if not running) and opens Chrome on the page. Stop it with `sh ~/bharatai/edge/scripts/stop_inspector.sh`.

**Optional autostart.** To make opening the Termux app start everything, add this line to `~/.bashrc` in Termux yourself:
```
[ -f ~/bharatai/edge/scripts/autostart.sh ] && . ~/bharatai/edge/scripts/autostart.sh  # bharatai autostart
```
To disable it, delete that line, or `touch ~/bharatai/no_autostart`.

## Run manually (on the phone, inside the Termux app; the NPU is only reachable from there)
```bash
cd ~/bharatai/edge && python -m app.main
```
Then open **http://localhost:8000** in Chrome on the phone and allow the camera.
1. Choose the product.
2. **Calibrate** with a good item: 20 frames, shift it slightly when asked.
3. **Inspect** items.

Stop the server with Ctrl+C. The old capture tool (port 8001) can run alongside.

## Tests (laptop)
```bash
~/bharatai-build/train-venv/bin/python -m pytest tests/ -q
```
