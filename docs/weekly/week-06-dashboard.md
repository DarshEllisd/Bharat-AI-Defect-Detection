# Week 06: Operator dashboard and one-tap start

- **Status:** In progress (2026-11-24). Brought forward at the user's request ("working application fast"); wood is parked, fabric is the demo product.
- **Plan reference:** PLAN.md W6 (live view, verdict banner with sound/vibration, acknowledge, advice panel, log with filters, stats, performance panel)

## Done (2026-11-24)
- **Server** (`edge/app/main.py`, `edge/app/storage/db.py`):
  - `POST /api/ack/{id}` (optional note) stores operator acknowledgements in a new `acks` table;
  - `GET /api/record/{id}` returns the full record with acknowledgement and advice;
  - `GET /api/log` adds disposition filter and paging; developer test inspections (note `npu_…`) are hidden;
  - `GET /api/stats` returns counts per verdict, pass rate, defect-type counts, average server time, LLM advice count and average time, and open (unacknowledged) alerts, optionally for today only;
  - CSV/JSON export now includes the acknowledgement and the advice.
- **Page** (`edge/web/index.html`), three tabs:
  - **Inspect:**
    - big INSPECT button, plus Calibrate / Teach good / Lock camera;
    - a large colour-coded **verdict banner**;
    - **sound alerts** (Web Audio, offline: one tone for PASS, two for REWORK, three for HOLD, an alarm for SCRAP, a low double tone for RETAKE) plus vibration, with a sound on/off toggle;
    - defect table with reasons;
    - **Acknowledge** (with optional note) for non-PASS results;
    - advice panel (instant knowledge-base advice, then on-device Llama);
    - image/heatmap toggle;
    - a **speed bar**: camera grab, upload, preparation, NPU, YOLO, fusion, save.
  - **Log:** verdict filter, paging, an "open/✓ ack" column; tap a row to reopen that inspection. CSV/JSON export.
  - **Stats:** today or all time, as tiles (inspections, pass rate, open alerts, average server time, AI advice) plus bars per verdict and per defect type.
  - The Log tab badge shows today's open alerts.
- **One-tap start** (`edge/scripts/start_inspector.sh`): if the inspector isn't running, start it in the background, wait until it answers, and open Chrome on `http://localhost:8000` (`termux-open-url`). `stop_inspector.sh` stops everything. Verified on the phone: started, then Chrome opened on the page.
  - **Autostart** (`edge/scripts/autostart.sh`): when sourced from `~/.bashrc`, opening the Termux app starts the inspector. Claude's safety check (correctly) didn't let Claude edit the phone's shell startup file, so **the user adds the one line** if wanted (see `edge/README.md`).
- 44 unit tests pass (acknowledge + stats added to the DB test). The JavaScript was syntax-checked.

## Still to do
- User test of the new page (sound, acknowledge, log/stats).
- An optional "new demo session" button that archives today's test inspections, so the demo log starts clean.
- Week 7: speed with the camera open, airplane-mode test, 10-minute heat test.
