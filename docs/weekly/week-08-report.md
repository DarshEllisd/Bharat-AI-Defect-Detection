# Week 08: Report, repository and submission preparation

- **Status:** In progress (2026-11-24). Submission deadline: 25 November 2026, 23:59 IST (single submission).
- **Plan reference:** PLAN.md W8 (technical report, demo video, README, cleanup)

## Done (2026-11-24)
- **Technical report** `docs/technical_report.md`, built to `technical_report.docx` and `.pdf` (8 pages) with `docs/report_assets/md_to_docx.py` (python-docx; the PDF is exported from the DOCX with Word).
  - Sections: problem, hardware, architecture, data, vision and advisory methodology, operator app, results, hardware utilisation, optimisations, limitations, future work, reproducibility.
  - Figures in `docs/report_assets/`: architecture diagram and a latency breakdown (camera page closed vs open).
  - Live numbers come from the phone's inspection log (76 live fabric inspections on 2026-11-24): server time p50 1,105 ms / p95 1,840 ms with Chrome's camera open; 22/22 Llama answers valid and grounded; Llama 48 s average alongside the camera.
- **Repository preparation:**
  - `.gitignore`: model files (`.pte`), Qualcomm runtime libraries (`.so`), prebuilt runners, caches, editor settings, databases, credentials, and the organisers' PDF are excluded (size and licence terms).
  - Personal paths removed: scripts under `training/vision/runs/` now locate the repository from their own location; READMEs use `<repo>` / `%UserProfile%` placeholders.
  - Top-level `README.md` added (overview, results, layout, quick start, licences).

## Still to do
- Demo video (2 minutes, 1080p, 16:9, MP4).
- Airplane-mode end-to-end run and a 10-minute heat run (results to add to the report).
- Publication of the repository and video: needs sign-off under the organisation's data policy before anything is made externally accessible.
