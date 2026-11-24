# ADR-008: Logging: SQLite + CSV/JSON export + saved defect crops

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0

## Context
The brief requires a structured record of **every inspection**, and the evaluation looks at "easy log" operator UX.

## Options considered
SQLite database · CSV/JSON export · saved defect crops and heatmaps · PDF shift report (multi-select).

## Decision
- **SQLite** with tables for inspections, defects, advice and operator acknowledgements.
- **CSV/JSON export** from the dashboard.
- **Saved crop and heatmap images** of each defective item.
- No PDF report for now.
- One log row per **item** (station mode: inspect once when the item is still), not per frame.

## Why
- SQLite is built into Python, queryable, and drives the history and statistics views.
- Exports cover reports and spreadsheets.
- Crops give traceability.

## Trade-offs and consequences
- Crops use phone storage, which is tight. Add a retention or cleanup policy.
- A PDF shift report can be added later if time allows.

## Evidence / references
- `PLAN.md` → repository layout (`edge/app/storage/`)
