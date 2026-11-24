"""SQLite log of inspections, defects and calibrations (ADR-008) + CSV/JSON export.
Images, heatmaps and crops are files next to the database; rows store their paths."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

from ..pipeline.record import InspectionRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, product TEXT NOT NULL, disposition TEXT NOT NULL,
  severity TEXT NOT NULL, image_score REAL, threshold REAL, calibrated INTEGER, calibration_id TEXT,
  n_defects INTEGER, total_ms REAL, image_path TEXT, heatmap_path TEXT, record_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS defects (
  inspection_id TEXT NOT NULL REFERENCES inspections(id), idx INTEGER NOT NULL, type TEXT, source TEXT,
  severity TEXT, level INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL, area_frac REAL, peak_score REAL,
  yolo_conf REAL, crop_path TEXT, PRIMARY KEY (inspection_id, idx));
CREATE TABLE IF NOT EXISTS calibrations (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, product TEXT NOT NULL, threshold REAL, n_frames INTEGER,
  stats_json TEXT);
CREATE TABLE IF NOT EXISTS advice (
  inspection_id TEXT PRIMARY KEY REFERENCES inspections(id), ts TEXT NOT NULL, status TEXT NOT NULL,
  source TEXT, advice_json TEXT, seconds REAL, raw TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS acks (
  inspection_id TEXT PRIMARY KEY REFERENCES inspections(id), ts TEXT NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS ix_insp_ts ON inspections(ts);
"""

CSV_FIELDS = ["id", "ts", "product", "disposition", "severity", "image_score", "threshold", "calibrated",
              "n_defects", "defect_types", "total_ms", "acknowledged", "ack_note", "advice_source", "advice_summary",
              "image_path", "heatmap_path"]


class InspectionDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.executescript(SCHEMA)
        import threading as _th
        self.lock = _th.RLock()       # the advice worker writes from its own thread

    def add_inspection(self, r: InspectionRecord) -> None:
        with self.lock, self.con:
            self.con.execute(
                "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.id, r.ts, r.product, r.disposition, r.severity, r.image_score, r.threshold, int(r.calibrated),
                 r.calibration_id, len(r.defects), r.timings_ms.get("total", 0.0), r.image_path, r.heatmap_path,
                 json.dumps(r.to_dict())))
            self.con.executemany(
                "INSERT INTO defects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(r.id, i, d.type, d.source, d.severity, d.level, *d.bbox, d.area_frac, d.peak_score, d.yolo_conf,
                  d.crop_path) for i, d in enumerate(r.defects)])

    def add_calibration(self, product: str, entry: dict) -> None:
        with self.lock, self.con:
            self.con.execute("INSERT INTO calibrations VALUES (?,?,?,?,?,?)",
                             (entry["id"], entry["ts"], product, entry["threshold"], entry["n_frames"], json.dumps(entry)))

    def set_advice(self, inspection_id: str, status: str, source: str, advice: dict, seconds: float = 0.0,
                   raw: str = "", note: str = "") -> None:
        """status: pending (template stored, LLM running) | llm | fallback"""
        import time as _t
        with self.lock, self.con:
            self.con.execute("INSERT OR REPLACE INTO advice VALUES (?,?,?,?,?,?,?,?)",
                             (inspection_id, _t.strftime("%Y-%m-%dT%H:%M:%S"), status, source, json.dumps(advice),
                              seconds, raw, note))

    def get_advice(self, inspection_id: str) -> dict | None:
        with self.lock:
            row = self.con.execute("SELECT status, source, advice_json, seconds, note FROM advice WHERE inspection_id=?",
                                   (inspection_id,)).fetchone()
        if not row:
            return None
        return {"status": row[0], "source": row[1], **json.loads(row[2]), "seconds": row[3], "note": row[4]}

    def get(self, inspection_id: str) -> InspectionRecord | None:
        with self.lock:
            row = self.con.execute("SELECT record_json FROM inspections WHERE id=?", (inspection_id,)).fetchone()
        return InspectionRecord.from_dict(json.loads(row[0])) if row else None

    def recent(self, limit: int = 20, product: str | None = None, disposition: str | None = None,
               offset: int = 0, include_tests: bool = False) -> list[InspectionRecord]:
        """Newest first. Diagnostic inspections sent by the developer (note 'npu_...') are hidden by default."""
        q, where, args = "SELECT record_json FROM inspections", [], []
        if product:
            where.append("product=?"); args.append(product)
        if disposition:
            where.append("disposition=?"); args.append(disposition)
        if not include_tests:
            where.append("COALESCE(json_extract(record_json,'$.note'),'') NOT LIKE 'npu_%'")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY ts DESC, rowid DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self.lock:
            rows = self.con.execute(q, args).fetchall()
        return [InspectionRecord.from_dict(json.loads(r[0])) for r in rows]

    # --- operator acknowledgement (week 6) ----------------------------------------------------------
    def ack(self, inspection_id: str, note: str = "") -> dict:
        import time as _t
        entry = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%S"), "note": note[:500]}
        with self.lock, self.con:
            self.con.execute("INSERT OR REPLACE INTO acks VALUES (?,?,?)", (inspection_id, entry["ts"], entry["note"]))
        return entry

    def get_ack(self, inspection_id: str) -> dict | None:
        with self.lock:
            row = self.con.execute("SELECT ts, note FROM acks WHERE inspection_id=?", (inspection_id,)).fetchone()
        return {"ts": row[0], "note": row[1]} if row else None

    def stats(self, product: str | None = None, since: str | None = None) -> dict:
        """Counts per disposition, pass rate, defect types, average timings, open (unacknowledged) alerts."""
        where, args = ["COALESCE(json_extract(i.record_json,'$.note'),'') NOT LIKE 'npu_%'"], []
        if product:
            where.append("i.product=?"); args.append(product)
        if since:
            where.append("i.ts>=?"); args.append(since)
        w = " WHERE " + " AND ".join(where)
        with self.lock:
            disp = dict(self.con.execute(f"SELECT i.disposition, COUNT(*) FROM inspections i{w} GROUP BY i.disposition", args).fetchall())
            types = dict(self.con.execute(
                f"SELECT d.type, COUNT(*) FROM defects d JOIN inspections i ON i.id=d.inspection_id{w} "
                "AND d.severity!='NOTE' GROUP BY d.type ORDER BY COUNT(*) DESC", args).fetchall())
            avg_ms = self.con.execute(f"SELECT AVG(i.total_ms) FROM inspections i{w}", args).fetchone()[0]
            adv = self.con.execute(
                f"SELECT AVG(a.seconds), COUNT(*) FROM advice a JOIN inspections i ON i.id=a.inspection_id{w} "
                "AND a.status='llm'", args).fetchone()
            open_alerts = self.con.execute(
                f"SELECT COUNT(*) FROM inspections i LEFT JOIN acks k ON k.inspection_id=i.id{w} "
                "AND i.disposition!='PASS' AND k.inspection_id IS NULL", args).fetchone()[0]
        total = sum(disp.values())
        return {"total": total, "by_disposition": disp, "pass_rate": round(disp.get("PASS", 0) / total, 3) if total else None,
                "defect_types": types, "avg_server_ms": round(avg_ms or 0, 1), "llm_advice": adv[1],
                "avg_llm_seconds": round(adv[0] or 0, 1), "open_alerts": open_alerts}

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in reversed(self.recent(limit=10 ** 9)):
            ack, adv = self.get_ack(r.id), self.get_advice(r.id) or {}
            w.writerow({"id": r.id, "ts": r.ts, "product": r.product, "disposition": r.disposition,
                        "severity": r.severity, "image_score": round(r.image_score, 5), "threshold": round(r.threshold, 5),
                        "calibrated": int(r.calibrated), "n_defects": len(r.defects),
                        "defect_types": ";".join(f"{d.type}:{d.severity}" for d in r.defects),
                        "total_ms": round(r.timings_ms.get("total", 0.0), 1),
                        "acknowledged": ack["ts"] if ack else "", "ack_note": ack["note"] if ack else "",
                        "advice_source": adv.get("status", ""), "advice_summary": adv.get("summary", ""),
                        "image_path": r.image_path, "heatmap_path": r.heatmap_path})
        return buf.getvalue()

    def export_json(self) -> str:
        out = []
        for r in reversed(self.recent(limit=10 ** 9)):
            d = r.to_dict()
            d["acknowledged"], d["advice"] = self.get_ack(r.id), self.get_advice(r.id)
            out.append(d)
        return json.dumps(out, indent=1)
