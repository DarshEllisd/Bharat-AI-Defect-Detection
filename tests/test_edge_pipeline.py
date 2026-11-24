"""Unit tests for the phone inspection pipeline (week 4, task D). Run on the laptop:

    ~/bharatai-build/train-venv/bin/python -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.app.pipeline.calibrate import CalibrationStore, threshold_from_scores  # noqa: E402
from edge.app.pipeline.fusion import decode_yolo, fuse, iou, regions  # noqa: E402
from edge.app.pipeline.inspector import FrameRejected, Inspector  # noqa: E402
from edge.app.pipeline.quality import check as frame_check, texture_period  # noqa: E402
from edge.app.pipeline.record import DefectRecord, InspectionRecord  # noqa: E402
from edge.app.pipeline.severity import SeverityRules  # noqa: E402
from edge.app.pipeline.vision import load_config  # noqa: E402
from edge.app.storage.db import InspectionDB  # noqa: E402

RULES = SeverityRules()


def defect(t="hole", area=0.001, peak=0.2, source="fused", conf=0.8):
    return DefectRecord(type=t, source=source, bbox=[0.1, 0.1, 0.2, 0.2], area_frac=area, peak_score=peak, yolo_conf=conf)


# --- severity ----------------------------------------------------------------------------------
@pytest.mark.parametrize("t,area,peak,expected", [
    ("scratch", 0.001, 0.15, "LOW"),          # base 1
    ("scratch", 0.02, 0.15, "MEDIUM"),        # +1 large (boundary: exactly 2%)
    ("scratch", 0.0199, 0.15, "LOW"),         # just below large
    ("scratch", 0.001, 1.00, "MEDIUM"),       # +1 strong: 1.0 >= 10 x 0.1 (boundary)
    ("scratch", 0.001, 0.9999, "LOW"),        # just below strong
    ("hole", 0.05, 0.9, "CRITICAL"),          # 3 + 1 + 1 capped at 4
    ("liquid", 0.001, 0.1, "MEDIUM"),
    ("mystery", 0.001, 0.1, "MEDIUM"),        # unknown type -> unknown_anomaly rule (base 2)
])
def test_severity_levels(t, area, peak, expected):
    assert RULES.grade(defect(t, area, peak), "wood", threshold=0.1).severity == expected


def test_detector_only_is_a_note_not_a_verdict():
    d = RULES.grade(defect("hole", 0.05, 0.0, source="detector_only"), "wood", threshold=0.1)
    assert d.severity == "NOTE" and d.level == 0 and any("NOTE only" in r for r in d.reasons)
    assert RULES.disposition([d]) == ("NONE", "PASS")


@pytest.mark.parametrize("rule_set", ["wood", "fabric"])
def test_colour_capped_at_medium(rule_set):
    d = RULES.grade(defect("color", area=0.05, peak=5.0), rule_set, threshold=0.1)   # large + strong
    assert d.severity == "MEDIUM" and any("capped" in r for r in d.reasons)
    assert RULES.disposition([d])[1] == "REWORK"


def test_metal_contamination_is_critical_on_fabric():
    assert RULES.grade(defect("metal_contamination"), "fabric", 0.1).severity == "CRITICAL"


@pytest.mark.parametrize("levels,disp", [([], "PASS"), ([1], "REWORK"), ([2, 1], "REWORK"), ([3], "HOLD"), ([1, 4], "SCRAP")])
def test_disposition(levels, disp):
    ds = []
    for lv in levels:
        d = defect()
        d.level = lv
        ds.append(d)
    assert RULES.disposition(ds)[1] == disp


# --- fusion ------------------------------------------------------------------------------------
def blob_map(size=256, centre=(64, 64), radius=8, peak=1.0, base=0.02):
    yy, xx = np.mgrid[:size, :size]
    m = base + peak * np.exp(-(((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * radius ** 2)))
    return m.astype(np.float32)


def test_regions_bbox_and_peak():
    regs = regions(blob_map(), threshold=0.5)
    assert len(regs) == 1
    x1, y1, x2, y2 = regs[0]["bbox"]
    assert x1 < 64 / 256 < x2 and y1 < 64 / 256 < y2
    assert regs[0]["peak"] == pytest.approx(1.02, abs=1e-3)


def test_no_region_below_threshold():
    assert regions(blob_map(peak=0.3), threshold=0.5) == []


def test_fuse_matched_unknown_and_detector_only():
    amap = np.maximum(blob_map(centre=(64, 64)), blob_map(centre=(200, 200)))
    dets = [{"type": "hole", "conf": 0.7, "bbox": [0.22, 0.22, 0.28, 0.28]},       # on the first blob
            {"type": "scratch", "conf": 0.9, "bbox": [0.80, 0.05, 0.95, 0.10]},    # nowhere near a blob
            {"type": "color", "conf": 0.3, "bbox": [0.5, 0.5, 0.55, 0.55]}]        # weak, no blob -> dropped
    ds = fuse(amap, 0.5, dets, detector_only_min_conf=0.5)
    kinds = sorted((d.type, d.source) for d in ds)
    assert kinds == [("hole", "fused"), ("scratch", "detector_only"), ("unknown_anomaly", "anomaly_only")]
    for d in ds:
        RULES.grade(d, "wood", 0.5)
    assert {d.type: d.severity for d in ds}["scratch"] == "NOTE"     # YOLO-only never drives the verdict


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_decode_yolo_filters_product_and_nms():
    names = ["wood_hole", "fabric_cut", "wood_scratch"]
    out = np.zeros((4 + 3, 4), np.float32)
    out[:4, 0] = [100, 100, 20, 20]; out[4, 0] = 0.9           # wood_hole
    out[:4, 1] = [102, 101, 20, 20]; out[4, 1] = 0.8           # duplicate wood_hole -> NMS
    out[:4, 2] = [200, 200, 30, 30]; out[5, 2] = 0.95          # fabric_cut -> filtered out (prefix)
    out[:4, 3] = [50, 250, 40, 10]; out[6, 3] = 0.05           # below conf
    dets = decode_yolo(out[None], names, "wood_", conf=0.1, nms_iou=0.5, input_px=320)
    assert [(d["type"], round(d["conf"], 2)) for d in dets] == [("hole", 0.9)]
    assert dets[0]["bbox"][0] == pytest.approx((100 - 10) / 320)


# --- calibration -------------------------------------------------------------------------------
def test_threshold_rule():
    scores = [0.10] * 19 + [0.12]
    t = threshold_from_scores(scores)
    p99 = np.percentile(scores, 99)
    assert t["threshold"] == pytest.approx(p99 + max(np.std(scores), 0.10 * p99))
    # near-identical frames (fixed stand): std ~ 0 -> the 10% floor keeps the same item from tripping it
    flat = threshold_from_scores([0.2] * 20)
    assert flat["threshold"] == pytest.approx(0.22)
    with pytest.raises(ValueError):
        threshold_from_scores([0.1] * 5)


def test_calibration_store_persists(tmp_path):
    st = CalibrationStore(tmp_path / "cal.json")
    e = st.save("wood", [0.1 + 0.001 * i for i in range(20)])
    st2 = CalibrationStore(tmp_path / "cal.json")
    assert st2.get("wood")["threshold"] == pytest.approx(e["threshold"]) and st2.get("fabric") is None


# --- record + db -------------------------------------------------------------------------------
def test_record_roundtrip_and_db(tmp_path):
    r = InspectionRecord(id="x1", ts="2026-11-22T10:00:00", product="wood", disposition="HOLD", severity="HIGH",
                         image_score=0.5, threshold=0.14, calibrated=True, calibration_id="c1",
                         defects=[RULES.grade(defect("hole"), "wood", 0.14)], timings_ms={"total": 70.0})
    assert InspectionRecord.from_dict(json.loads(json.dumps(r.to_dict()))) == r
    db = InspectionDB(tmp_path / "i.db")
    db.add_inspection(r)
    st = db.stats()
    assert db.get("x1") == r and st["by_disposition"] == {"HOLD": 1} and st["open_alerts"] == 1 and st["pass_rate"] == 0.0
    db.ack("x1", "moved to QC rack")
    assert db.get_ack("x1")["note"] == "moved to QC rack" and db.stats()["open_alerts"] == 0
    csv_text = db.export_csv()
    assert "hole:HIGH" in csv_text and csv_text.count("\n") == 2
    assert json.loads(db.export_json())[0]["id"] == "x1"


# --- inspector end to end with a fake vision runtime -------------------------------------------
class FakeVision:
    """Stands in for VisionRuntime: a blob anomaly at a known place, one YOLO box on it."""
    def __init__(self, cfg, blob=True):
        self.cfg, self.blob = cfg, blob

    def anomaly(self, product, square):
        t = self.cfg["products"][product]["tiles"]
        m = blob_map(256 * t, centre=(80 * t, 80 * t), radius=6 * t, peak=1.0 if self.blob else 0.0)
        return m, {"npu": 28.0 * t}

    def detect(self, product, square):
        if not self.blob:
            return [], {"yolo_cpu": 22.0}
        c = 80 / 256
        return [{"type": "hole", "conf": 0.8, "bbox": [c - 0.03, c - 0.03, c + 0.03, c + 0.03]}], {"yolo_cpu": 22.0}


@pytest.mark.parametrize("product", ["wood", "fabric"])
def test_inspector_end_to_end(tmp_path, product):
    cfg = load_config()
    insp = Inspector(FakeVision(cfg), RULES, InspectionDB(tmp_path / "i.db"), CalibrationStore(tmp_path / "c.json"), tmp_path / "img")
    frame = np.full((1024, 1024, 3), 128, np.uint8)
    r = insp.inspect(product, frame)
    assert r.disposition in ("HOLD", "SCRAP") and r.defects[0].type == "hole" and r.defects[0].source == "fused"
    assert not r.calibrated and r.threshold == cfg["products"][product]["default_threshold"]
    assert Path(r.image_path).exists() and Path(r.heatmap_path).exists() and Path(r.defects[0].crop_path).exists()
    assert insp.db.get(r.id) == r


def test_inspector_good_item_and_calibration(tmp_path):
    cfg = load_config()
    fv = FakeVision(cfg, blob=False)
    insp = Inspector(fv, RULES, InspectionDB(tmp_path / "i.db"), CalibrationStore(tmp_path / "c.json"), tmp_path / "img")
    frame = np.full((1024, 1024, 3), 128, np.uint8)
    entry = insp.calibrate("wood", [frame] * 12)
    assert entry["n_frames"] == 12
    r = insp.inspect("wood", frame)
    assert r.disposition == "PASS" and r.defects == [] and r.calibrated and r.calibration_id == entry["id"]


def test_region_area_excludes_halo_of_strong_defect():
    # a very strong, small defect on a weak background: the area must stay small (halo trimmed)
    amap = blob_map(centre=(128, 128), radius=4, peak=10.0, base=0.05)
    r = regions(amap, threshold=0.1)
    assert len(r) == 1 and r[0]["area_frac"] < 0.01


def test_background_below_threshold_is_not_a_region():
    amap = blob_map(centre=(128, 128), radius=4, peak=1.0, base=0.09)   # background 0.09 < thr 0.1
    r = regions(amap, threshold=0.1)
    assert len(r) == 1 and r[0]["area_frac"] < 0.02


def test_nearby_fragments_merge_into_one_region():
    amap = np.maximum(blob_map(centre=(100, 100), radius=2, peak=1.0), blob_map(centre=(106, 100), radius=2, peak=1.0))
    assert len(regions(amap, threshold=0.5)) == 1          # 6 px apart (< 3% of 256 = ~8 px) -> one defect
    far = np.maximum(blob_map(centre=(60, 60), radius=2, peak=1.0), blob_map(centre=(200, 200), radius=2, peak=1.0))
    assert len(regions(far, threshold=0.5)) == 2


# --- frame check (P-68) --------------------------------------------------------------------------
def stripes(period_px, size=1024, bright=150):
    x = np.arange(size)
    row = bright + 40 * np.sin(2 * np.pi * x / period_px)
    img = np.repeat(row[None, :], size, 0)
    return np.dstack([img] * 3).clip(0, 255).astype(np.uint8)


def test_texture_period_estimate():
    assert texture_period(stripes(20)) == pytest.approx(20, rel=0.08)
    assert texture_period(stripes(45)) == pytest.approx(45, rel=0.08)


def test_frame_check_fabric_scale_and_darkness():
    cfg = {"label": "fabric", "scale_px": [11, 36]}
    assert frame_check(stripes(20), cfg)["ok"]
    too_close = frame_check(stripes(45), cfg)
    assert not too_close["ok"] and "too CLOSE" in too_close["message"]
    dark = frame_check(np.full((1024, 1024, 3), 5, np.uint8), cfg)
    assert not dark["ok"] and "too dark" in dark["message"]
    assert frame_check(np.full((1024, 1024, 3), 128, np.uint8), cfg)["ok"]     # featureless: not rejected
    assert frame_check(stripes(45), {"label": "wood"})["ok"]                     # no scale rule for wood


def test_inspector_rejects_dark_frame_without_logging(tmp_path):
    cfg = load_config()
    insp = Inspector(FakeVision(cfg), RULES, InspectionDB(tmp_path / "i.db"), CalibrationStore(tmp_path / "c.json"), tmp_path / "img")
    with pytest.raises(FrameRejected):
        insp.inspect("fabric", np.full((1024, 1024, 3), 5, np.uint8))
    assert insp.db.recent() == []


# --- advisor (week 5) ----------------------------------------------------------------------------
from edge.app.advisor.llm_worker import AdviceWorker  # noqa: E402
from edge.app.advisor.prompt import PREFILL, build_prompt, extract_json, extract_markdown, fallback, ground, load_kb  # noqa: E402


def adv_record(types=("hole",), disposition="HOLD"):
    defects = []
    for i, t in enumerate(types):
        d = RULES.grade(defect(t, area=0.01, peak=0.5), "wood", threshold=0.1)
        defects.append(d)
    r = InspectionRecord(id=f"r-{'-'.join(types)}", ts="2026-11-24T10:00:00", product="wood", disposition=disposition,
                         severity=max((d.severity for d in defects), default="NONE"), image_score=0.5, threshold=0.1,
                         calibrated=True, defects=defects)
    return r.to_dict()


def test_prompt_contains_record_and_only_relevant_kb():
    kb = load_kb("wood")
    p = build_prompt(adv_record(("hole", "scratch")), kb, "Wood board")
    assert p.startswith("<|begin_of_text|>") and p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n" + PREFILL)
    assert "W-H1" in p and "W-S1" in p and "W-L1" not in p          # liquid not in this record


def test_extract_json_repairs_missing_brace_from_w1_output():
    w1 = ('Here is the response:\n\n{\n  "summary": "Screw has a scratch head defect.",\n  "probable_cause": "Worn bit.",\n'
          '  "recommended_actions": [\n    "A1: Quarantine.",\n    "A2: Replace bit."\n  ],\n  "log_note": "scratch"<|eot_id|>')
    obj = extract_json(w1)
    assert obj and obj["summary"].startswith("Screw") and len(obj["recommended_actions"]) == 2
    assert extract_json("no json here") is None


def test_grounding_keeps_only_kb_ids_of_this_record():
    kb, rec = load_kb("wood"), adv_record(("hole",))
    ok = ground({"summary": "Hole found.", "probable_cause": "Mis-drill.",
                 "recommended_actions": ["W-H1: Quarantine", "W-S1", "Z-Q9 invent", "W-H2"], "log_note": "hole"}, rec, kb)
    assert [a["id"] for a in ok["actions"]] == ["W-H1", "W-H2"]       # W-S1 (other defect) and invented ids dropped
    assert ground({"summary": "x", "recommended_actions": ["call a friend"]}, rec, kb) is None


def test_fallback_is_grounded_and_skips_notes():
    kb, rec = load_kb("fabric"), adv_record(("cut",))
    fb = fallback(rec, kb, "Fabric")
    assert fb["actions"] and all(a["id"].startswith("F-") for a in fb["actions"]) and "cut" in fb["summary"]
    note_only = adv_record(("hole",), disposition="PASS")
    note_only["defects"][0]["severity"] = "NOTE"
    assert fallback(note_only, kb, "Fabric")["actions"] == []


def test_advice_worker_llm_then_fallback(tmp_path, monkeypatch):
    db = InspectionDB(tmp_path / "i.db")
    w = AdviceWorker(db, tmp_path / "llama_main", tmp_path / "m.pte", tmp_path / "t.model")
    w.available = True
    rec = adv_record(("hole",))
    db.add_inspection(InspectionRecord.from_dict(rec))
    # what llama_main generates after the pre-filled '{"summary": "' (no closing brace, as seen in W1)
    good = 'A hole in the board.", "probable_cause": "Drill mis-hit.", "recommended_actions": ["W-H1", "W-H2"], "log_note": "hole"'
    monkeypatch.setattr(w, "generate", lambda prompt: (good, 12.0))
    first = w.submit(rec, "Wood board", "wood")
    assert first["status"] == "pending" and first["actions"]
    import time as _t
    for _ in range(50):
        a = db.get_advice(rec["id"])
        if a["status"] != "pending":
            break
        _t.sleep(0.05)
    assert a["status"] == "llm" and [x["id"] for x in a["actions"]] == ["W-H1", "W-H2"] and a["seconds"] == 12.0
    rec2 = adv_record(("scratch",)); db.add_inspection(InspectionRecord.from_dict(rec2))
    monkeypatch.setattr(w, "generate", lambda prompt: ("I think you should call someone.", 9.0))
    w.submit(rec2, "Wood board", "wood")
    for _ in range(50):
        a = db.get_advice(rec2["id"])
        if a["status"] != "pending":
            break
        _t.sleep(0.05)
    assert a["status"] == "fallback" and "rejected" in a["note"] and a["actions"][0]["id"].startswith("W-S")


def test_markdown_answer_is_parsed_and_grounded():
    # the actual style of Llama 3.2 1B's first phone answer (week 5)
    md = ("Here is the explanation:\n\n**Summary:** A cut was found in the fabric.\n\n**Probable Cause:** Sharp edge.\n\n"
          "**Recommended Actions:**\n* (F-K1) Quarantine the piece\n* (F-K2) Inspect cutting tools\n\n**Log Note:** cut found<|eot_id|>")
    obj = extract_markdown(md)
    assert obj["summary"].startswith("A cut") and obj["recommended_actions"] == ["F-K1", "F-K2"]
    kb, rec = load_kb("fabric"), adv_record(("cut",))
    assert [a["id"] for a in ground(obj, rec, kb)["actions"]] == ["F-K1", "F-K2"]


def test_worker_parses_real_phone_output_with_echoed_prefill(tmp_path, monkeypatch):
    # verbatim shape of the week-5 phone run: the runner echoes the pre-filled '{"summary": "'
    real = ('\n\n{"summary": "Fabric looped with cut edge, likely caused by blade or scissors contact.", \n'
            ' "probable_cause": "Typical cause: blade or scissors contact during cutting or trimming.", \n'
            ' "recommended_actions": ["W-H1", "F-K1"], \n "log_note": "Cut edge found."}<|eot_id|>\n')
    db = InspectionDB(tmp_path / "i.db")
    w = AdviceWorker(db, tmp_path / "x", tmp_path / "m.pte", tmp_path / "t.model"); w.available = True
    rec = adv_record(("cut",)); rec["product"] = "fabric"
    db.add_inspection(InspectionRecord.from_dict(rec))
    monkeypatch.setattr(w, "generate", lambda prompt: (real, 11.4))
    w._run(rec, "Looped fabric", "fabric")
    a = db.get_advice(rec["id"])
    assert a["status"] == "llm" and a["probable_cause"].startswith("Typical cause")
    assert [x["id"] for x in a["actions"]] == ["F-K1"] and a["log_note"] == "Cut edge found."   # W-H1 (wood) dropped


def test_calibration_is_invalidated_when_the_model_file_changes(tmp_path):
    cfg = load_config()
    mdir = tmp_path / "models"; mdir.mkdir()
    cfg["models_dir"] = str(mdir)
    pte = mdir / cfg["products"]["wood"]["anomaly_pte"]; pte.write_bytes(b"model-v1")
    insp = Inspector(FakeVision(cfg, blob=False), RULES, InspectionDB(tmp_path / "i.db"), CalibrationStore(tmp_path / "c.json"), tmp_path / "img")
    insp.calibrate("wood", [np.full((1024, 1024, 3), 128, np.uint8)] * 12)
    assert insp.threshold("wood")[1] is True
    import os as _os
    pte.write_bytes(b"model-v2"); _os.utime(pte, (1, 1))                 # retrained: same name, new file
    assert insp.threshold("wood")[1] is False and insp.calibration("wood") is None


def test_advice_worker_skips_stale_jobs(tmp_path, monkeypatch):
    import threading as _th, time as _t
    db = InspectionDB(tmp_path / "i.db")
    w = AdviceWorker(db, tmp_path / "x", tmp_path / "m.pte", tmp_path / "t.model"); w.available = True
    gate = _th.Event()
    def slow(prompt):
        gate.wait(5)
        return ('A hole.", "probable_cause": "x", "recommended_actions": ["W-H1"], "log_note": "h"', 1.0)
    monkeypatch.setattr(w, "generate", slow)
    recs = []
    for t in ("hole", "scratch", "liquid"):
        r = adv_record((t,)); db.add_inspection(InspectionRecord.from_dict(r)); recs.append(r); w.submit(r, "Wood", "wood")
    gate.set()
    for _ in range(100):
        if all(db.get_advice(r["id"])["status"] != "pending" for r in recs):
            break
        _t.sleep(0.05)
    st = {r["defects"][0]["type"]: db.get_advice(r["id"]) for r in recs}
    assert st["scratch"]["status"] == "fallback" and "skipped" in st["scratch"]["note"]   # stale: skipped
    assert st["liquid"]["status"] in ("llm", "fallback") and "skipped" not in (st["liquid"]["note"] or "")


def test_tiny_region_with_the_image_maximum_is_kept():
    # a 2x2-pixel spot above threshold (smaller than MIN_REGION_PX_FRAC) must still be reported if it holds the max
    amap = np.full((512, 512), 0.1, np.float32)
    amap[300:302, 200:202] = 0.9
    amap[100, 100] = 0.6                      # another tiny speck, not the maximum -> ignored
    r = regions(amap, threshold=0.5)
    assert len(r) == 1 and r[0]["peak"] == pytest.approx(0.9)
