"""Edge inspector server (week 4). Runs in Termux ON THE PHONE, started from inside the Termux app (the NPU
is only reachable from the app's processes):

    python -m app.main            (from ~/bharatai/edge)       ->  open http://localhost:8000 in Chrome

- starts vision_server (ADR-022) and keeps the models resident;
- POST /api/inspect?product=wood        body = full camera frame (JPEG), header X-Roi = "x,y,s,s"
- POST /api/calibrate/add?product=wood  body = frame (as above) -> score; ...repeat ~20x...
- POST /api/calibrate/commit?product=wood                     -> saves the threshold (ADR-021)
- GET  /api/products, /api/log?limit=&product=, /api/stats, /api/export.csv, /api/export.json
- GET  /files/<relative path>  (saved images, heatmaps, crops)
Frames arrive exactly like the capture tool's training photos (full frame + square ROI), ADR-020.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

from .advisor.llm_worker import AdviceWorker
from .pipeline.calibrate import CalibrationStore, threshold_from_scores
from .pipeline.inspector import FrameRejected, Inspector
from .pipeline.quality import check as frame_check, texture_period
from .pipeline import preprocess as pp
from .pipeline.severity import SeverityRules
from .pipeline.vision import VisionRuntime, load_config
from .pipeline.vision_client import VisionClient, VisionServer
from .storage.db import InspectionDB

HOME = Path(os.environ.get("BHARATAI_HOME", Path.home() / "bharatai"))
DATA = HOME / "inspections"
WEB = Path(__file__).resolve().parents[1] / "web"
PORT = int(os.environ.get("INSPECTOR_PORT", "8000"))


def decode_frame(body: bytes, roi_header: str | None) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise web.HTTPBadRequest(text="not a JPEG/PNG image")
    if roi_header:
        x, y, s, _ = (int(v) for v in roi_header.split(","))
        return pp.to_square(img, (x, y, s, s))
    return pp.to_square(img)        # already square (replay / tests)


def record_json(r) -> dict:
    d = r.to_dict()
    rel = lambda p: "/files/" + str(Path(p).relative_to(DATA)) if p else ""
    d["image_url"], d["heatmap_url"] = rel(r.image_path), rel(r.heatmap_path)
    for dd, src in zip(d["defects"], r.defects):
        dd["crop_url"] = rel(src.crop_path)
    return d


class App:
    def __init__(self) -> None:
        self.cfg = load_config()
        run = HOME / "run"
        run.mkdir(parents=True, exist_ok=True)
        self.server = VisionServer.start(str(HOME / "edge-bin"), str(HOME / "npu"), str(run / "vision.sock"),
                                         str(run / "vision_server.log"))
        self.vision = VisionRuntime(VisionClient(str(run / "vision.sock")), self.cfg)
        for p, c in self.cfg["products"].items():
            if not c.get("replay_only"):
                self.vision.ensure_loaded(p)
        self.db = InspectionDB(DATA / "inspections.db")
        self.calib = CalibrationStore(HOME / "state/calibration.json")
        self.inspector = Inspector(self.vision, SeverityRules(), self.db, self.calib, DATA)
        llama_dir = HOME / "models/llama"
        model = next(iter(sorted(llama_dir.glob("*.pte"))), llama_dir / "missing.pte")
        self.advisor = AdviceWorker(self.db, HOME / "bin/llama_main", model, llama_dir / "tokenizer.model")
        self.lock = threading.Lock()          # one vision request at a time (single socket)
        self.pending: dict[str, list[float]] = {}

    async def run(self, fn, *a):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._locked(fn, *a))

    def _locked(self, fn, *a):
        with self.lock:
            return fn(*a)

    def product(self, req: web.Request) -> str:
        p = req.query.get("product", "")
        if p not in self.cfg["products"]:
            raise web.HTTPBadRequest(text=f"unknown product {p!r}")
        return p

    # --- handlers --------------------------------------------------------------------------------
    async def index(self, req):
        return web.FileResponse(WEB / "index.html")

    async def products(self, req):
        out = []
        for p, c in self.cfg["products"].items():
            cal = self.inspector.calibration(p)       # None if made for another model file (P-71)
            out.append({"id": p, "label": c["label"], "tiles": c["tiles"], "replay_only": c.get("replay_only", False),
                        "calibration": {k: cal[k] for k in ("id", "ts", "threshold", "n_frames", "p99", "std")} if cal else None,
                        "default_threshold": c["default_threshold"]})
        return web.json_response(out)

    async def inspect(self, req):
        p = self.product(req)
        frame = decode_frame(await req.read(), req.headers.get("X-Roi"))
        try:
            r = await self.run(self.inspector.inspect, p, frame, req.query.get("note", ""))
        except FrameRejected as e:
            return web.json_response({"retake": True, **e.info}, status=422)
        out = record_json(r)
        c = self.cfg["products"][p]
        out["advice"] = self.advisor.submit(r.to_dict(), c["label"], c["rules"])   # instant template; LLM follows
        return web.json_response(out)

    async def advice(self, req):
        a = self.db.get_advice(req.match_info["id"])
        if a is None:
            raise web.HTTPNotFound()
        return web.json_response(a)

    async def calib_add(self, req):
        p = self.product(req)
        frame = decode_frame(await req.read(), req.headers.get("X-Roi"))
        try:
            await self.run(self.inspector.check_frame, p, frame)
        except FrameRejected as e:
            return web.json_response({"retake": True, **e.info}, status=422)
        amap, _ = await self.run(self.vision.anomaly, p, frame)
        self.pending.setdefault(p, []).append(float(amap.max()))
        return web.json_response({"n": len(self.pending[p]), "score": float(amap.max())})

    async def teach_add(self, req):
        """'Teach good': save a GOOD frame from the stand for retraining the product's anomaly model (P-69).
        Saved exactly like the training photos: 1024 px capture square, JPEG q95 (merge_own_data.py)."""
        p = self.product(req)
        frame = decode_frame(await req.read(), req.headers.get("X-Roi"))
        # teaching defines the new working distance, so only darkness is checked here; the retrain then sets the
        # product's accepted texture-scale range from the taught photos (teach_retrain.sh, P-70)
        info = frame_check(frame, {k: v for k, v in self.cfg["products"][p].items() if k != "scale_px"})
        if not info["ok"]:
            return web.json_response({"retake": True, **info}, status=422)
        info["period"] = round(texture_period(frame), 1) if self.cfg["products"][p].get("scale_px") else None
        d = HOME / "teach" / p / req.query.get("session", "default")
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.jpg")))
        cv2.imwrite(str(d / f"{p}_teach_{n:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return web.json_response({"n": n + 1, "period": info.get("period"), "dir": str(d)})

    async def calib_commit(self, req):
        p = self.product(req)
        scores = self.pending.pop(p, [])
        try:
            threshold_from_scores(scores)     # validates the count before saving
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        entry = self.calib.save(p, scores, model=self.inspector.model_id(p))
        self.db.add_calibration(p, entry)
        return web.json_response(entry)

    async def calib_reset(self, req):
        self.pending.pop(self.product(req), None)
        return web.json_response({"ok": True})

    def full_record(self, r) -> dict:
        d = record_json(r)
        d["ack"], d["advice"] = self.db.get_ack(r.id), self.db.get_advice(r.id)
        return d

    async def log(self, req):
        q = req.query
        rows = self.db.recent(min(int(q.get("limit", "20")), 500), q.get("product") or None,
                              q.get("disposition") or None, int(q.get("offset", "0")))
        return web.json_response([self.full_record(r) for r in rows])

    async def record(self, req):
        r = self.db.get(req.match_info["id"])
        if r is None:
            raise web.HTTPNotFound()
        return web.json_response(self.full_record(r))

    async def ack(self, req):
        iid = req.match_info["id"]
        if self.db.get(iid) is None:
            raise web.HTTPNotFound()
        body = await req.json() if req.can_read_body else {}
        return web.json_response(self.db.ack(iid, str(body.get("note", ""))))

    async def stats(self, req):
        since = time.strftime("%Y-%m-%dT00:00:00") if req.query.get("today") == "1" else None
        return web.json_response(self.db.stats(req.query.get("product") or None, since))

    async def export_csv(self, req):
        return web.Response(text=self.db.export_csv(), content_type="text/csv",
                            headers={"Content-Disposition": "attachment; filename=inspections.csv"})

    async def export_json(self, req):
        return web.Response(text=self.db.export_json(), content_type="application/json",
                            headers={"Content-Disposition": "attachment; filename=inspections.json"})

    async def files(self, req):
        path = (DATA / req.match_info["path"]).resolve()
        if DATA.resolve() not in path.parents or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    def make(self) -> web.Application:
        app = web.Application(client_max_size=40 * 1024 * 1024)
        app.add_routes([
            web.get("/", self.index), web.get("/api/products", self.products),
            web.post("/api/inspect", self.inspect),
            web.post("/api/calibrate/add", self.calib_add), web.post("/api/calibrate/commit", self.calib_commit),
            web.post("/api/calibrate/reset", self.calib_reset), web.post("/api/teach/add", self.teach_add),
            web.get("/api/log", self.log), web.get("/api/stats", self.stats), web.get("/api/advice/{id}", self.advice),
            web.get("/api/record/{id}", self.record), web.post("/api/ack/{id}", self.ack),
            web.get("/api/export.csv", self.export_csv), web.get("/api/export.json", self.export_json),
            web.get("/files/{path:.+}", self.files),
        ])
        app.on_shutdown.append(lambda _app: asyncio.get_running_loop().run_in_executor(None, self.server.stop))
        return app


def main() -> None:
    a = App()
    print(f"inspector: http://localhost:{PORT}  (models: {sorted(a.vision.loaded)}; "
          f"advisor: {'llama ' + a.advisor.model.name if a.advisor.available else 'templates only'})", flush=True)
    web.run_app(a.make(), host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
