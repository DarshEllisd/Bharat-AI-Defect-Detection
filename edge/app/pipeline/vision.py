"""VisionRuntime: the vision pipeline's interface to vision_server (ADR-022) - anomaly maps (NPU, tiled
for fabric) and YOLO detections (CPU) for a 1024 px capture square."""
from __future__ import annotations

import json
import os
import time
import tomllib
from pathlib import Path

import numpy as np

from . import preprocess as pp
from .fusion import decode_yolo
from .vision_client import VisionClient

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    with open(config_dir / "categories.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["models_dir"] = os.path.expanduser(cfg["models_dir"])
    cfg["yolo"]["names"] = json.loads((config_dir / "yolo_classes.json").read_text())["names"]
    return cfg


class VisionRuntime:
    def __init__(self, client: VisionClient, cfg: dict) -> None:
        self.c, self.cfg = client, cfg
        self.loaded: set[str] = set()

    def _path(self, pte: str) -> str:
        return str(Path(self.cfg["models_dir"]) / pte)

    def ensure_loaded(self, product: str | None = None) -> None:
        """YOLO + the product's anomaly model (models stay resident; loading is a one-off)."""
        if "yolo" not in self.loaded:
            self.c.load("yolo", self._path(self.cfg["yolo"]["pte"]))
            self.loaded.add("yolo")
        if product and product not in self.loaded:
            self.c.load(product, self._path(self.cfg["products"][product]["anomaly_pte"]))
            self.loaded.add(product)

    def anomaly(self, product: str, square_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """-> stitched anomaly map ((256*T) x (256*T)), timings"""
        self.ensure_loaded(product)
        t = int(self.cfg["products"][product]["tiles"])
        t0 = time.perf_counter()
        tiles = pp.anomaly_inputs(square_bgr, t)
        t1 = time.perf_counter()
        maps, npu_us = [], 0
        for x in tiles:
            outs, us = self.c.infer(product, x)
            maps.append(outs[0])
            npu_us += us
        t2 = time.perf_counter()
        return pp.stitch(maps, t), {"pre_anomaly": (t1 - t0) * 1e3, "npu": npu_us / 1e3,
                                     "anomaly_total": (t2 - t1) * 1e3, "tiles": t}

    def detect(self, product: str, square_bgr: np.ndarray) -> tuple[list[dict], dict]:
        self.ensure_loaded()
        y = self.cfg["yolo"]
        t0 = time.perf_counter()
        x = pp.yolo_input(square_bgr)
        t1 = time.perf_counter()
        outs, us = self.c.infer("yolo", x)
        t2 = time.perf_counter()
        dets = decode_yolo(outs[0], y["names"], self.cfg["products"][product]["yolo_prefix"], y["conf"],
                           y["nms_iou"], y["input_px"])
        t3 = time.perf_counter()
        return dets, {"pre_yolo": (t1 - t0) * 1e3, "yolo_cpu": us / 1e3, "yolo_total": (t2 - t1) * 1e3,
                      "yolo_decode": (t3 - t2) * 1e3}
