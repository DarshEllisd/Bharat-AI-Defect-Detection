"""W4-A acceptance test for vision_server, run ON THE PHONE inside the Termux app (NPU access):

    python ~/bharatai/edge/scripts/test_vision_server.py

1. Starts vision_server, loads the week-3 NPU models (QNN 16a8w) and YOLO (XNNPACK fp32), all resident.
2. Parity: outputs on the week-3 check inputs vs (a) qnn_executor_runner's NPU outputs from the
   week-3 phone run and (b) the laptop references. Expected: identical to (a).
3. Latency: 100 calls per model, compute time (inside the server) and round-trip time (Python view).
4. Memory: server RSS after loading and after the loop; 1,000-call soak on the anomaly model.
Writes ~/bharatai/results/vision_server_test.json
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HOME = Path.home() / "bharatai"
sys.path.insert(0, str(HOME / "edge"))
from app.pipeline.vision_client import VisionClient, VisionServer  # noqa: E402

M = HOME / "models_w3"
PROBE = HOME / "results/npu_w3/outputs"
SOCK = str(HOME / "run/vision.sock")
ANOMALY = {
    "own_wood": "efficientad_own_wood_16a8w_SSG2115P.pte",
    "own_fabric_t2": "efficientad_own_fabric_t2_16a8w_SSG2115P.pte",
}


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]


def main() -> None:
    (HOME / "run").mkdir(exist_ok=True)
    res = {"device_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    srv = VisionServer.start(str(HOME / "edge-bin"), str(HOME / "npu"), SOCK, str(HOME / "run/vision_server.log"))
    c = VisionClient(SOCK)
    try:
        res["rss_mb_start"] = round(rss_mb(srv.proc.pid), 1)
        loads = {}
        for name, pte in ANOMALY.items():
            loads[name] = round(c.load(name, str(M / pte)) / 1000, 1)
        yolo_pte = M / "yolo11n_combined_fp32_xnnpack.pte"
        if yolo_pte.exists():
            loads["yolo"] = round(c.load("yolo", str(yolo_pte)) / 1000, 1)
        res["load_ms"] = loads
        res["rss_mb_loaded"] = round(rss_mb(srv.proc.pid), 1)

        # parity on the week-3 check inputs
        parity = {}
        for name in ANOMALY:
            for tag in ("good", "defect"):
                prefix = f"{name}_16a8w"
                x = np.fromfile(M / f"{prefix}__{tag}.input.raw", np.float32).reshape(1, 3, 256, 256)
                out = c.infer(name, x)[0][0].ravel()
                npu_ref = np.fromfile(PROBE / f"{prefix}_{tag}/output_0_0.raw", np.float32)
                sim_ref = np.fromfile(M / f"{prefix}__{tag}.ref.raw", np.float32)
                parity[f"{name}_{tag}"] = {
                    "max_abs_vs_qnn_executor_runner": float(np.abs(out - npu_ref).max()),
                    "max_abs_vs_laptop_sim": float(np.abs(out - sim_ref).max()),
                    "score_server": float(out.max()), "score_runner": float(npu_ref.max()),
                }
        if "yolo" in loads:
            x = np.fromfile(M / "yolo__own.input.raw", np.float32).reshape(1, 3, 320, 320)
            out = c.infer("yolo", x)[0][0].ravel()
            ref = np.fromfile(M / "yolo__own.fp32.raw", np.float32)
            parity["yolo_own"] = {"max_abs_vs_laptop_fp32": float(np.abs(out - ref).max()),
                                  "max_score_server": float(out.reshape(52, -1)[4:].max()),
                                  "max_score_laptop": float(ref.reshape(52, -1)[4:].max())}
        res["parity"] = parity

        # latency
        lat = {}
        for name, shape in [("own_wood", (1, 3, 256, 256)), ("own_fabric_t2", (1, 3, 256, 256))] + \
                           ([("yolo", (1, 3, 320, 320))] if "yolo" in loads else []):
            x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
            comp, rtt = [], []
            for _ in range(100):
                t0 = time.perf_counter()
                _, us = c.infer(name, x)
                rtt.append((time.perf_counter() - t0) * 1000)
                comp.append(us / 1000)
            lat[name] = {"compute_p50_ms": round(pct(comp, 50), 2), "compute_p95_ms": round(pct(comp, 95), 2),
                         "roundtrip_p50_ms": round(pct(rtt, 50), 2), "roundtrip_p95_ms": round(pct(rtt, 95), 2),
                         "ipc_overhead_p50_ms": round(pct(rtt, 50) - pct(comp, 50), 2)}
        res["latency"] = lat

        # soak
        x = np.random.default_rng(1).standard_normal((1, 3, 256, 256)).astype(np.float32)
        r0 = rss_mb(srv.proc.pid)
        t0 = time.time()
        for _ in range(1000):
            c.infer("own_wood", x)
        res["soak_1000"] = {"seconds": round(time.time() - t0, 1), "rss_mb_before": round(r0, 1),
                            "rss_mb_after": round(rss_mb(srv.proc.pid), 1)}
    finally:
        c.close()
        srv.stop()
    out = HOME / "results/vision_server_test.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("VISION_SERVER_TEST_DONE")


if __name__ == "__main__":
    main()
