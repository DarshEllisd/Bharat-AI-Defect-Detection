"""Route A benchmark: time vision .pte models through the ExecuTorch Python runtime.

Input shape is inferred from the file name (yolo11n_320.pte -> 1x3x320x320) or set
with --shape. Also reports whether Python LLM-runner bindings are importable, which
decides whether the advisory pipeline could live in Python too.
"""
import argparse
import json
import platform
import re
import statistics
import time
from pathlib import Path

import torch
from executorch.runtime import Runtime


def infer_shape(path: Path, override: str | None) -> list[int]:
    if override:
        return [int(d) for d in override.split(",")]
    m = re.search(r"_(\d{3,4})$", path.stem)
    size = int(m.group(1)) if m else 640
    return [1, 3, size, size]


def set_threads(n: int) -> bool:
    try:
        from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool
    except ImportError:
        return False
    _unsafe_reset_threadpool(n)
    return True


def bench(runtime: Runtime, path: Path, shape: list[int], runs: int, warmup: int) -> dict:
    t0 = time.perf_counter()
    method = runtime.load_program(path).load_method("forward")
    load_ms = (time.perf_counter() - t0) * 1000
    x = torch.rand(shape)
    for _ in range(warmup):
        method.execute([x])
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        method.execute([x])
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "load_ms": round(load_ms, 1),
        "mean_ms": round(statistics.mean(times), 1),
        "p50_ms": round(times[len(times) // 2], 1),
        "p95_ms": round(times[int(len(times) * 0.95) - 1], 1),
        "fps_p50": round(1000 / times[len(times) // 2], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="directory of .pte files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shape", default=None, help="e.g. 1,3,320,320")
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 2, 8])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    import executorch
    report = {
        "python": platform.python_version(),
        "executorch": getattr(executorch, "__version__", "?"),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "results": [],
    }
    try:
        import executorch.extension.llm.runner  # noqa: F401
        report["python_llm_runner_bindings"] = True
    except ImportError as e:
        report["python_llm_runner_bindings"] = f"no ({e})"

    runtime = Runtime.get()
    for path in sorted(Path(args.models).glob("*.pte")):
        shape = infer_shape(path, args.shape)
        for n in args.threads:
            entry = {"model": path.name, "shape": shape, "threads": n, "threads_applied": set_threads(n)}
            try:
                entry.update(bench(runtime, path, shape, args.runs, args.warmup))
            except Exception as e:  # keep going so one bad model doesn't hide the others
                entry["error"] = repr(e)
            print(json.dumps(entry))
            report["results"].append(entry)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
