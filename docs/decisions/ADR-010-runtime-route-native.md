# ADR-010: Run ExecuTorch natively in Termux (route B), not via Python in a proot Debian (route A)

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
Termux uses Android's bionic libc, so PyPI's manylinux ExecuTorch wheels can't load there. Two ways to run ExecuTorch on the phone were tested side by side.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Route B: native C++ runners cross-compiled with the NDK** (`llama_main`, `executor_runner`, `qnn_executor_runner`), driven by Python in Termux | The officially documented Android path; full speed; carries over to the Android app (JNI) | C++ processes to manage; orchestrator ↔ runner IPC |
| Route A: ExecuTorch Python runtime inside a `proot-distro` Debian (glibc) | Everything in Python; the wheel even has LLM-runner bindings | proot overhead; a 2.1 GB container on a storage-tight phone |

## Decision
**Route B.** Native ExecuTorch processes for both pipelines, orchestrated by Termux's own Python 3.14 (aiohttp) over stdin or a Unix socket. The proot container was **removed** afterwards to free 2.1 GB.

## Why
Same YOLO11n `.pte` files, both measured on the phone:

| Input | Threads | Route A (proot Python) | Route B (native) | A / B |
|---|---|---|---|---|
| 320 | 4 | 149.8 ms | **32.2 ms** | 4.6× |
| 320 | 2 | 101.1 ms | 53.8 ms | 1.9× |
| 640 | 4 | 256.6 ms | **140.7 ms** | 1.8× |

The decision rule was "use route A only if within ~15% of route B". It wasn't close. Route A also **gets slower with more threads** (101 → 150 → 180 ms for 2 → 4 → 8 threads). The likely cause is proot's system-call interception making thread synchronisation expensive (not verified).

## Trade-offs and consequences
- Week 4 needs small persistent C++ servers (vision runner, LLM runner) instead of one-shot CLIs. `executor_runner --server_mode` is enough for prototyping.
- Llama's 3 s load means the LLM process must stay running.

## Evidence / references
- `spike/results/W1_RESULTS.md` → Route A
- `spike/results/phone_live/route_a.json`
- `spike/phone/05_route_a_proot.sh`, `route_a_inner.sh`, `bench_pte.py`
