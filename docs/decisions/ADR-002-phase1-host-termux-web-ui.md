# ADR-002: Phase 1 runs in Termux with a local web dashboard; native Android app later

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0

## Context
With an Android phone as the target ([ADR-001](ADR-001-target-device.md)), we need somewhere to run the pipeline and a way to show the operator UI. The brief scores "operator experience", and everything must run on the device (no cloud).

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Termux (Linux userland on the phone) + browser dashboard on `localhost`** | Python, OpenCV and a web server on the phone; fast iteration; truly on-device | Termux quirks (bionic libc, Android security rules); Chrome must stay in the foreground |
| Laptop first, phone later | Fastest to develop | Not the edge device, so it breaks the "runs on device" requirement until ported |
| Native Android app from day 1 | Best final UX; full camera and NPU access | Kotlin/JNI work before anything is proven; slow iteration |
| Hybrid: develop on the laptop, benchmark on the phone | Fast loop | Still two environments to keep in sync |

## Decision
Phase 1: a **Python backend inside Termux** serving a **local web dashboard** that Chrome opens on the same phone. A **native Android app** (CameraX + ExecuTorch Android library) is a later TODO that reuses the same `.pte` models and config files.

## Why
- The user's explicit choice: a web dashboard for now, an Android app later.
- Keeps everything on the phone from the start, which satisfies the no-cloud rule.
- Python makes iteration quick.

## Trade-offs and consequences
- Chrome stops the camera when the tab goes to the background, so the dashboard must stay on screen (fixed by the native app later).
- Android's security rules apply to Termux processes. For example, NPU access only works when processes are launched from the Termux app itself, not from a `run-as` shell (see [ADR-012](ADR-012-npu-vision-cpu-llm.md)).
- OEM battery optimisation can kill Termux: set its battery mode to *Unrestricted* and enable *Disable child process restrictions*.

## Evidence / references
- `PLAN.md` → Decisions table
- `spike/results/W1_RESULTS.md` → Termux environment, camera link
