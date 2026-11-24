# ADR-011: Web stack in Termux is aiohttp (not FastAPI); OpenCV comes from x11-repo

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
The backend serves the dashboard and WebSockets from Termux (Python 3.14, bionic libc, no prebuilt wheels for many packages).

## Options considered
| Option | Pros | Cons |
|---|---|---|
| FastAPI + uvicorn | Popular; typed models | `pydantic-core` is Rust: pip can't bootstrap Rust on `aarch64-linux-android`, so the install failed. Possible via `pkg install rust` + a long on-phone compile |
| **aiohttp** | Installed cleanly; HTTP + WebSocket in one library; already supported by the test server | Less automatic validation |

OpenCV: `pip install opencv-python` (would build from source, which fails) vs **Termux `python-opencv-python` from x11-repo** vs Pillow + SciPy only.

## Decision
- **aiohttp 3.14.5** for the backend. FastAPI remains possible via `WITH_FASTAPI=1` in the setup script.
- **OpenCV 5.0.0** from x11-repo (`python-opencv-python`), plus `dbus`. Pillow + SciPy are kept as a fallback.

## Why
- aiohttp works today, with no compiler toolchain on the phone. The camera test server ran on it at 29 FPS at 640 px.
- OpenCV gives JPEG decoding, resizing and `connectedComponents` (for anomaly regions); all verified on the phone.

## Trade-offs and consequences
- Request validation has to be done manually (or with a lightweight schema helper).
- The OpenCV package pulls in Qt from x11-repo. It needed `dbus` (missing from its declared dependencies) and `dpkg --configure -a` (see P-07, P-08).

## Evidence / references
- `spike/phone/00_setup_termux.sh`
- `spike/results/W1_RESULTS.md` → Termux environment
- `docs/problems/PROBLEMS_AND_FIXES.md` (P-06, P-07, P-08)
