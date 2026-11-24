# ADR-014: WSL uses mirrored networking (IPv6) for downloads; no proxy

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
The first ExecuTorch build stalled: WSL downloaded PyTorch at ~30 KB/s while Windows got ~12 MB/s. See P-10 for the full diagnosis.

## Root cause (measured)
The home broadband connection's **IPv4 route** to several CDNs is extremely slow, **on Windows too**: Fastly/PyPI ~40 KB/s, CloudFront/pytorch.org ~15 KB/s, GitHub ~8 KB/s. **IPv6** to the same servers runs at 11-14 MB/s. Windows "looked fast" because it used IPv6. WSL's default **NAT mode is IPv4-only**.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Windows-side HTTP proxy for WSL (tried; worked at ~11 MB/s) | Works in NAT mode | An extra process to keep running; only helps dual-stack hosts; more moving parts |
| **WSL mirrored networking** | WSL shares Windows' network, including IPv6; no extra process | Failed to start the first time (`0x8007054f`); came up after a clean `wsl --shutdown` |
| Different network (phone hotspot on mobile data) for IPv4-only hosts | Avoids the slow IPv4 route | Mobile data cost; manual switching |

## Decision
- **`networkingMode=Mirrored`** in `.wslconfig`.
- The temporary proxy was removed at the user's request.
- For large downloads from **IPv4-only** hosts that are slow on this connection (GitHub), use another network if needed. Several IPv4 hosts are fine anyway (Cloudflare, Qualcomm's CloudFront download at 9.6 MB/s).

## Why
With mirrored mode, Python, pip and curl in WSL prefer IPv6 automatically: PyPI 10.8 MB/s, pytorch.org 12.3 MB/s, dl.google.com 10.8 MB/s. No extra process is needed.

## Trade-offs and consequences
- GitHub remains slow over IPv4. Small files are fine; prefer Hugging Face mirrors for weights (`Ultralytics/YOLO11`).
- If mirrored mode stops working, `spike/laptop/check_hosts.sh` shows the speed and IP version per host.
- RSC was disabled during debugging and re-enabled by the user afterwards: it wasn't the cause.

## Evidence / references
- `spike/laptop/check_hosts.sh`, `check_wsl_net.sh`, `check_py_net.py`
- `docs/problems/PROBLEMS_AND_FIXES.md` → P-10
