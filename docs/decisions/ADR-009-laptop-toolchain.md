# ADR-009: Laptop toolchain: WSL2 Ubuntu, uv Python 3.12, ExecuTorch v1.5.0 (editable), NDK r28c

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
Models must be exported and Android binaries cross-compiled on the laptop (Windows 11, 16 GB RAM, RTX 4050 6 GB). ExecuTorch's build scripts are Linux-oriented. The `.pte` files must match the runtime version on the phone.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **WSL2 Ubuntu 22.04 (already installed)** | Linux toolchain; GPU visible; free | Ships Python 3.10; memory capped by `.wslconfig` |
| Native Windows | No VM | ExecuTorch Android builds and scripts assume Linux |
| Cloud GPU (Colab/Kaggle) for training | Free GPU | The user has a local NVIDIA GPU; training only, not builds |

Python: system 3.10 (too old for ExecuTorch 1.5) vs the deadsnakes PPA vs **uv-managed 3.12**.

## Decision
- **WSL2 Ubuntu 22.04.**
- **Python 3.12 via `uv`** (the system Python is left alone).
- **ExecuTorch v1.5.0** (latest stable at the time), installed with `install_executorch.sh --editable`.
- **Android NDK r28c** (the version ExecuTorch's Llama README is tested with), API 23, arm64-v8a.
- Build with **4 parallel jobs**; WSL memory raised to **10 GB plus 16 GB swap**.
- Builds live on the Linux filesystem (`~/bharatai-build`), not `/mnt/c`.
- **Strip** release binaries before sending them to the phone.

## Why
- The **editable install** makes backends built later in-tree (the Qualcomm/QNN compiler adaptor) importable without reinstalling.
- One ExecuTorch version for exporting and for the phone runtime avoids `.pte` format mismatches.
- **Stripping** cut `executor_runner` from 155 MB to 6.7 MB and `qnn_executor_runner` from 55 MB to 3.4 MB.

## Trade-offs and consequences
- WSL needed networking fixes for downloads ([ADR-014](ADR-014-wsl-networking.md)).
- ExecuTorch v1.5.0 pins **Qualcomm toolkit 2.37.0.250724** and has a smaller chip list than the development branch. Always check docs **at the tag you build** (`spike/laptop/npu/check_qnn_src.sh`).
- Exporting Llama 3B ourselves would need ~24 GB of RAM, which isn't feasible here.

## Evidence / references
- `spike/laptop/build_android_runners.sh`
- `spike/artifacts/BUILD_INFO.txt`
- `spike/results/W1_RESULTS.md` → laptop build
