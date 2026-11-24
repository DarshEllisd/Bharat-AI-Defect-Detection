# ADR-001: Target device is an Android phone (OnePlus CPH2613, Snapdragon 7 Gen 3)

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0 (planning), confirmed in week 1

## Context
The brief allows a Raspberry Pi 5 (CPU only), a Pi 5 with the AI HAT+ (Hailo NPU), other Arm Linux devices, or **Arm Android devices with a camera**. The hardware decides the runtime, the UI and the camera path.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Raspberry Pi 5, CPU only | Plain Linux, the brief's reference platform | Must be bought; small models only; separate camera needed |
| Pi 5 + AI HAT+ (Hailo) | Matches the brief's two-pipeline example exactly | Must be bought; HailoRT toolchain to learn |
| **Android phone (owned)** | Already available; built-in camera; capable CPU; has an NPU | No plain Linux; NPU support in ExecuTorch uncertain |

## Decision
Use the team's own **OnePlus CPH2613** phone: **Qualcomm SM7550 (Snapdragon 7 Gen 3)**, 8 GB RAM, Android 14.

## Why
- No hardware to buy, and the camera is built in.
- Measured in week 1: 4× Cortex-A510 @ 1.8 GHz plus 4× Cortex-A715 (3 @ 2.4 GHz, 1 @ 2.63 GHz), with `i8mm`, dot-product and `bf16` CPU instructions that speed up int8 and 4-bit inference.
- It turned out to have a usable **Hexagon V73 NPU** (see [ADR-012](ADR-012-npu-vision-cpu-llm.md)).

## Trade-offs and consequences
- Phase 1 needs a Linux-like environment on Android: Termux ([ADR-002](ADR-002-phase1-host-termux-web-ui.md)).
- RAM visible to Android is 7.07 GiB, with ~2.8 GB free while normal apps are open: close apps before benchmarks and the demo.
- Storage is tight (~10-15 GB free), which affects how many models and runtimes can sit on the phone at once.
- The device was first recorded as a "Snapdragon 7s Gen 4"; the user corrected it to the CPH2613 / 7 Gen 3, and `getprop ro.soc.model` = `SM7550` confirmed it.

## Evidence / references
- `spike/results/W1_RESULTS.md` → Device
- `spike/phone/01_device_profile.sh`
