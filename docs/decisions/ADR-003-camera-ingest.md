# ADR-003: Camera frames reach the backend via Chrome `getUserMedia` → WebSocket

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0, validated in week 1

## Context
Termux has no direct high-frame-rate camera API. The backend still needs a continuous stream of frames.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Browser camera (`getUserMedia`) → JPEG → WebSocket to the Termux backend** | No extra app; `http://localhost` counts as a secure context; maps cleanly onto CameraX later | Encoding happens in the browser; the tab must stay in the foreground |
| IP Webcam app → MJPEG → OpenCV | Easy | Depends on a third-party app |
| `termux-camera-photo` (Termux:API) | Simple | About 1 frame per second at best: not real time |

## Decision
The dashboard page opens the camera, draws frames to a canvas, encodes them as JPEG and sends them over a WebSocket to the backend.

## Why
Measured in week 1 (back camera, JPEG q=0.8, one frame in flight at a time):

| Width sent | FPS | Round trip avg / p95 | Browser encode avg |
|---|---|---|---|
| 320 | 48.5 | 6.9 / 10.2 ms | 13.7 ms |
| **640** | **29.0** | **14.8 / 19.9 ms** | 19.7 ms |
| 1280 (upscaled) | 10.8 | 37.5 / 45.9 ms | 55.1 ms |

The target was ≥ 15 FPS at 640 px with a p95 round trip ≤ 60 ms; both are met comfortably.

## Trade-offs and consequences
- **Browser JPEG encoding costs more than the transfer.** Since the model input is 320 px ([ADR-013](ADR-013-vision-input-resolution.md)), send ~320-480 px crops of the inspection area, not full frames.
- The camera delivered **720×1280 portrait**. Lock the orientation or crop a square region of interest so frames aren't mostly background.

## Evidence / references
- `spike/phone/ws_echo/` (server + page)
- `spike/results/phone_live/camera/*.json`

## Update 2026-11-22
The streaming path stays for the live **preview**. The image that is **inspected** comes from a full-resolution frame grab on an Inspect button press ([ADR-020](ADR-020-button-trigger-fixed-rig.md)).
