# ADR-015: Drive the phone from the laptop over adb (run-as helpers + typing into Termux)

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 1

## Context
Typing every command on the phone is slow and error-prone. The GitHub build of Termux is **debuggable**, so `adb shell run-as com.termux` can act as the Termux user.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| Type commands in Termux by hand | Exact app context | Slow; results must be copied back |
| Termux `RUN_COMMAND` intent | Real app context | Needs permission setup for the caller |
| **`run-as com.termux` over adb** (helpers below) | Scriptable; output comes straight back to the laptop | SELinux domain `runas_app` ≠ the app's `untrusted_app` (NPU blocked); no access to `~/storage`; ends if USB is unplugged |
| **`adb shell input text` into the open Termux app** | Real app context | Phone must be unlocked with Termux in the foreground; quoting is tricky |

## Decision
Use a mix:
- **`spike/laptop/termux_run.sh '<cmd>'`:** runs a command as Termux over adb, with Termux's environment.
- **`spike/laptop/termux_push.sh <src> '~/dest'`:** copies files into Termux's home and checks each file's size. Small files stream through `adb exec-in`; files over 32 MB go via `adb push` to `/data/local/tmp`, then `run-as cp`.
- **`adb shell input text`:** for anything that must run in the real app context (NPU tests).
- **`adb exec-out run-as com.termux cat …`:** to read results back.

## Why
- It let us run setup, benchmarks and the camera server without touching the phone, and pull every result to the laptop.
- `input text` was the only way to test NPU access in the true app context.

## Trade-offs and consequences
- Remember the context difference: **NPU work must be launched from inside the Termux app.**
- Quoting rules learned the hard way (see P-11, P-12, P-21):
  - Quote `'~'` in Git Bash.
  - Set `MSYS_NO_PATHCONV=1` for phone paths, and convert laptop paths with `cygpath -w`.
  - Escape `$` as `\$` in `input text`.
  - Use `$HOME` rather than `~` inside `env VAR=…`.
- Under `run-as`, package postinst scripts can break with `termux-exec` loaded, so use `unset LD_PRELOAD` for pkg/dpkg.

## Evidence / references
- `spike/laptop/termux_run.sh`, `termux_push.sh`
- `docs/problems/PROBLEMS_AND_FIXES.md`
