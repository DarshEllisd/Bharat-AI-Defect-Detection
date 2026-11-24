# Problems and Fixes

Every non-trivial problem hit during the project: the symptom, the root cause, the fix, and where the fix lives. Newest entries go at the bottom; numbers are never reused.

| # | Week | Area | Problem | Status |
|---|---|---|---|---|
| [P-01](#p-01) | 1 | Laptop | WSL capped at 4 GB RAM | Fixed |
| [P-02](#p-02) | 1 | Laptop | Ubuntu's Python 3.10 too old for ExecuTorch 1.5 | Fixed |
| [P-03](#p-03) | 1 | Laptop | adb missing / not on PATH | Fixed |
| [P-04](#p-04) | 1 | Planning | Wrong phone chip recorded | Fixed |
| [P-05](#p-05) | 1 | Termux | `opencv-python` package not found | Fixed |
| [P-06](#p-06) | 1 | Termux | FastAPI install fails (Rust needed) | Worked around |
| [P-07](#p-07) | 1 | Termux | OpenCV left half-configured by a failed postinst | Fixed |
| [P-08](#p-08) | 1 | Termux | `import cv2` fails: `libdbus-1.so` missing | Fixed |
| [P-09](#p-09) | 1 | Tooling | PowerShell mangles quoted `wsl -e bash -c` commands | Worked around |
| [P-10](#p-10) | 1 | Network | WSL downloads at ~30 KB/s | Fixed |
| [P-11](#p-11) | 1 | Tooling | Git Bash rewrites `/mnt/...` and phone paths | Fixed |
| [P-12](#p-12) | 1 | Tooling | Git Bash expands `~` to the laptop's home | Fixed |
| [P-13](#p-13) | 1 | Tooling | File copy helper deleted trailing "r" characters | Fixed |
| [P-14](#p-14) | 1 | Tooling | Binaries truncated to ~1 KB; copy loop stopped early | Fixed |
| [P-15](#p-15) | 1 | Tooling | adb server crashes while streaming a 1.1 GB file | Fixed |
| [P-16](#p-16) | 1 | Phone | `llama_main` exit 126 right after copying | Explained |
| [P-17](#p-17) | 1 | Build | `executor_runner` 155 MB | Fixed |
| [P-18](#p-18) | 1 | Route A | ExecuTorch wheel doesn't install PyTorch | Fixed |
| [P-19](#p-19) | 1 | Route A | proot-distro 5.x changed its CLI | Fixed |
| [P-20](#p-20) | 1 | NPU | Scripts used the wrong Qualcomm toolkit version and chip list | Fixed (caught before building) |
| [P-21](#p-21) | 1 | NPU | NPU blocked in the remote shell; typing commands into Termux | Fixed |
| [P-22](#p-22) | 1 | NPU | Qualcomm `envsetup.sh` fails under `set -u` | Fixed |
| [P-23](#p-23) | 1 | NPU | NPU rejects most compiled models | Explained → SSG2115P |
| [P-24](#p-24) | 1 | NPU | Timing parser reported a 20-run total as per-run | Fixed |
| [P-25](#p-25) | 1 | NPU | YOLO int8 export fails (`PendingUnbackedSymbolNotFound`) | Fixed |
| [P-26](#p-26) | 1 | NPU | fp16 models don't compile for some chips | Constraint (int8 only) |
| [P-27](#p-27) | 1 | Phone | Storage tight (88-92% used) | Managed |
| [P-28](#p-28) | 1 | LLM | Llama JSON missing its closing brace | Open (fix in W5) |
| [P-29](#p-29) | 1 | Tooling | NPU compare step can't find its file | Fixed |
| [P-30](#p-30) | 1 | Tooling | Orphaned proxy workers held port 8899 | Fixed |
| [P-31](#p-31) | 2 | Laptop | C: drive full → WSL crashed mid-download and wouldn't start | Fixed (WSL moved to D:) |
| [P-32](#p-32) | 2 | Data | Re-extracting MVTec fails: `Permission denied` | Fixed |
| [P-33](#p-33) | 2 | Network | WSL starts with no network (mirrored mode silently fails) | Fixed (restart); recurring |
| [P-34](#p-34) | 2 | Tooling | Half-created venv reused by setup script | Fixed |
| [P-35](#p-35) | 2 | NPU | EfficientAD QNN compile fails: broadcast of a 2-element tensor | Fixed |
| [P-36](#p-36) | 2 | Network | Label Studio takes ~2+ min to start (closed localhost ports hang in WSL mirrored mode) | Explained |
| [P-37](#p-37) | 2 | Tooling | Label Studio rejects the API token | Worked around |
| [P-38](#p-38) | 2 | Tooling | Pulling 530 MB of photos as one tar stream breaks mid-way | Fixed (per-file, resumable) |
| [P-39](#p-39) | 2 | Phone | adb sees no device after ~19 days (authorization expired) | Fixed |
| [P-40](#p-40) | 2 | Tooling | Pull reported "no captures" / success when the phone couldn't be read | Fixed |
| [P-41](#p-41) | 2 | Phone | USB connection drops as soon as a file transfer starts | Worked around (wireless debugging) |
| [P-42](#p-42) | 2 | Data | Curation rules matched on time of day only, not date | Fixed (caught before applying) |
| [P-43](#p-43) | 2 | Tooling | "Already running" check matched its own command line, so the server was never started | Fixed |
| [P-44](#p-44) | 3 | Model | EfficientAD PDN padded only its pooling layers (58×58 maps instead of the reference 56×56) | Fixed |

---

### P-01
**WSL capped at 4 GB RAM**
- **Symptom:** WSL showed ~3 GB total; the ExecuTorch build and the Llama export need ~8-12 GB.
- **Root cause:** `%UserProfile%\.wslconfig` had `memory=4GB`.
- **Fix:** The user raised it to `memory=10GB` and `swap=16GB`, then ran `wsl --shutdown`. The build script now warns when WSL has less than 8 GB, and compiles with `JOBS=4`.
- **Where:** `spike/laptop/build_android_runners.sh`

### P-02
**Ubuntu's Python 3.10 too old for ExecuTorch 1.5**
- **Symptom:** Ubuntu 22.04 ships 3.10; ExecuTorch 1.5 needs ≥ 3.11.
- **Fix:** `uv venv --seed --python 3.12` creates a standalone Python 3.12 without touching the system Python.
- **Where:** `spike/laptop/build_android_runners.sh`

### P-03
**adb missing / not on PATH**
- **Symptom:** `adb` not found on Windows or in WSL.
- **Fix:** The user installed platform-tools at `C:\Android\platform-tools` and added it to the user PATH. Terminals opened earlier don't see the new PATH, so scripts fall back to the full path.
- **Where:** `spike/laptop/termux_run.sh`, `termux_push.sh`

### P-04
**Wrong phone chip recorded**
- **Symptom:** The plan said "Snapdragon 7s Gen 4 (SM7635)".
- **Fix:** The user corrected it to OnePlus CPH2613 / Snapdragon 7 Gen 3; `getprop ro.soc.model` = `SM7550` confirmed it. The plan, scripts (expected core types) and docs were updated.
- **Lesson:** Confirm hardware facts on the device early (`01_device_profile.sh`).

### P-05
**`opencv-python` package not found in Termux**
- **Symptom:** `E: Unable to locate package opencv-python`.
- **Root cause:** The OpenCV Python bindings live in **x11-repo** as `python-opencv-python`, not in termux-main.
- **Fix:** `pkg install x11-repo` then `pkg install python-opencv-python`.
- **Where:** `spike/phone/00_setup_termux.sh`

### P-06
**FastAPI install fails in Termux**
- **Symptom:** `Failed to build 'pydantic-core'` … `Target triple not supported by rustup: aarch64-unknown-linux-android`.
- **Root cause:** `pydantic-core` is Rust, and pip's automatic Rust install doesn't support Android.
- **Fix:** Use **aiohttp** instead ([ADR-011](../decisions/ADR-011-web-stack-aiohttp.md)). FastAPI is possible with `WITH_FASTAPI=1` (installs Termux's `rust` package first).

### P-07
**OpenCV left half-configured**
- **Symptom:** `dpkg: dependency problems … Package opencv is not configured yet`. The chain was xdg-utils → qt6-qtbase → opencv → python-opencv-python.
- **Root cause:** `xdg-utils`' postinst builds a Perl module with CPAN. Under the `run-as` remote shell, `termux-exec` (LD_PRELOAD) rewrote `perl Makefile.PL` into a linker call with a relative path: `error: expected absolute path: "Makefile.PL"`.
- **Fix:** `unset LD_PRELOAD; dpkg --configure -a`. The setup script now runs `dpkg --configure -a`, and remote pkg/dpkg calls unset `LD_PRELOAD`.
- **Where:** `spike/phone/00_setup_termux.sh`, [ADR-015](../decisions/ADR-015-remote-phone-workflow.md)

### P-08
**`import cv2` fails: `libdbus-1.so` not found**
- **Symptom:** `dlopen failed: library "libdbus-1.so" not found: needed by libQt6DBus.so`.
- **Root cause:** OpenCV pulls in Qt, whose DBus module needs `dbus`, but it isn't declared as a dependency (as of 2026-11-01).
- **Fix:** `pkg install dbus`. Verified that OpenCV 5.0.0 can JPEG-encode and run `connectedComponents`.
- **Where:** `spike/phone/00_setup_termux.sh`

### P-09
**PowerShell mangles `wsl -e bash -c '…'` commands**
- **Symptom:** Commands printed only `==`, or `curl: no URL specified`.
- **Root cause:** Nested quoting between PowerShell, wsl.exe and bash.
- **Fix:** Put multi-line checks in script files and run `wsl -e bash /mnt/c/.../script.sh`.
- **Where:** `spike/laptop/check_*.sh`

### P-10
**WSL downloads at ~30 KB/s while Windows gets ~12 MB/s**
- **Symptom:** The first build stalled for 20+ minutes on PyTorch at 15-40 KB/s.
- **Diagnosis, step by step:**
  1. MTU 1280 (inherited from the Tailscale adapter) → set 1500 → no change, restored.
  2. Tailscale disconnected → no change.
  3. Mirrored networking → failed to start (`0x8007054f`), fell back to *no* network → reverted.
  4. RSC on the Wi-Fi adapter → disabled → no change (the user re-enabled it later).
  5. WSL → Windows link measured at **66 MB/s**, so the virtual switch is fine. That led to a Windows-side proxy workaround (~11 MB/s in WSL), later removed.
  6. **Real cause:** testing IPv4 vs IPv6 on *Windows* showed the **home ISP's IPv4 route to Fastly (PyPI), CloudFront (pytorch.org), GitHub and some AWS hosts is ~10-40 KB/s**, while IPv6 runs at 11-14 MB/s. Windows was fast only because it used IPv6. WSL NAT mode is IPv4-only.
- **Fix:** The user enabled `networkingMode=Mirrored` (WSL Settings). After a clean `wsl --shutdown` it started, WSL got IPv6, and pip downloads ran at ~12 MB/s. The proxy was removed. IPv4-only hosts (GitHub) stay slow; prefer Hugging Face mirrors.
- **Where:** [ADR-014](../decisions/ADR-014-wsl-networking.md); diagnostics in `spike/laptop/check_hosts.sh`, `check_wsl_net.sh`, `check_py_net.py`
- **Lesson:** Test IPv4 and IPv6 separately on the host before blaming the VM.

### P-11
**Git Bash rewrites `/mnt/...` and phone paths**
- **Symptom:** `wsl -e bash /mnt/c/...` silently did nothing. `adb push … /data/local/tmp/…` got Windows-mangled remote paths. Later, with conversion off: `adb: cannot stat '/c/Users/...'`.
- **Root cause:** MSYS converts any argument that looks like a POSIX path when calling a Windows program (wsl.exe, adb.exe).
- **Fix:** `export MSYS_NO_PATHCONV=1` protects phone paths, and laptop paths are converted explicitly with `cygpath -w` (or `wslpath -w` in WSL).
- **Where:** `spike/laptop/push_to_phone.sh`, `npu/run_npu_adb.sh`

### P-12
**Git Bash expands `~` to the laptop's home**
- **Symptom:** `mkdir: '/c': Read-only file system` when copying to `~/bharatai/...`.
- **Fix:** Quote the destination (`'~/bharatai/spike'`); the helper maps `~` to Termux's home.
- **Where:** `spike/laptop/termux_push.sh` (usage comment)

### P-13
**File copy helper deleted trailing "r" characters**
- **Symptom:** Two pushed Python files differed from the originals: "looser" became "loose".
- **Root cause:** `adb shell run-as … sed -i 's/\r$//' file`: the phone-side shell dropped the backslash, so it ran `s/r$//`.
- **Fix:** Strip CRs **on the laptop** (`tr -d '\r'`) before sending. All files were re-pushed and checksums compared.
- **Where:** `spike/laptop/termux_push.sh`

### P-14
**Binaries truncated to ~1 KB; copy loop stopped after one file**
- **Symptoms:** `llama_main` and `executor_runner` arrived as 912 bytes. Only the first file of a folder was copied.
- **Root causes:** `adb shell -T` stdin isn't binary-safe. And an `adb` call inside a `while read` loop consumed the loop's stdin (the file list).
- **Fix:** Use `adb exec-in` (binary-safe) for content and `< /dev/null` on the other adb calls. **Every copy is now size-checked** on the phone.
- **Where:** `spike/laptop/termux_push.sh`

### P-15
**adb server crashes while streaming a 1.1 GB file**
- **Symptom:** `daemon not running; starting now` mid-transfer, at 158-231 MB, twice.
- **Fix:** For files over 32 MB, `adb push` to `/data/local/tmp` (20.6 MB/s), then `run-as com.termux cp`, then delete the temp copy. Checksums verified on both sides.
- **Where:** `spike/laptop/termux_push.sh`

### P-16
**`llama_main` exit 126 right after copying**
- **Symptom:** "Cannot execute" the instant after the copy finished; it worked a moment later.
- **Root cause:** The file was still open for writing (ETXTBSY) when it was executed.
- **Fix:** None needed; run after the copy completes.

### P-17
**`executor_runner` is 155 MB**
- **Root cause:** Built with debug info.
- **Fix:** `llvm-strip --strip-unneeded` (155 MB → 6.7 MB; `qnn_executor_runner` 55 MB → 3.4 MB). Now part of the build scripts.
- **Where:** `spike/laptop/build_android_runners.sh`, `npu/build_qnn.sh`

### P-18
**ExecuTorch pip wheel doesn't install PyTorch**
- **Symptom:** `ModuleNotFoundError: No module named 'torch'` in the route-A benchmark.
- **Fix:** Install `torch==2.14.0` from the CPU-only index (`download.pytorch.org/whl/cpu`) before ExecuTorch. The default PyPI ARM build can pull in GPU libraries.
- **Where:** `spike/phone/route_a_inner.sh`

### P-19
**proot-distro 5.x changed its CLI**
- **Symptom:** The old plugin files and flags didn't exist.
- **Root cause:** proot-distro was rewritten in Python. It installs from **Docker Hub** (fast here: Cloudflare-backed), options go **before** the container name, and it shares the host home by default.
- **Fix:** `proot-distro login --bind SRC:DEST debian -- …`; venv placed in `/opt` inside the container.
- **Where:** `spike/phone/05_route_a_proot.sh`, `route_a_inner.sh`

### P-20
**Scripts used the wrong Qualcomm toolkit version and chip list**
- **Symptom (caught before the 40-minute build):** The scripts used QAIRT **2.51** and stand-in chip **SM7675**, both taken from ExecuTorch's *development-branch* docs.
- **Root cause:** ExecuTorch **v1.5.0** pins **QAIRT 2.37.0.250724**, and its `QcomChipset` has no SM7675.
- **Fix:** A preflight script greps the local v1.5.0 source. The build now uses 2.37.0.250724, the exporter skips chips missing from the installed version, and the V73 stand-ins are SM8550 and SSG2115P.
- **Where:** `spike/laptop/npu/check_qnn_src.sh`, `build_qnn.sh`, `export_qnn_probe.py`
- **Lesson:** Always read docs at the exact tag you build.

### P-21
**NPU blocked in the remote shell; typing commands into Termux**
- **Symptoms:**
  - Under `run-as`, Qualcomm's runtime reported `Missing shared libraries: … libcdsprpc.so`. With copied vendor libs: `createUnsignedPD … not supported`.
  - The first attempts to type the test into Termux used wrong paths.
- **Root causes:**
  - Apps may load the vendor `libcdsprpc.so` only if declared in their manifest; the `run-as` SELinux domain (`runas_app`) is further restricted.
  - `env VAR=~/…` doesn't expand `~`, and the adb shell expanded `$HOME` to `/` before typing.
- **Fix:**
  - Run NPU work **from inside the Termux app** (`untrusted_app`) with `/vendor/lib64` on `LD_LIBRARY_PATH` (the probe retries with it automatically).
  - Type commands with `adb shell input text`, escaping `$` as `\$` and using `$HOME`.
- **Where:** `spike/phone/07_npu_probe.sh`, `npu_app_test.sh`, [ADR-012](../decisions/ADR-012-npu-vision-cpu-llm.md), [ADR-015](../decisions/ADR-015-remote-phone-workflow.md)

### P-22
**Qualcomm `envsetup.sh` fails under `set -u`**
- **Symptom:** `envsetup.sh: line 33: PYTHONPATH: unbound variable`. The build finished, but the probe models didn't compile.
- **Fix:** `export PYTHONPATH="${PYTHONPATH:-}"` and wrap the `source` in `set +u … set -u`. Added `ONLY_MODELS=1` to rerun just that step.
- **Where:** `spike/laptop/npu/build_qnn.sh`

### P-23
**The NPU rejects most compiled models**
- **Symptoms:** `Request feature vtcm size with value 4194304/8388608 unsupported`; `Request feature arch with value 75/79/81 unsupported`.
- **Root cause:** The phone's HTP is **V73 with ≤ 2 MB VTCM**. QNN contexts encode the target's VTCM size and architecture.
- **Fix:** Compile for **SSG2115P** (V73, 2 MB VTCM): those contexts load and match the reference.
- **Where:** [ADR-012](../decisions/ADR-012-npu-vision-cpu-llm.md)

### P-24
**Timing parser reported a 20-run total as per-run**
- **Symptom:** The NPU probe net was reported as "6.28 ms".
- **Root cause:** `qnn_executor_runner` prints `20 inference took X ms, avg Y ms` (Y is per run) and then `Total 1 inference took X ms, avg X ms` (the total). The parser took the last match.
- **Fix:** Parse the non-"Total" line. The probe net is **0.31 ms** per run; reports were corrected.
- **Where:** `spike/phone/npu_compare.py`, `07_npu_probe.sh`

### P-25
**YOLO int8 export fails**
- **Symptom:** `PendingUnbackedSymbolNotFound: Pending unbacked symbols {zuf0}` during `torch.export`.
- **Root cause:** YOLO's Detect head generates anchors data-dependently unless they're cached for the input size.
- **Fix:** As Ultralytics' own exporter does: set Detect `export=True` and `dynamic=False`, run one dry forward pass at 320 px, and use `torch.export(strict=False)`.
- **Where:** `spike/laptop/npu/export_qnn_probe.py`

### P-26
**fp16 models don't compile for some chips**
- **Symptoms:** `Failed to generate Qnn context binary` (YOLO fp16 for SM8350/SSG2115P); `adaptive_avg_pool2d … not decomposed or delegated` (probe fp16 for SSG2115P).
- **Consequence:** The NPU path for this phone is **int8 only**; all vision models need int8 and QNN-supported ops.

### P-27
**Phone storage tight**
- **Symptom:** 88-92% used; free space dropped to ~8.8 GB.
- **Mitigation:** Strip binaries; remove the proot container (2.1 GB) after route A was rejected; delete temporary copies in `/data/local/tmp` after use. Watch it before adding models or crops.

### P-28
**Llama JSON missing its closing brace (open)**
- **Symptom:** The output started with "Here is the response:" and ended without the final `}`.
- **Plan:** The W5 parser extracts the first `{…`, repairs small structural errors, validates the fields, and checks actions against the knowledge base, falling back to a template on failure.

### P-29
**NPU compare step can't find its file**
- **Symptom:** `can't open file 'C:\c\Users\...npu_compare.py'`.
- **Root cause:** `MSYS_NO_PATHCONV=1` (set for adb) was inherited by Windows Python.
- **Fix:** `env -u MSYS_NO_PATHCONV python …`.
- **Where:** `spike/laptop/npu/run_npu_adb.sh`

### P-30
**Orphaned proxy workers held port 8899**
- **Symptom:** After stopping the proxy, the port was still listening.
- **Root cause:** `proxy.py --num-workers 4` spawns child processes that outlived the killed parent.
- **Fix:** Killed all processes running from the proxy venv. The proxy was later removed entirely ([ADR-014](../decisions/ADR-014-wsl-networking.md)).

### P-31
**C: drive full: WSL crashed mid-download and wouldn't start**
- **Symptoms:**
  - The MVTec download died with `Bus error (core dumped)`; then even `grep`/`tail` failed with `Input/output error`.
  - Afterwards: `The distribution failed to start. Error code: 6 … Wsl/Service/CreateInstance/E_FAIL`.
- **Root cause:** The Windows **C: drive had 0 MB free**. WSL stores its whole Linux filesystem in one growing file (`ext4.vhdx`, 25.5 GB at the time) on C:. Inside Linux `df` still showed ~216 GB free (the virtual disk's maximum), so nothing inside WSL warned. When the file couldn't grow, writes failed and the VM crashed. Earlier builds (ExecuTorch, Qualcomm toolkit, NDK) plus the MVTec download had filled the drive.
- **Fix (with the user's approval):**
  1. Cleared unused Windows temp files (`%TEMP%` 6.5 GB → 0.03 GB; Claude Code's own folder and in-use files skipped).
  2. Moved the distro to D: (80 GB free) with `wsl --manage Ubuntu --move D:\WSL\Ubuntu`.
  Docker Desktop's data (10.3 GB) was left alone.
- **Prevention:**
  - Check **host** free space, not WSL's `df`, before large downloads or builds.
  - Delete archives after extraction.
  - Keep datasets and training runs on D: via WSL.
- **Lesson:** Inside WSL, "disk free" describes the virtual disk's ceiling, not the real drive.

### P-32
**Re-extracting MVTec fails: `Permission denied`**
- **Symptom:** After the P-31 crash, the hardened download script verified `cable.tar.xz`, then `rm -rf` of the half-extracted `cable/` failed with hundreds of `Permission denied` errors.
- **Root cause:** MVTec's archives store their directories as **read-only** (`dr-xr-x---`), and `tar` preserves that. Files inside a non-writable directory can't be deleted, even by their owner.
- **Fix:** `chmod -R u+w` before deleting a half-extracted tree and right after every extraction.
- **Where:** `training/data/download_mvtec.sh`

### P-33
**WSL starts with no network: mirrored mode silently fails**
- **Symptoms:** After WSL restarted from its new location (P-31), `pip` failed with `Temporary failure in name resolution`. Inside WSL there were no network interfaces, no `/etc/resolv.conf`, and connections by IP failed too. Windows networking was fine.
- **Root cause:** WSL's mirrored networking sometimes fails to initialise on start and falls back to *no* networking. Earlier it showed the error `0x8007054f` (P-10); this time it was silent.
- **Fix:** `wsl --shutdown`, then start again. The interfaces came back (Wi-Fi address, IPv6) and DNS worked.
- **Prevention:** Network-dependent scripts should fail fast with a clear hint. If `getent hosts pypi.org` fails inside WSL, run `wsl --shutdown` and retry.

### P-34
**Half-created venv reused by the setup script**
- **Symptom:** The Label Studio install "finished", but the venv was 84 KB with no `pip`. The rerun then failed with `ls-venv/bin/pip: No such file or directory`.
- **Root cause:** The first `uv venv` was interrupted (it failed during the no-network episode, P-33) and left a directory behind. The script's `[ -d venv ] || create` check treated it as done.
- **Fix:** Recreate the venv whenever `bin/pip` isn't executable.
- **Where:** `training/data/label_studio/setup_label_studio.sh`

### P-35
**EfficientAD QNN compile fails: broadcast of a 2-element tensor**
- **Symptom:** `RuntimeError: Attempting to broadcast a dimension of length 2 … expected shape should be broadcastable to [1, 1, 58, 58]` inside the QNN backend's `I64toI32` pass (after `Remove0DTensor`). The XNNPACK export of the same model was fine.
- **Root cause:** The model normalised its anomaly maps with `q_st[0]` and `q_st[1]`, scalars indexed out of a 2-element buffer. A QNN preprocessing pass mishandled those 0-D selections, leaving the whole `[2]` tensor in the graph.
- **Fix:** Fold the normalisation into per-map `scale`/`shift` buffers of shape 1×1×1×1 (`map * scale + shift`). It's mathematically identical with fewer ops. Result: **123 of 124 nodes delegated** to the NPU.
- **Lesson for QNN:** Avoid indexing small tensors into scalars inside the graph; prefer broadcastable 4-D constants.
- **Where:** `training/vision/efficientad_model.py`

### P-36
**Label Studio takes ~2+ minutes to start**
- **Symptoms:** After `label-studio start`, `/health` didn't answer for minutes. The process sat in `SYN-SENT` to `127.0.0.1:8080`, its own port. `label-studio --version` also hung.
- **Root cause:** In WSL **mirrored networking**, connecting to a *closed* localhost port isn't refused; it hangs until the TCP SYN timeout (`curl` to closed ports 8099/5555 → timeout after 5 s; `tcp_syn_retries=6` ≈ 2 min). Label Studio checks whether its port is already taken before binding, and that check waits out the timeout.
- **Fix / workaround:** Just wait (~2-3 min) on first start; it then serves normally (reachable from Windows at `http://localhost:8080`). The setup script uses `pip show` instead of `label-studio --version`.
- **Lesson:** In mirrored mode, a hang on localhost usually means "nothing is listening".

### P-37
**Label Studio rejects the API token**
- **Symptom:** `401` for `Authorization: Token …`, even though the server was started with `--user-token`.
- **Root cause:** Label Studio 1.23 disables legacy API tokens by default in favour of its newer token system.
- **Fix:** `setup_projects.py` falls back to a browser-style session login (CSRF cookie + credentials, then `X-CSRFToken` on writes). Settings were left unchanged. Credentials live only in `~/bharatai-build/label-studio/credentials.env` (mode 600, outside the repo).
- **Where:** `training/data/label_studio/setup_projects.py`

### P-38
**Pulling 530 MB of photos as one tar stream breaks mid-way**
- **Symptoms:** `tar: Unexpected EOF in archive`. A second attempt stopped with `error: no devices/emulators found` when the phone dropped off USB mid-copy.
- **Root cause:** Long single streams over adb (`exec-out tar`) are fragile, as with P-15. Any hiccup or disconnect loses the whole stream.
- **Fix:** `pull_captures.sh` copies **one file at a time** (`exec-out cat`), checks each file's size, and **skips files already copied**, so re-running resumes. Final check: every phone file is present on the laptop at the same size. 298 files were verified.
- **Note:** Per-file copying runs at ~0.5 MB/s (~15 min for 530 MB). That's acceptable for occasional pulls.
- **Where:** `training/data/pull_captures.sh`

### P-39
**adb sees no device after ~19 days**
- **Symptoms:** `adb devices` was empty. First Windows didn't see the phone at all (cable/USB mode). Then it showed the `ADB Interface` (WinUSB), but adb still listed nothing, even with the libusb backend.
- **Root cause:** Android expires USB-debugging authorizations after ~7 days without use, and the phone's adb daemon was stuck rather than prompting.
- **Fix (on the phone):** Revoke USB debugging authorizations → toggle USB debugging off and on → replug → accept "Allow USB debugging" with *Always allow*. Optional: enable *Disable adb authorization timeout*.

### P-40
**Pull reported "no captures" (and success) when the phone couldn't be read**
- **Symptom:** In a retry loop, attempt 3 printed `No captures on the phone yet.` and `PULL OK`, but 0 of the 105 new photos had been copied.
- **Root cause:** `adb exec-out` exits 0 even when the phone-side command fails or the device drops. An empty file listing was therefore treated as "nothing to copy".
- **Fix:** The phone-side listing now ends with a marker (`__LISTING_OK__`), and an explicit `__NO_CAPTURES__` when the folder is missing. No marker → `ERROR … could not read the phone` and exit 1. Tested with the device dropping: it now fails loudly.
- **Lesson:** Never infer "empty" from missing output over adb; require a positive end-of-output marker.
- **Where:** `training/data/pull_captures.sh`

### P-41
**USB connection drops as soon as a file transfer starts**
- **Symptoms:**
  - Windows sometimes showed `Unknown USB Device (Device Descriptor Request Failed)`; there was no USB notification on the phone.
  - When the phone did connect, adb listed it and the file listing worked, but the device vanished on the **first file copy**, every time (`size mismatch … error: no devices/emulators found`).
- **Likely cause:** A cable or port that's fine for charging and idle enumeration but fails under sustained data load (or a dirty or worn port).
- **Workarounds:** Android **Wireless debugging** (`adb pair` + `adb connect` over the same Wi-Fi; authenticated, no cable), or a known-good data cable in a different laptop port. `pull_captures.sh` resumes in either case.
- **Used (2026-11-21):** paired once with `adb pair IP:PAIRPORT CODE` (the code popup's port), then `adb connect IP:PORT` (the port shown on the Wireless debugging screen itself; the pairing port refuses connections). Copying ran at ~1.2 MB/s, faster than USB had been.
- **Quirks:**
  - Wireless debugging **switches itself off** (screen lock / Wi-Fi change) and gets a **new port** each time it's re-enabled. Pairing persists, so just `adb connect` the new port.
  - It dropped once mid-pull after the photos but before the sidecars. `check_pairs.sh` caught it, and a re-pull fetched the 99 missing sidecars.
  - Keep the phone unlocked with a long screen timeout while pulling.
  - Pin the device with `ANDROID_SERIAL=IP:PORT` (plus `WSLENV=ANDROID_SERIAL` in WSL) so scripts don't hit "more than one device" if USB also appears.

### P-42
**Curation rules matched on time of day only, not date**
- **Symptom (caught before it did damage):** The rule "fabric photos taken 15:00-23:59 → printed fabric", written for 2026-11-20, would also have excluded the new looped-fabric photos from 2026-11-21, which were taken at 15:44-15:52.
- **Root cause:** `curate_own.py` compared only `HH:MM:SS` from `captured_at`.
- **Fix:** Every rule carries an explicit date. A dry run after the fix showed only the intended photos matched.
- **Lesson:** Time-window rules over multi-day data need the date; always dry-run curation.
- **Where:** `training/data/curate_own.py`

### P-43
**"Already running" check matched its own command line**
- **Symptom:** Starting Label Studio reported nothing useful; `/health` stayed at HTTP 000 and the server log was the old one from days before.
- **Root cause:** `pgrep -f "label-studio start"` ran inside `bash -c '… label-studio start …'`. Its own command line contained the pattern, so it always found a "running" server and skipped the start.
- **Fix:** `ensure_label_studio.sh` uses the pattern `[l]abel-studio start`, which can't match itself, starts the server with `setsid nohup` (detached, so the calling shell returns), and waits for `/health`.
- **Where:** `training/data/label_studio/ensure_label_studio.sh`

### P-44
**EfficientAD PDN padded only its pooling layers**
- **Symptom:** Loading anomalib's pretrained teacher into our `PDNSmall` worked (identical layer names and shapes), but a 256 px input gave a **58×58** feature map. The EfficientAD reference (padding off) gives **56×56**.
- **Root cause:** In `efficientad_model.py` the convolutions had no padding but `AvgPool2d` used `padding=1`. That mix matches neither EfficientAD configuration (all padded / none padded), so the pretrained teacher would have seen a feature layout it wasn't distilled with.
- **Fix:** `AvgPool2d(2, 2)` (no padding), consistent with `padding=False`. The teacher now loads strictly and outputs 56×56; the full model still returns a 256×256 map. Caught by `training/vision/check_teacher.py` before any training.
- **Consequence:** The W2 NPU feasibility numbers (15.3 ms) were measured on the 58×58 variant. They're re-measured with the trained models in W3-E.
- **Teacher provenance:** `efficientad_pretrained_weights.zip` from anomalib's GitHub release, sha256 `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a` (`pretrained_teacher_small.pth`, 10.8 MB).

### P-45
**EfficientAD training slower than planned; parallel runs didn't help**
- **Symptom:** The plan assumed 30-60 min per model. Measured 11 it/s, so 70,000 iterations (paper setting) take ~1.8 h per model, ~7 h for four. Two models trained at once ran at 5.5 it/s each: the same total.
- **Root cause:** At batch size 1 the GPU is compute-bound, not idle. One iteration is ~420 GFLOP (three student passes for image, penalty image and augmented image, two teacher passes and the backward pass), and the convolutions ran in TF32.
- **Fix:** bf16 autocast for the forward passes on the RTX 4050's tensor cores. The losses and the 0.999 hard-feature quantile stay in fp32, and `cudnn.benchmark` is on. Speed went to ~19 it/s total, ~4 h for all four models (two lanes of ~9.4 it/s; the CPU-side image decode now overlaps). A 600-iteration smoke test gave image AUROC 0.930 (bf16) vs 0.948 (fp32) on MVTec wood. At 600 iterations that difference is run-to-run noise; the full runs decide. `--no-amp` restores fp32.
- **Where:** `training/vision/train_efficientad.py`, `training/vision/runs/train_all.sh`

### P-46
**`ModuleNotFoundError: yaml` in the training venv**
- **Symptom:** `train_yolo.py` failed at `import yaml`.
- **Root cause:** Ultralytics 8.4 no longer depends on PyYAML, so the venv didn't have it.
- **Fix:** Installed `pyyaml` and added it to `training/setup_train_venv.sh`.

### P-47
**Ultralytics sends usage analytics by default**
- **Symptom (caught before training):** Ultralytics' `sync` setting defaults to `True`, which sends anonymous usage events to Ultralytics during training and validation.
- **Why it matters:** Organisation policy says no project information leaves the machine for external services.
- **Fix:** `settings.update({"sync": False})`, persisted in the shared `~/.config/Ultralytics/settings.json`, and set again at the top of `train_yolo.py` so it can't be re-enabled silently. The script also `chdir`s to `~/bharatai-build` (where `yolo11n.pt` already exists), so Ultralytics' AMP check doesn't download weights into the repo.
- **Where:** `training/vision/train_yolo.py`

### P-48
**Training kept the CPU busy (data pipeline on the CPU, spin-waiting on the GPU)**
- **Symptom:** Each EfficientAD training process used 1.3-1.75 CPU cores. With YOLO's data loader and a CPU export running at the same time, the 6-core WSL VM was saturated, and EfficientAD dropped to ~4 it/s per process. Request: run training on the GPU only.
- **Root cause:**
  1. Every iteration decoded a penalty JPEG with PIL and ran colour jitter and flips on the CPU.
  2. Every iteration forced several CPU-GPU synchronisations (four `.item()` calls, `torch.quantile` plus boolean indexing in the hard-feature loss). CUDA spin-waits on the CPU at each one.
- **Fix:**
  - Training images and the 3,000-image penalty pool are decoded once and kept in VRAM (uint8, ~0.6 GB); flips, 90° rotations and ColorJitter run on GPU tensors.
  - The hard-feature loss uses `topk` over the top 0.1% of values, which is mathematically the same as "values ≥ the 0.999 quantile", with no sync.
  - Loss logging accumulates on the GPU.
  - A *blocking* CUDA event per iteration lets the CPU sleep instead of spin, at most one iteration ahead.
  - `torch.set_num_threads(1)`.
- **Result:** one process at ~21-23 it/s (faster than the two parallel lanes combined), GPU ~85%, about one CPU core. That core is the Python thread launching kernels, the floor for eager PyTorch. The four models now train one after another (`runs/train_all.sh`), ~55 min each. CPU-heavy jobs (int8 export and simulated-int8 evaluation) wait until training finishes.
- **Where:** `training/vision/train_efficientad.py`, `training/vision/runs/train_all.sh`

### P-49
**Shell scripts rewritten from Windows Python got CRLF line endings**
- **Symptom:** `train_all.sh` exited at once: `set: pipefail: invalid option name`, `cd: $'...\r'`.
- **Root cause:** `Path.write_text()` on Windows translates `\n` to `\r\n`. Bash then reads `\r` as part of each command.
- **Fix:** `sed -i 's/\r$//'` after any scripted edit of a `.sh` file. (Another repeat of the P-43 trap happened the same day: a `pkill -f` pattern inside a `bash -c` string matched that same shell and killed it. Long-running commands now go in script files, never inline next to a `pkill`.)

### P-50
**EfficientAD heatmap misaligned (no map padding before upsampling)**
- **Symptom:** MVTec wood image AUROC was 0.984, in line with the paper, but **pixel AUROC was only 0.873** (typical is ~0.95 or higher).
- **Root cause:** With unpadded convolutions the PDN's 56×56 output covers the image minus a border. The original EfficientAD code (and anomalib's `pad_maps=True`) pads the combined map by 4 cells on each side before resizing to the input size, which aligns each cell with the patch it describes. Our `forward()` resized the 56×56 map straight to 256×256, so every heatmap was shifted and stretched. Image scores (max of the map) are barely affected; localisation is: pixel AUROC, the box hit rate, and the heatmap shown to the operator.
- **Fix:** `F.pad(amap, (4, 4, 4, 4))` before `interpolate` in `efficientad_model.py`. Inference only, so no retraining. All four models are re-evaluated; the earlier results are kept as `eval_nopad.json`. A constant pad is an ordinary QNN op.
- **Where:** `training/vision/efficientad_model.py`
- **Result (2026-11-22) - the hypothesis was NOT confirmed:** re-evaluation with the pad gave MVTec wood pixel AUROC **0.859 (was 0.873)**, image AUROC 0.985 (was 0.984), own_wood box hits 55/56 (was 54/56). So the misalignment wasn't the cause of the low pixel AUROC; the effect is negligible either way. A likely contributor: the zero border pulls scores down for defects touching the image edge. The pad is kept because it matches the reference implementation's geometry and the effect is within noise, but the low pixel AUROC stays an open item. It doesn't affect the image-level decision used by the app.

### P-51
**Own fabric defect photos taken ~1.6× closer than the good photos**
- **Symptom:** own_fabric image AUROC was 0.853.
  - By type: colour 1.00, hole 1.00, cut 0.76, **thread 0.64**.
  - Several thread and cut photos scored *below every good photo*.
- **Diagnosis:**
  - Every photo has the same frame (3072×4096) and the same capture-tool crop (2458 px square), with no digital zoom.
  - A visual check showed bigger loops in the defect photos. A texture-period measure (dominant spatial period from the radially averaged power spectrum) confirmed it. Median loop period in the 1024 px crop: train good 15.5, val good 14.8, test good 18.1, colour 16.1, but **cut 26.6, thread 25.0, hole 26.3**.
  - So the phone was held about 1.6× closer for cut/thread/hole. The test result is confounded: the model partly scores camera distance rather than the defect. That likely inflated hole and deflated thread and cut.
  - Wood grain has no regular period, so the measure doesn't apply to wood. A visual check of wood shows a similar scale across sets, but some hole photos have strong glare the good photos lack. Wood val/test good are near-identical burst shots, so the wood threshold rests on little variety.
- **Fix:**
  1. **Scale augmentation** (`--scale-aug 1.8`, random zoom-in on the GPU) for a retrained `own_fabric_s18`, so normal texture is learned across the distances that occur.
  2. **Capture fix (user):** fixed camera distance on the rig for both demo and data. Good photos at the defect distance would also give a clean, unconfounded test.
- **Protocol note:** this was found by looking at test scores, so it is reported as a data-capture diagnosis, not tuning. The augmentation targets a known capture difference; it was not chosen by trying settings against the test set. Both models' numbers are reported.
- **Lesson:** fix camera height and distance in the rig, and record it per session; check scale statistics per label before training.

### P-52
**Pausing training lost progress (no mid-run checkpoints)**
- **Symptom:** The user needed to pause twice on 2026-11-21 (moving the laptop). Each stop threw away the MVTec carpet run in progress (40.5k, then 10k iterations), because `train_efficientad.py` only saved at the end.
- **Fix:**
  - A checkpoint (model, optimizer, LR schedule, iteration, elapsed time) is written atomically to `<model>/ckpt.pt` every 5k iterations. A restarted run continues from it, appending to its logs, and the checkpoint is deleted once `model.pt` is saved.
  - The queue scripts are resumable: `runs/resume_w3b.sh` skips models that already have `model.pt` + `eval.json`.
  - Tested: killed after the 5k checkpoint, restarted, and it continued at 5,001 and finished. Python RNG state isn't restored, so a resumed run is equivalent but not bit-identical.
- **Where:** `training/vision/train_efficientad.py`, `training/vision/runs/{train_all,after_w3b,resume_w3b}.sh`

### P-53
**`pull_captures.sh --clear` would have re-copied curated-out photos**
- **Symptom (caught before it happened):** The pull script compares the phone's files with `raw/`. `curate_own.py` moves set-aside photos to `raw/_excluded/`. If photos are set aside *before* the phone is cleared, the next pull sees them as missing, copies them back into `raw/` (back into the dataset), and only then clears the phone.
- **Fix:**
  - On 2026-11-22 the phone was cleared first (all 170 files verified), then the curation was applied.
  - The script now treats `raw/_excluded/<product>/<label>/<file>` with the right size as already present, both when copying and when verifying, and leaves `_excluded` out of the per-label counts.
- **Where:** `training/data/pull_captures.sh`

### P-54
**Re-merging would have reshuffled every existing split**
- **Symptom (caught in review):** `merge_own_data.py` shuffled *all* good bursts of a product on every run. Adding session 6 would have moved old fabric photos between train/val/test. That makes old and new results incomparable, and because the merge never deletes, it would have left stale copies of moved photos in their old split folders.
- **Fix:** Splits are stable. Photos already listed in `reports/own_splits.csv` keep their split; only new groups are split, with the same rules applied to the new groups alone. A defect class's new group becomes the held-out test group if the class had none. `--reshuffle` restores the old behaviour. The dry run showed wood unchanged and old fabric unchanged.
- **Where:** `training/data/merge_own_data.py` (ADR-017 update)

### P-55
**Training queue stopped by the host when Windows ran low on memory**
- **Symptom (2026-11-22 14:31):** The background training queue was stopped by Claude Code's low-memory protection while the session was idle. mvtec_carpet was at ~58k/70k iterations. Afterwards Windows had **2.6 GB free of 15.6 GB**; inside WSL only ~0.8 GB was in use by programs but ~4 GB was page cache. The WSL VM is allowed 10 GB and doesn't hand cache back to Windows quickly.
- **Impact:** small, thanks to P-52. The 55k checkpoint is intact, so a resume continues at 55,001 (~3k iterations of work redone).
- **Not restarted automatically:** memory may still be short. The user decides when to resume.
- **Mitigations:** close memory-heavy Windows apps before long runs; stop Label Studio when not labelling; optionally lower the WSL memory limit or enable `autoMemoryReclaim` in `.wslconfig` so cache is returned to Windows.

### P-56
**MVTec carpet: threshold from train-split validation gives 54% false alarms on test**
- **Symptom:** mvtec_carpet image AUROC **0.988**, pixel AUROC 0.968, recall 89/89. But at the protocol threshold (99th percentile of val-good scores) **15 of 28 test-good images are flagged** (false-alarm rate 0.536).
- **Root cause:** val is a 10% hold-out of MVTec's *training* images, which are more alike than MVTec's test-good images, so the val score distribution sits lower than test-good. The ranking is excellent (AUROC 0.988), but the threshold doesn't transfer. The same risk applies to the live demo: lighting and position will differ from the capture sessions.
- **Decision (protocol unchanged):** the threshold is still chosen on val only (ADR-019), and this result is reported as is.
- **Planned mitigation for the app:** an operator **"calibrate" step**. Show ~20 good frames of the product under the current setup; the threshold is set from those (same 99th-percentile rule). That is threshold selection on in-condition good data, which is standard practice for anomaly detection in deployment. To be designed in week 4.

### P-57
**Fabric thread defects invisible to the anomaly model at 256 px (resolution limit)**
- **Symptom:** On the new fabric test set (session 6) the original own_fabric model scored image AUROC 0.805 and caught **thread 4/30**. The retrain with 1.8× zoom augmentation (`own_fabric_s18`) was worse: AUROC 0.729, thread **0/30**.
  - Per-type AUROC (old / s18): colour 1.00/1.00, hole 1.00/1.00, cut 0.875/0.887, thread 0.53/0.31.
  - The new thread defect alone scored 0.30/0.25, i.e. *below* the good photos.
- **Diagnosis (heatmaps, `training/vision/reports/eval_own_fabric_s18.jpg`):**
  - Colour, hole and cut are localised correctly. The cut shows as a sharp hot line, but its peak barely exceeds the diffuse glow that uneven lighting produces on good photos.
  - Thread produces no response at all. A pulled thread is a thin pale loop on a looped fabric. With the whole 2458 px capture square resized to 256 px it is 1-2 px wide, and each heatmap cell covers ~40 px of the original.
  - Zoom augmentation can't add resolution and made the model more tolerant of texture changes.
  - Two things changed in s18 at once (zoom augmentation and the new training photos, including creases), so their individual effects aren't separated.
- **Fix under test: tiling.** `--tiles 2`: the capture square is resized to 512 px and the model is trained on, and scores, 2×2 tiles of 256 px. That doubles the resolution (~20 px per heatmap cell), gives 4× the training patches, and the NPU runs the same 256 px model 4 times (~4 × 15 ms). The image score is the max of the stitched map. Evaluation reads `tiles` from the model's `config.json`.
  - A 20-iteration tiled smoke model already gave fabric image AUROC 0.951 (vs 0.805 for the fully trained untiled model), which supports the resolution diagnosis.
  - The full run (`own_fabric_t2`, 70k iterations, no zoom augmentation) is queued after the YOLO retrain.
- **Where:** `training/vision/train_efficientad.py`, `training/vision/evaluate_efficientad.py`, `training/vision/runs/tiles_exp.sh`

### P-58
**YOLO own-photo results swing a lot between training runs (tiny test groups)**
- **Observation:** YOLO11n was retrained after session 6. The only training change was 30 more good fabric background images, plus 15 good for validation.
  - On the **same** new own test set: v1 mAP50 0.509, v2 0.376.
  - v2 dropped on classes whose data didn't change: wood_liquid 0.906 → 0.488, fabric_hole 0.795 → 0.536.
  - Meanwhile MVTec test went up (0.709 → 0.728) and val was equal (0.701 vs 0.707).
  - fabric_thread (new test group) is ~0 for both.
- **Interpretation:** this is run-to-run variance, not a data effect. Each own test class is one physical defect with 5-20 photos (wood_liquid 5, fabric_hole 5), so the per-class AP of a single run has very wide error bars.
- **Decision:** the model is chosen on **val only** (ADR-019). Val is a tie, so the retrained model (trained on all current data) stays current, and v1 is kept as `models/yolo/yolo11n_combined_v1`. The report gives own-photo YOLO numbers with this caveat; the anomaly model is the primary detector on own products (ADR-005).
- **Possible follow-up:** 3 seeds per configuration and the mean ± spread reported. That needs ~1 h of GPU and would only be worth it if YOLO's own-photo numbers become a decision input.
- **Result (2026-11-22), `own_fabric_t2` (2×2 tiles, 70k iterations, no zoom aug), on the session-6 fabric test set:**
  - Image AUROC **0.879** (was 0.805 untiled). Per type: colour 1.00, **cut 1.00** (was 0.875), hole 1.00, thread 0.66 (was 0.53).
  - At the val threshold: 0/20 false alarms, recall 55/85 (cut 19/20, hole 14/15, colour 20/20, **thread 2/30**). Box hits 55/85.
  - Heatmaps (`reports/eval_own_fabric_t2.jpg`): good photos are now clean (the lighting glow is gone); cut and hole are sharp spots. **Thread still gives no response.** A pulled thread is a pale loop on a looped fabric, and its features match normal loops even at 2× resolution. The new thread defect's AUROC is 0.52 (chance).
  - The 20-iteration smoke model scored higher (0.951). That score came largely from an untrained student, i.e. teacher feature magnitude, and can't be used: choosing training length by test results would break ADR-019.
  - **Decision:** fabric uses 2×2 tiles (4 NPU runs per frame). Thread is an open limitation.
  - **Options:** a raking side light at capture, so a pulled thread casts a shadow (the W2 photo advice); 3×3 tiles (9 runs, ~135 ms); or report thread as not reliably detected.

### P-59
**Quantised EfficientAD output collapsed to a constant (activation ranges calibrated on good images only)**
- **Symptom:** The first trained-model exports (MVTec wood) compiled with 146/147 nodes on the NPU, but every image got the same score (0.191) in both **8a8w and 16a8w**: AUROC 0.5, correlation with fp32 NaN. This explains the earlier smoke test's "threshold 0.0".
- **Diagnosis (stage-by-stage quantisation, 16 calibration images):**
  - The student quantises perfectly (cosine 1.00) and the normalised teacher well (0.98).
  - The student-teacher map collapses on defects: fp32 max **24.1** on a hole image vs quantised **0.022**.
  - Root cause: each activation's range is fixed by calibration, and good images only produce values up to ~0.02, so every defect saturates at the same ceiling. The teacher features clip too (14.4 → 5.2).
  - More activation bits don't help, because the ceiling, not the step size, is wrong. Separately, 8-bit steps over a range wide enough for defects are coarser than good-image map values, so 8a8w can't represent "normal" either.
- **Fix:**
  - **Calibration = 50% good images + 50% good images with synthetic defects** (1-3 patches pasted from the Imagenette penalty pool plus 1-2 random strokes), so the ranges cover defect-sized responses.
  - **16-bit activations** (16a8w, the planned ADR-019 fallback), so small good-image values keep their resolution. 16a8w compiles for SSG2115P.
  - Synthetic defects are used **only** to set quantisation ranges. They are never used for training or evaluation, and test data is never used for calibration.
- **Result (MVTec wood):** 16a8w AUROC **0.945** vs fp32 0.975 (Δ −0.030, just outside the 0.02 target), score correlation 0.92. The other models are being exported.
- **Where:** `training/vision/export_npu_efficientad.py`, `training/vision/runs/export_all.sh`

### P-60
**Evaluation resized images differently from training (aliasing)**
- **Symptom:** For the same model, the export script's fp32 AUROC (0.975) didn't match `evaluate_efficientad.py` (0.985).
- **Root cause:**
  - Training loads images with PIL's bilinear resize, which low-pass filters when shrinking.
  - `efficientad_model.preprocess()`, used by the evaluator, called `F.interpolate(bilinear)` **without antialiasing**. Shrinking 1024 → 256 then aliases, especially fine texture.
  - So every evaluation so far fed the models differently-processed images than they trained on.
- **Fix:** `preprocess()` uses `antialias=True`, which matches training (and the app, which will use OpenCV `INTER_AREA`). All four deployed models were re-evaluated; earlier results are kept as `eval_noaa.json`.
- **Effect** (corrected numbers in the week-3 results):
  - own_fabric_t2: AUROC 0.879 → **0.920**, recall 55 → 68/85, **thread 2 → 13/30**. Aliasing had been erasing thin threads.
  - mvtec_carpet: 0.988 → 0.999 (pixel 0.968 → 0.986).
  - mvtec_wood: 0.985 → 0.976.
  - own_wood: unchanged at 1.000.
- **Lesson:** one preprocessing function for training, evaluation, export and the app. The trainer's PIL loader and `preprocess()` should be merged; noted for the app work.

### P-61
**YOLO NPU export: in-place SiLU left unquantised; segfault inside Ultralytics' validator**
- **Symptoms:**
  1. The QNN quantizer printed "No quant config is implemented for op, aten.silu_.default" for every activation.
  2. The process died with `Segmentation fault` when the accuracy check called `YOLO.val()`.
- **Root causes:**
  1. Ultralytics 8.4.197's SiLU modules run in place (`aten.silu_`), which the QNN quantizer doesn't annotate. They would have stayed unquantised and could fall off the NPU.
  2. A faulthandler trace showed the crash while **importing matplotlib** (`font_manager` → `_mathtext_data`), which the validator does lazily. It happened after the QNN SDK's `envsetup.sh` libraries and ExecuTorch were loaded, i.e. a native-library clash.
- **Fixes:**
  1. Set `inplace = False` on all `nn.SiLU` modules before export; the warnings are gone.
  2. Import matplotlib (Agg backend, `font_manager`, `pyplot`) at the very top of `export_npu_yolo.py`, before any ExecuTorch/QNN import.
  3. `workers=0` for the validator, to avoid forking a process that has the SDK loaded.
- **Where:** `training/vision/export_npu_yolo.py`
- **Also fixed while getting the accuracy check to run:** Ultralytics' rectangular validation batches pad a 320 px image to a 352×352 frame (centred border), but the exported graph is fixed at 320×320 (`Guard failed: x.size()[2] == 320`). The check now crops the centred 320 px region and shifts boxes back, for both graphs.

### P-62
**YOLO11n class scores don't survive QNN quantisation (any precision); YOLO stays fp32 on the CPU**
- **Symptom:** The YOLO 8a8w graph compiles (3.0 MB) but scores **mAP50 0.0** on every split (fp32 export graph: val 0.687, MVTec test 0.730).
- **Investigation (one test image, fp32 vs simulated quantised graph):**
  1. Box and score channels share one output tensor, and a single 8-bit range over 0-320 px boxes would round 0-1 scores to 0. Boxes were normalised to 0-1 inside the graph, as Ultralytics' int8 TFLite export does: boxes cosine 0.999, **scores still exactly 0**.
  2. 16a8w: scores partly survive (cosine 0.86) but are squashed (max 0.12 vs 0.36).
  3. Raw-head export, i.e. NPU outputs box-distribution logits and class logits as two separate tensors and the CPU decodes them with Ultralytics' own `_get_decode_boxes` + sigmoid: 8a8w cosine 0.87, 16a8w 0.86, still squashed.
  4. 16a16w: scores constant 0.5 (worse).
  - So the degradation is inside YOLO11's class branch at any precision, while box regression quantises fine. The quantizer also logs "Could not resolve quant range" for the split/chunk nodes of YOLO11's C3k2/C2PSA blocks. This looks like a quantizer/annotation issue in this toolchain version, not a calibration problem.
- **Decision (ADR-012 update):** YOLO11n runs **fp32 on the CPU (XNNPACK)**: 32 ms at 320 px on the phone (W1), with accuracy identical to fp32. The anomaly models, which need the NPU (CPU 1,419 ms vs NPU ~15 ms per run), stay on the NPU at 16a8w. The XNNPACK `.pte` is in `models/npu/yolo11n_combined_8a8w/yolo11n_combined_fp32_xnnpack.pte` (output rows 0-3 = boxes / 320, normalised; rows 4+ = class scores).
- **Future work (not blocking):** quantise YOLO with mixed precision (class branch kept in fp16/16-bit or skipped), or try a newer QNN SDK/ExecuTorch, whose quantizer has better YOLO11 annotations.
- **Where:** `training/vision/export_npu_yolo.py`; debug scripts were kept out of the repo.

### P-63
**Phone preprocessing with OpenCV `INTER_AREA` dropped fabric AUROC 0.918 → 0.877**
- **Symptom:** The week-4 parity check (`edge/scripts/parity_preprocess.py`) ran the fp32 models on the own test sets with the training preprocessing and with the new phone preprocessing.
  - own_wood: identical AUROC (1.000).
  - **own_fabric_t2: 0.918 vs 0.877.** Score correlation 0.998, but individual scores up to 5× apart. Inputs differed by up to 0.35 (normalised units).
- **Root cause:** Training resizes with **PIL bilinear**, which antialiases with a *triangle* filter whose width scales with the shrink factor. The phone code used OpenCV `INTER_AREA`, a *box* filter. The looped fabric texture sits near the resolution limit, so the different low-pass response changes the texture the model sees. The same family of issue as P-60.
- **Fix:** The phone uses **Pillow for the anomaly resize**, the same function as training. Pillow 13.0 is installed on both the phone (Termux) and the laptop. The parity check now gives **bit-identical inputs** (max abs diff 0.0) and identical scores/AUROC for both products. The YOLO path keeps OpenCV `INTER_AREA`, which is what Ultralytics uses in training.
- **Lesson:** "antialiased resize" isn't one thing. Use the same library function, and check parity at the input-tensor level.
- **Where:** `edge/app/pipeline/preprocess.py`, `edge/scripts/parity_preprocess.py`

### P-64
**vision_server build: duplicate symbols, then an abort at startup ("Re-registering aten::…")**
- **Symptoms:**
  1. Linking failed with hundreds of `duplicate symbol: executorch::backends::xnnpack::…`.
  2. After fixing that, the binary aborted on the phone: `Kernel registration failed … Re-registering aten::_adaptive_avg_pool2d.out`.
- **Root causes:**
  1. ExecuTorch's exported CMake targets already carry `--whole-archive` (and `--no-as-needed` for the QNN backend) in their link options. My extra whole-archive wrapping linked every archive twice.
  2. `${EXECUTORCH_LIBRARIES}` from `executorch-config.cmake` appends **every** optional library it finds, including several kernel *registration* libraries (portable, optimized, optimized_native_cpu, quantized). Each registers the same operators.
- **Fix:** link explicit targets only: `qnn_executorch_backend`, `xnnpack_backend`, **one** CPU kernel library (`optimized_native_cpu_ops_lib`), `quantized_ops_lib` (quantize/dequantize ops only), the module/tensor/data-loader extensions, and `executorch` + `executorch_core`. `build.ninja` now has exactly two kernel registration libraries.
- **Also:** the NDK toolchain file limits `find_package` to the sysroot, so the build passes `-Dexecutorch_DIR=<install>/lib/cmake/ExecuTorch` and `CMAKE_FIND_ROOT_PATH`.
- **Where:** `edge/vision_server/CMakeLists.txt`, `edge/vision_server/build_android.sh`

### P-65
**YOLO XNNPACK `.pte` from our exporter failed at run time ("Attempted to resize a static tensor")**
- **Symptom:** On the phone, `vision_server` loaded YOLO but `forward` failed: `Expected shape (1, 32, 80, 80), but received (320, 32, 1, 80)` inside `XNNExecutor`. The same `.pte` failed identically in ExecuTorch's Python runtime on the laptop, so it was an export problem.
- **Root cause (likely):** XNNPACK split YOLO11 into several delegated segments around its split/chunk ops (the same nodes that troubled the QNN quantizer, P-62). Our wrapper (export-mode Detect + box normalisation) left a dim-order mismatch at one segment boundary.
- **Fix:** export YOLO with **Ultralytics' own ExecuTorch exporter** (`YOLO(best.pt).export(format="executorch", imgsz=320)`, Ultralytics 8.4.197 + ExecuTorch 1.5.0 in the et-venv).
  - On the laptop runtime it matches fp32 PyTorch (boxes max diff 0.0006 px, scores 8e-7).
  - On the phone via `vision_server`: max diff 0.0009.
  - Output: standard `(1, 52, 2100)`, boxes in **pixels** (cx, cy, w, h), then 48 class scores.
- **Where:** `models/npu/yolo_et_export/` (laptop); the model card is updated.

### P-66
**`termux_push.sh` of a folder silently stopped part-way over wireless adb**
- **Symptom:** Pushing the edge app (22 files) and the replay images (196) reported no error, but the phone had 15/22 and 19/196 files (including a missing `categories.toml`).
- **Root cause:** wireless adb drops individual `exec-in` streams under sustained transfer (cf. P-41). The folder push didn't re-verify and resend.
- **Fix:** `spike/laptop/termux_sync.sh <local> <termux dir>`.
  - Lists the phone folder with sizes (with an end marker, P-40), sends only missing or size-mismatched files, and repeats up to 5 rounds until every file matches.
  - Both transfers completed in 2 rounds. All later deployments use it.
- **Where:** `spike/laptop/termux_sync.sh`

### P-67
**First phone replay: inflated defect regions, a "strong" bump that always fired, and a calibration that was too tight**
- **Symptoms** (replay of the own test sets through the full phone pipeline):
  1. Wood liquid defects covered a median **50%** of the square, scratches 12%, so nearly everything got "+1 large".
  2. Defect peaks were a median **15-44× the threshold**, so "+1 strong" (≥ 3×) fired for almost every defect. Results: all wood liquid → SCRAP, all fabric colour → HOLD.
  3. One stain was split into many "defects" (up to 9 regions).
  4. Wood: 7/15 false alarms.
- **Root causes:**
  1. Regions grew down to 0.5 × threshold. With a calibrated threshold that swallowed normal texture, plus the upsampling halo of very strong defects.
  2. The strong ratio was chosen before seeing NPU score scales.
  3. Separate above-threshold fragments of one defect were reported separately.
  4. The replay "calibrated" wood on its 5 val-good photos: near-identical shots of one burst (std 0.013, threshold 0.111), while good test photos score 0.053-0.153.
- **Fixes:**
  1. A region = pixels ≥ threshold; its extent and area count only pixels ≥ max(threshold, 25% of the region's own peak).
  2. `strong_ratio` 3 → **10** (in the severity draft for user review).
  3. Above-threshold pixels within ~3% of the map width are grouped into one region (dilated mask).
  4. **Calibration needs variety:** the Calibrate button pauses every 5 frames and asks the operator to shift or rotate the good item. The margin rule also gained a floor (P-67 unit test): threshold = p99 + max(1 × std, 10% × p99).
  - With 12 varied good frames over HTTP, wood calibrated to 0.169, at which all 15 good test photos pass.
- **Results after the fixes (phone replay, `spike/results/replay_w4.json`):**
  - median defect areas 0.15-2.7%;
  - fabric 1.5-1.9 regions per defect image;
  - fabric good: 20/20 PASS;
  - wood/fabric AUROC 1.000 / 0.914 (laptop 1.000 / 0.917).
- **Where:** `edge/app/pipeline/fusion.py`, `calibrate.py`, `edge/config/severity.toml`, `edge/web/index.html`, `tests/test_edge_pipeline.py` (29 tests)

### P-68
**First live fabric session: camera distance varied 3×, a black frame was judged, and processing ran 2-6× slower**
- **Symptoms** (2026-11-24, 14 live fabric inspections, 3 calibrations):
  1. **Scale:** the texture-scale measure (P-51) gave towel loop periods of **13-47 px**; all training/test photos are 12.6-34.1 px. Five shots were 41-47 px (2-3× closer than training) and blurrier. Good frames scored 0.23-1.68 (replay val good ≤ 0.21), so calibration landed at a threshold of **1.78**, ~7× the replay value. One calibration attempt gave 18.3.
  2. **Black frame:** one frame was fully black (brightness 5/255) and was judged HOLD ("whole image abnormal").
  3. **Speed:** NPU time ~410 ms per inspection (4 tiles) instead of 116 ms; YOLO 93-269 ms instead of ~40; preprocessing 2-3× slower. The server took 0.72-1.1 s per inspection.
- **Diagnosis for the speed:** with Chrome closed, the same image runs at normal speed (NPU 116 ms, total ~0.34 s), and thermal status was 0. With Chrome's camera in the foreground, everything in Termux slowed down, both CPU and NPU calls: background-app CPU limits and/or camera-pipeline contention. The NPU models already request burst mode (ExecuTorch's default `kHtpBurst`), so that isn't it. **Open item:** an A/B measurement with the page open vs closed, then mitigation (e.g. a lower preview resolution with the full-resolution grab only on press).
- **Fix for 1-2:** a **frame check** before every inspection and calibration frame (`edge/app/pipeline/quality.py`).
  - Too dark (mean < 40) → "retake".
  - Fabric texture scale outside **11-36 px** → "camera too CLOSE/FAR, raise/lower the phone".
  - The frame isn't judged or logged. During calibration it isn't counted, and the same frame number is retaken.
  - Blur is not a reject reason (training defect photos are often soft).
  - On the live frames it rejects exactly the black frame and the four 41-47 px shots. On the 310 fabric dataset photos it rejects 3 (≈1%, the period estimator misreads some frames; a retake fixes it). 34 unit tests pass.
- **Also confirmed:** the colour mark was detected and located correctly (REWORK, after the user's colour cap).

### P-69
**Live fabric frames at the right distance still looked "abnormal" everywhere (session 2): the model is tied to its capture conditions**
- **Symptom (2026-11-24, fabric session 2, fixed height):**
  - Texture scale was fine (25-34 px), but the 20 good calibration frames scored **3.1-8.8**; training-condition good photos score ~0.2, and session-1 frames at the right distance scored 0.17-0.42.
  - Calibration landed at a threshold of 10.1, so nothing could be flagged. The colour mark passed; only YOLO noted it.
- **Diagnosis:**
  - Live frames had a pink/warm cast: mean BGR ≈ [128, 137, 146], saturation ~34, vs training ≈ [145, 150, 157], saturation ~18. The loops also looked crisper.
  - The user changed nothing knowingly, so the camera's automatic white balance and exposure most likely shifted between sessions.
  - Global colour matching (per-channel mean/std to training statistics) did **not** help: it raised good training-condition scores from ~0.2 to ~1.0 and left live frames at 2-7.
  - EfficientAD learned "normal" from ~65 photos taken under one set of conditions, so it treats a different camera rendering as an anomaly.
- **Decision (user):** **"teach normal" on the stand.** This is the standard industrial workflow: retrain the anomaly model on good samples from the actual station.
  - **App:** a "Teach good" button saves 60 GOOD frames (frame-checked, every 5 photos the operator moves or rotates the material) exactly like the training photos (1024 px capture square, JPEG q95) into `~/bharatai/teach/<product>/<session>/`.
  - **App:** a "Lock camera" button (exposure / white balance / focus, best effort, as in the capture tool) keeps conditions stable after teaching.
  - **Laptop:** `training/vision/runs/teach_retrain.sh <product>` pulls the photos (`spike/laptop/termux_pull.sh`), splits them by 5-photo groups (every 5th group → val), trains `<product>_teach` (fabric 2×2 tiles, 20k iterations ≈ 15 min), evaluates, exports 16a8w for the NPU, pushes the `.pte` and points `categories.toml` at it. The operator then calibrates as usual.
- **Evaluation caveat:** the old own test photos were taken under the old conditions, so they're only a reference for the taught model. The live test session is the real check.

### P-70
**Taught fabric model, live session 3: small cuts lost in condition drift; the distance check blocked a closer camera**
- **Results** (2026-11-24 16:49-16:57, 9 inspections, taught model `fabric_teach`):
  - **holes 2/2 and colour marks 3/3 detected and boxed correctly**;
  - a **small cut was missed** (score 1.63 vs threshold 1.74, although the heatmap marked it);
  - another cut was flagged for the wrong reason (a dark shadow band at the frame edge);
  - a towel fold drove one HOLD.
- **Root cause:** the conditions drifted between teaching (~14:40; good stand frames scored 0.28-0.56) and calibrating (16:48; good frames scored 0.25-1.55, threshold 1.74). Daylight changed over the afternoon, plus folds and shadows in view, so small defects (1.6-2.4) overlapped the noise.
- **Advice** worked live: 6/9 got Llama advice. It was grounded but generic, because YOLO rarely names live defects (mostly `unknown_anomaly`). It took **40-61 s**, sharing the CPU with Chrome's camera, so `max_new_tokens` was cut from 200 to 140 (the JSON needs ~100).
- **Fixes / procedure:**
  - **Teach** now only checks darkness. The camera may be moved closer to make small defects bigger, and the retrain **sets the product's accepted texture-scale range from the taught photos** (median ± 35%).
  - The retrain uses **only the newest teach session** (`ALL_SESSIONS=1` for all).
  - Procedure: the lamp as the dominant light, Lock camera, flat material with no folds or edges in view, and **teach then calibrate in the same sitting**.

### P-71
**Retrained model not deployed (sync compared sizes), and an old calibration applied to the new model**
- **Symptoms (2026-11-24, second fabric teach):**
  1. The retrain reported "SYNC OK", but the phone still had the *first* taught model: same file name, same size (8.7 MB), different content (MD5 0d570d88… vs 916913be…).
  2. After a restart the app showed fabric as "calibrated" with threshold 1.74, which was calibrated for the previous model.
- **Root causes:**
  1. `termux_sync.sh` decided what to send by **file size**. Retrained QNN models have identical sizes.
  2. Calibrations were stored per product, not per model.
- **Fixes:**
  1. `termux_sync.sh` compares **MD5** (`md5sum` on both sides). The new model was then sent and verified (916913be…).
  2. Each calibration stores the deployed model's identity (file name + size + mtime). The app treats a product as **not calibrated** when the current model file differs, so after any retrain the operator must calibrate again. Unit test added (42 pass).

### P-72
**Taught fabric model at the closer height: defects caught, but false alarms from a too-narrow calibration; the advice queue fell behind**
- **Live session 4** (2026-11-24 18:17-18:23, 12 inspections, phone NPU):
  - **6/6 defect shots flagged**: pulled thread 0.26, colour 1.37-1.57, holes 0.54-0.82, each boxed at the defect.
  - Good towel: 4/6 PASS, **2 false alarms** (0.27, 0.31).
  - Calibration: 20 frames scoring 0.005-0.131 → threshold 0.167.
- **Diagnosis:**
  - The 20 calibration frames were near-identical (slight shifts), so they didn't represent the normal variety of good towel at different spots (up to 0.31). The threshold sat inside normal variation.
  - False-alarm heatmaps show whole-tile "texture noise" (often the top-right tile), not a localised defect.
  - Tried: a **position-aware (per-pixel) calibration** reference. On these frames it made no difference (residual ≈ global score), because the noise moves with the material, not with the frame position. **Not adopted.**
  - With varied calibration frames the threshold would be ~0.33: holes and colour clearly caught, good passes, but the **pulled thread (0.26) would be missed**. It scores like normal texture variation; an accepted limitation (cf. P-57).
- **Advice:** Llama took 47-68 s per answer while the camera ran, so the queue fell behind (2 still pending at the end).
- **Fixes:**
  1. Calibrate asks the operator to move the item to a **new spot** every 5 frames (was "shift slightly").
  2. Advice: only the **newest** waiting inspection gets LLM advice. Older queued ones keep their instant template advice, with the note "LLM skipped: a newer inspection was waiting".
  3. `llama_main` runs at lower CPU priority (`nice 10`) so inspections come first. 43 tests pass.

### P-73
**Live session 5: small cuts and threads passed; image score above threshold but no region reported**
- **Symptom** (2026-11-24 19:19-19:22, varied 20-frame calibration → threshold 0.471; 12 defects + 1 good):
  - only the colour mark was flagged; **7 small cuts and 4 pulled threads PASSED**;
  - three of them (a cut and two threads) scored *above* the threshold (0.50-0.57) yet got PASS with no defect.
- **Root causes:**
  1. **Region filter:** `regions()` dropped above-threshold areas smaller than 0.02% of the map, including the one holding the image's maximum, so tiny defects were scored but never reported.
  2. **Score compression on the NPU:** see P-74.
- **Fix (1):** the region holding the image's maximum is always kept, however small (unit test). Other tiny specks are still ignored.
- **Also tried: 3×3 tiles** (`fabric_teach_t3`, trained on the same taught photos). On the laptop (fp32) it separated defects from good no better than 2×2 (cut/good-max ratio ~1.7-5 vs 2.5-6) at more than twice the NPU time, so it was **not deployed**.
- The 3 "outliers" in the taught validation set (scores 1.1-2.4) were one burst over a spot with a tight hot point (lint or a tiny flaw), so they're not truly good samples; noted for future teach sessions.

### P-74
**NPU (16a8w) clipped real small defects: cut scores compressed 2-4× vs fp32; fixed by widening activation ranges**
- **Evidence:** the same live photos scored on the laptop in fp32 vs a simulation of the deployed quantised graph: cut 1.10 → 0.32, 1.02 → 0.44, 0.48 → 0.28, while good stayed ~0.2. This matches the phone, where the cuts scored 0.23-0.57 inside the good range.
- **What did NOT help:**
  - fewer synthetic calibration frames (15%): still compressed;
  - calibrating with real week-2 defect photos: collapse to a constant (the P-59 failure);
  - running only the feature networks on the NPU with the map computed in fp32 on the CPU: still compressed;
  - 16-bit weights (16a16w): *identical* to 16a8w.
- **Root cause (stage-by-stage trace on a cut tile):**
  - the normalised teacher features reach **8.8** at the cut, but the quantised graph caps them at **6.06**, the largest value seen in calibration (good + synthetic);
  - the clipped features shrink the teacher-student difference, so the map and the score shrink (1.10 → 0.44);
  - the student and autoencoder stages were fine (cosine ≥ 0.999).
- **Fix:** after calibration, every **16-bit per-tensor activation range is widened 3×** around its centre (79 observers); 8-bit per-channel weights are untouched. With 65,536 steps the resolution cost is negligible.
  - Simulation: cut 1.10 → **1.03**, 1.02 → **0.98**, 0.48 → **0.43**, thread 0.64 → **0.63**, good 0.18 → 0.15. 6× was slightly worse than 3×.
  - `export_npu_efficientad.py --widen 3` (default); applies to all future exports, including the wood teach.
- **Correction (same evening): the range-widening fix FAILED on the phone.**
  - Deployed, the widened model returned **0.0 for every image** on the real NPU, although the laptop simulation matched fp32.
  - Likely cause: the QNN quantizer *links* some ranges (e.g. pad/upsample/reshape share their input's scale). Editing each observer separately broke those links; the simulation tolerates that, the compiled HTP graph doesn't. `--widen` is now off by default.
- **Working fix: wider ranges through calibration data.** 25% extra calibration tiles = good tiles with **contrast boosted ×6** around their mean (`--boost-frac 0.25 --boost-gain 6`, the new default). They drive every layer to defect-like magnitudes through normal calibration, so linked ranges stay consistent.
  - Simulation, contrast gain 2.5 / 4 / **6**: cut 1.10 → 0.52 / 0.83 / **1.05**; good 0.18 → 0.15.
  - **Verified on the phone's NPU**, same photos: cut 0.24 → **0.79**, cut 0.33 → **0.80**, cut 0.40 → **0.62**, thread 0.50 → 0.52, good 0.17 → **0.12**. Cuts now score 3-6× above good towel instead of overlapping it. NPU time unchanged (116 ms for 4 tiles).
- **Lessons:**
  - synthetic-defect calibration fixed the gross clipping (P-59) but not per-layer clipping for *real* defects;
  - **always verify quantisation changes on the phone**, not only in simulation.
