"""Client for vision_server (ADR-022): models stay resident in a native process; Python sends float32
tensors over a Unix domain socket. Protocol documented in edge/vision_server/vision_server.cpp.

    srv = VisionServer.start(bin_dir, npu_dir, socket_path)   # must run inside the Termux app (NPU)
    c = VisionClient(socket_path); c.load("anomaly", "/path/model.pte")
    outs, us = c.infer("anomaly", x)                           # x: float32 ndarray -> list of ndarrays
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

REQ_MAGIC, REP_MAGIC = 0x51525356, 0x50525356
LOAD, INFER, UNLOAD, PING, QUIT = 1, 2, 3, 4, 5


class VisionError(RuntimeError):
    pass


class VisionClient:
    def __init__(self, socket_path: str, timeout: float = 30.0) -> None:
        self.path = socket_path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(socket_path)

    # --- framing -----------------------------------------------------------------------------
    def _recv_into(self, view: memoryview) -> None:
        """Fill `view` completely (zero-copy into the destination buffer)."""
        got = 0
        while got < len(view):
            n = self.sock.recv_into(view[got:], len(view) - got)
            if n == 0:
                raise VisionError("vision_server closed the connection")
            got += n

    def _recv(self, n: int) -> bytes:
        buf = bytearray(n)
        self._recv_into(memoryview(buf))
        return bytes(buf)

    def _request(self, cmd: int, name: str, payload: bytes = b"", data: np.ndarray | None = None
                 ) -> tuple[str, int, list[np.ndarray]]:
        nb = name.encode()
        self.sock.sendall(struct.pack("<III", REQ_MAGIC, cmd, len(nb)) + nb + payload)
        if data is not None:  # tensor body sent straight from the array's memory (no tobytes copy)
            self.sock.sendall(memoryview(data).cast("B"))
        magic, status, msg_len = struct.unpack("<IiI", self._recv(12))
        if magic != REP_MAGIC:
            raise VisionError("bad reply magic")
        msg = self._recv(msg_len).decode(errors="replace")
        us, n_out = struct.unpack("<QI", self._recv(12))
        outs = []
        for _ in range(n_out):
            (ndim,) = struct.unpack("<I", self._recv(4))
            dims = struct.unpack(f"<{ndim}I", self._recv(4 * ndim))
            (nbytes,) = struct.unpack("<Q", self._recv(8))
            arr = np.empty(nbytes // 4, dtype=np.float32)
            self._recv_into(memoryview(arr).cast("B"))
            outs.append(arr.reshape(dims))
        if status != 0:
            raise VisionError(f"vision_server error {status}: {msg}")
        return msg, us, outs

    # --- commands ----------------------------------------------------------------------------
    def ping(self) -> str:
        return self._request(PING, "")[0]

    def load(self, name: str, pte_path: str) -> int:
        p = str(pte_path).encode()
        return self._request(LOAD, name, struct.pack("<I", len(p)) + p)[1]

    def unload(self, name: str) -> None:
        self._request(UNLOAD, name)

    def infer(self, name: str, x: np.ndarray) -> tuple[list[np.ndarray], int]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        hdr = struct.pack(f"<I{x.ndim}IQ", x.ndim, *x.shape, x.nbytes)
        _, us, outs = self._request(INFER, name, hdr, data=x)
        return outs, us

    def quit(self) -> None:
        try:
            self._request(QUIT, "")
        finally:
            self.close()

    def close(self) -> None:
        self.sock.close()


class VisionServer:
    """Starts vision_server with the environment the NPU needs (from inside the Termux app, P-15)."""

    def __init__(self, proc: subprocess.Popen, socket_path: str) -> None:
        self.proc, self.socket_path = proc, socket_path

    @classmethod
    def start(cls, bin_dir: str, npu_dir: str, socket_path: str, log_path: str | None = None,
              wait_s: float = 10.0) -> "VisionServer":
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{npu_dir}/lib:/vendor/lib64:/system/lib64"
        dsp = ";".join(str(d) for d in sorted(Path(npu_dir, "dsp").glob("v*")))
        env["ADSP_LIBRARY_PATH"] = f"{dsp};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp;/dsp"
        log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        proc = subprocess.Popen([f"{bin_dir}/vision_server", socket_path], env=env, stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if proc.poll() is not None:
                raise VisionError(f"vision_server exited with code {proc.returncode} (see {log_path})")
            if os.path.exists(socket_path):
                try:
                    VisionClient(socket_path).ping()
                    return cls(proc, socket_path)
                except OSError:
                    pass
            time.sleep(0.1)
        proc.kill()
        raise VisionError("vision_server did not come up")

    def stop(self) -> None:
        try:
            VisionClient(self.socket_path, timeout=3).quit()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
