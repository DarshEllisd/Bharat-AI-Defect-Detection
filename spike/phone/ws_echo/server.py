"""W1 spike - camera -> WebSocket -> backend round trip, served from Termux.

Chrome on the phone opens http://localhost:8000, streams JPEG frames from the camera
over /ws/frames; the server decodes each frame (as the real pipeline will) and replies
with timing. The page POSTs its summary to /results, saved under ~/bharatai/results/.

    python ~/bharatai/spike/ws_echo/server.py                    # auto: FastAPI, else aiohttp
    python ~/bharatai/spike/ws_echo/server.py --framework aiohttp
"""
import argparse
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("BHARATAI_RESULTS", Path.home() / "bharatai" / "results"))


def make_decoder():
    """Return (name, fn) where fn(jpeg_bytes) -> (height, width)."""
    try:
        import cv2
        import numpy as np

        return "opencv", lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    except ImportError:
        pass
    try:
        from PIL import Image

        return "pillow", lambda b: Image.open(io.BytesIO(b)).convert("RGB").size[::-1]
    except ImportError:
        return "none", lambda b: (0, 0)


DECODER_NAME, decode = make_decoder()
SERVER_INFO = {"decoder": DECODER_NAME, "framework": None}


def handle_frame(data: bytes) -> str:
    t0 = time.perf_counter()
    h, w = decode(data)
    return json.dumps({"decode_ms": round((time.perf_counter() - t0) * 1000, 2),
                       "bytes": len(data), "w": int(w), "h": int(h)})


def save_results(payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"ws_echo_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"server": SERVER_INFO, **payload}, indent=2))
    print(f"saved {path}")
    return path


def run_fastapi(host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    app = FastAPI()

    @app.get("/")
    async def index():
        return FileResponse(HERE / "index.html")

    @app.get("/info")
    async def info():
        return SERVER_INFO

    @app.post("/results")
    async def results(request: Request):
        return {"saved": str(save_results(await request.json()))}

    @app.websocket("/ws/frames")
    async def frames(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                await ws.send_text(handle_frame(await ws.receive_bytes()))
        except WebSocketDisconnect:
            pass

    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_aiohttp(host: str, port: int) -> None:
    from aiohttp import WSMsgType, web

    async def index(request):
        return web.FileResponse(HERE / "index.html")

    async def info(request):
        return web.json_response(SERVER_INFO)

    async def results(request):
        return web.json_response({"saved": str(save_results(await request.json()))})

    async def frames(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                await ws.send_str(handle_frame(msg.data))
        return ws

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.add_routes([web.get("/", index), web.get("/info", info),
                    web.post("/results", results), web.get("/ws/frames", frames)])
    web.run_app(app, host=host, port=port, print=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--framework", choices=["auto", "fastapi", "aiohttp"], default="auto")
    args = parser.parse_args()

    framework = args.framework
    if framework == "auto":
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401

            framework = "fastapi"
        except ImportError:
            framework = "aiohttp"
    SERVER_INFO["framework"] = framework
    print(f"framework={framework} decoder={DECODER_NAME} -> open http://localhost:{args.port} in Chrome")
    (run_fastapi if framework == "fastapi" else run_aiohttp)(args.host, args.port)


if __name__ == "__main__":
    main()
