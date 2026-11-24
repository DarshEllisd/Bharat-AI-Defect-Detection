"""W2 capture tool - photograph real wood/fabric samples with labels, from the phone itself.

Runs in Termux (aiohttp, ADR-011); Chrome on the same phone opens http://localhost:8001.

    python ~/bharatai/capture/server.py            # start it from the Termux app
    python ~/bharatai/capture/server.py --port 8001 --root ~/bharatai/data/raw

Each photo is saved as  <root>/<product>/<label>/<name>.jpg  plus a JSON sidecar
(product, label, defect instance id, square ROI, frame size, camera settings, session, time).
The defect instance id groups all photos of one physical defect, so splits can be done per
instance and the same scratch never lands in both train and test (ADR-017).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
# Defect types follow MVTec's names (wood; fabric uses MVTec carpet's) - ADR-016
PRODUCTS = {
    "wood": ["good", "scratch", "hole", "liquid", "color"],
    "fabric": ["good", "cut", "hole", "color", "thread"],
}
TARGETS = {"good": 120, "defect_photos": 10, "defect_instances": 3}


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history: list[tuple[Path, Path]] = []  # for undo, newest last

    def counts(self) -> dict:
        out = {}
        for product, labels in PRODUCTS.items():
            out[product] = {}
            for label in labels:
                d = self.root / product / label
                photos, instances = 0, set()
                if d.exists():
                    photos = sum(1 for _ in d.glob("*.jpg"))
                    for side in d.glob("*.json"):
                        try:
                            inst = json.loads(side.read_text()).get("instance_id")
                        except (OSError, json.JSONDecodeError):
                            continue
                        if inst:
                            instances.add(inst)
                out[product][label] = {"photos": photos, "instances": len(instances),
                                       "max_instance": max(instances, default=0)}
        return out

    def save(self, jpeg: bytes, product: str, label: str, instance: int | None, seq: int, meta: dict) -> Path:
        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        inst = f"_i{instance:03d}" if instance else ""
        name = f"{product}_{label}{inst}_{stamp}_{seq:02d}"
        d = self.root / product / label
        d.mkdir(parents=True, exist_ok=True)
        jpg, side = d / f"{name}.jpg", d / f"{name}.json"
        jpg.write_bytes(jpeg)
        side.write_text(json.dumps({
            "file": jpg.name, "product": product, "label": label, "instance_id": instance,
            "burst_index": seq, "session": self.session, "captured_at": now.isoformat(timespec="milliseconds"),
            "bytes": len(jpeg), **meta,
        }, indent=2))
        self.history.append((jpg, side))
        return jpg

    def undo(self) -> str | None:
        if not self.history:
            return None
        jpg, side = self.history.pop()
        for p in (jpg, side):
            p.unlink(missing_ok=True)
        return jpg.name


def make_app(store: Store) -> web.Application:
    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(HERE / "index.html")

    async def config(request: web.Request) -> web.Response:
        return web.json_response({"products": PRODUCTS, "targets": TARGETS, "session": store.session,
                                  "root": str(store.root)})

    async def counts(request: web.Request) -> web.Response:
        return web.json_response(store.counts())

    async def capture(request: web.Request) -> web.Response:
        q = request.query
        product, label = q.get("product", ""), q.get("label", "")
        if label not in PRODUCTS.get(product, []):
            return web.json_response({"error": f"unknown product/label {product}/{label}"}, status=400)
        instance = None
        if label != "good":
            try:
                instance = int(q.get("instance", "0"))
            except ValueError:
                instance = 0
            if instance < 1:
                return web.json_response({"error": "defect photos need an instance id >= 1"}, status=400)
        jpeg = await request.read()
        if not jpeg.startswith(b"\xff\xd8"):
            return web.json_response({"error": "body is not a JPEG"}, status=400)
        try:
            meta = json.loads(request.headers.get("X-Capture-Meta", "{}"))
        except json.JSONDecodeError:
            meta = {}
        saved = store.save(jpeg, product, label, instance, int(q.get("seq", "0")), meta)
        return web.json_response({"saved": saved.name, "counts": store.counts()})

    async def undo(request: web.Request) -> web.Response:
        removed = store.undo()
        return web.json_response({"removed": removed, "counts": store.counts()})

    app = web.Application(client_max_size=32 * 1024 * 1024)
    app.add_routes([web.get("/", index), web.get("/api/config", config), web.get("/api/counts", counts),
                    web.post("/api/capture", capture), web.post("/api/undo", undo)])
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--root", type=Path, default=Path.home() / "bharatai" / "data" / "raw")
    args = parser.parse_args()
    store = Store(args.root.expanduser())
    print(f"capture tool: open http://localhost:{args.port} in Chrome  (saving to {store.root}, session {store.session})")
    web.run_app(make_app(store), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
