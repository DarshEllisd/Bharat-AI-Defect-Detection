"""Self-test of the photo import path: put one image in wood/scratch, sync, check the task and that
its image is served, then delete the task and the file again. Leaves the projects untouched."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_projects as sp  # noqa: E402

src = sorted((Path.home() / "bharatai-build/data/mvtec/wood/test/scratch").glob("*.png"))[0]
dst = sp.OWN_RAW / "wood" / "scratch" / "_selftest.jpg"
dst.parent.mkdir(parents=True, exist_ok=True)
Image.open(src).convert("RGB").save(dst, quality=90)

s = sp.session()
pid = sp.find_project(s, "wood defects")["id"]
sp.ensure_storages(s, pid, "wood", sp.PROJECTS["wood"], sync=True)
tasks = s.get(f"{sp.BASE}/api/tasks", params={"project": pid, "page_size": 50}, timeout=10).json()
tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
mine = [t for t in tasks if "_selftest.jpg" in str(t["data"].get("image", ""))]
assert mine, f"self-test image not imported; tasks={tasks}"
url = mine[0]["data"]["image"]
img = s.get(f"{sp.BASE}{url}" if url.startswith("/") else url, timeout=10)
print(f"task {mine[0]['id']} image url -> HTTP {img.status_code}, {len(img.content)} bytes, "
      f"content-type {img.headers.get('content-type')}")
assert img.status_code == 200 and img.content[:2] == b"\xff\xd8", "image not served"

for t in mine:
    s.delete(f"{sp.BASE}/api/tasks/{t['id']}", timeout=10).raise_for_status()
dst.unlink()
left = s.get(f"{sp.BASE}/api/projects/{pid}", timeout=10).json().get("task_number")
print(f"SELFTEST OK - cleaned up, wood project tasks now: {left}")
