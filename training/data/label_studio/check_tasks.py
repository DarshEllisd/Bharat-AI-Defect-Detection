"""Check that each labelling project has tasks and that a task's photo is actually served."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_projects as sp  # noqa: E402

s = sp.session()
for title in ("wood defects", "fabric defects"):
    project = sp.find_project(s, title)
    tasks = s.get(f"{sp.BASE}/api/tasks", params={"project": project["id"], "page_size": 1}, timeout=10).json()
    tasks = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
    url = tasks[0]["data"]["image"]
    r = s.get(sp.BASE + url if url.startswith("/") else url, timeout=30)
    ok = r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    print(f"{title}: {project.get('task_number')} tasks; sample image HTTP {r.status_code}, "
          f"{len(r.content) / 1e6:.1f} MB, {'valid JPEG' if ok else 'NOT a valid JPEG'}")
