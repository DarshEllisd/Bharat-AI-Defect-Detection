"""Create (or update) the wood and fabric labelling projects in the local Label Studio, and point
each one at the captured defect photos on disk. Safe to re-run: after new photos are pulled,
run with --sync to add them as tasks.

    ~/bharatai-build/ls-venv/bin/python training/data/label_studio/setup_projects.py          # create
    ~/bharatai-build/ls-venv/bin/python training/data/label_studio/setup_projects.py --sync   # import photos

Requires start_label_studio.sh to be running. Photos are served from disk via Label Studio's
local-files storage (no copies). Only defect folders are imported: `good` photos need no boxes.
"""
import argparse
import os
import sys
from pathlib import Path

import requests

WORK = Path(os.environ.get("WORK", Path.home() / "bharatai-build"))
CRED = WORK / "label-studio/credentials.env"
OWN_RAW = WORK / "data/own/raw"
BASE = os.environ.get("LS_URL", "http://127.0.0.1:8080")

# Same names as the capture tool and MVTec (ADR-016/017)
PROJECTS = {
    "wood": ["scratch", "hole", "liquid", "color"],
    "fabric": ["cut", "hole", "color", "thread"],
}
COLOURS = ["#e8601c", "#2e86de", "#20bf6b", "#a55eea"]
INSTRUCTIONS = (
    "Draw one tight box around each defect. If a photo shows two separate defects, draw two boxes. "
    "Pick the label that matches the folder the photo came from (shown in the image path). "
    "Skip (or mark) photos that are blurry or where the defect is not visible."
)


def label_config(labels: list[str]) -> str:
    tags = "\n".join(f'    <Label value="{l}" background="{c}"/>' for l, c in zip(labels, COLOURS))
    return f"""<View>
  <Header value="$image"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="3">
{tags}
  </RectangleLabels>
</View>"""


def session() -> requests.Session:
    """Authenticate with the legacy API token if enabled, else log in like the browser does.

    Label Studio 1.23 disables legacy tokens by default (P-37), so the fallback is a Django
    session: fetch the login page for a CSRF cookie, post the credentials, then send the
    rotated CSRF token on every write.
    """
    creds = dict(line.split("=", 1) for line in CRED.read_text().split() if "=" in line)
    s = requests.Session()
    s.headers["Authorization"] = f"Token {creds['LS_TOKEN']}"
    if s.get(f"{BASE}/api/projects", timeout=10).status_code != 401:
        return s
    del s.headers["Authorization"]
    login = f"{BASE}/user/login/"
    s.get(login, timeout=10).raise_for_status()
    r = s.post(login, timeout=10, headers={"Referer": login}, allow_redirects=False,
               data={"email": creds["LS_USER"], "password": creds["LS_PASSWORD"],
                     "csrfmiddlewaretoken": s.cookies.get("csrftoken", "")})
    if "sessionid" not in s.cookies:
        sys.exit(f"Label Studio login failed (HTTP {r.status_code}). Is start_label_studio.sh running "
                 "with the credentials in credentials.env?")
    s.headers["X-CSRFToken"] = s.cookies.get("csrftoken", "")
    s.headers["Referer"] = BASE
    s.get(f"{BASE}/api/projects", timeout=10).raise_for_status()
    return s


def find_project(s: requests.Session, title: str) -> dict | None:
    r = s.get(f"{BASE}/api/projects", params={"page_size": 100}, timeout=10)
    r.raise_for_status()
    data = r.json()
    items = data.get("results", data) if isinstance(data, dict) else data
    return next((p for p in items if p["title"] == title), None)


def ensure_project(s: requests.Session, product: str, labels: list[str]) -> int:
    title = f"{product} defects"
    body = {"title": title, "label_config": label_config(labels), "expert_instruction": INSTRUCTIONS,
            "show_instruction": True}
    existing = find_project(s, title)
    if existing:
        s.patch(f"{BASE}/api/projects/{existing['id']}", json=body, timeout=10).raise_for_status()
        return existing["id"]
    r = s.post(f"{BASE}/api/projects", json=body, timeout=10)
    r.raise_for_status()
    return r.json()["id"]


def ensure_storages(s: requests.Session, project_id: int, product: str, labels: list[str], sync: bool) -> None:
    r = s.get(f"{BASE}/api/storages/localfiles", params={"project": project_id}, timeout=10)
    r.raise_for_status()
    existing = {st["path"]: st for st in r.json()}
    for label in labels:
        folder = OWN_RAW / product / label
        folder.mkdir(parents=True, exist_ok=True)
        st = existing.get(str(folder))
        if st is None:
            r = s.post(f"{BASE}/api/storages/localfiles", timeout=10, json={
                "project": project_id, "title": f"{product}/{label}", "path": str(folder),
                "regex_filter": r".*\.jpg$", "use_blob_urls": True})
            r.raise_for_status()
            st = r.json()
        if sync:
            s.post(f"{BASE}/api/storages/localfiles/{st['id']}/sync", timeout=120).raise_for_status()
    n = s.get(f"{BASE}/api/projects/{project_id}", timeout=10).json().get("task_number", "?")
    print(f"[ok] project '{product} defects' (id {project_id}): {n} tasks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="import photos currently on disk as tasks")
    args = parser.parse_args()
    s = session()
    for product, labels in PROJECTS.items():
        pid = ensure_project(s, product, labels)
        ensure_storages(s, pid, product, labels, args.sync)


if __name__ == "__main__":
    main()
