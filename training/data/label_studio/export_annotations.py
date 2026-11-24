"""Export the wood/fabric labelling projects from the local Label Studio and validate them.

    ~/bharatai-build/ls-venv/bin/python training/data/label_studio/export_annotations.py

Saves raw exports to ~/bharatai-build/data/own/labels/ls_export_<product>.json and a flat file
boxes_<product>.json: one entry per photo {file, folder_label, status, boxes:[{label, x, y, w, h}]},
with boxes in pixels of the ORIGINAL photo. Prints a validation report:
  - photos without an annotation (not labelled) or skipped
  - boxes whose label differs from the photo's folder (possible mislabel)
  - photos annotated with zero boxes, degenerate boxes
Exit code 1 if anything needs attention.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup_projects as sp  # noqa: E402

OUT = sp.WORK / "data/own/labels"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = sp.session()
    problems = 0
    for product in sp.PROJECTS:
        project = sp.find_project(s, f"{product} defects")
        r = s.get(f"{sp.BASE}/api/projects/{project['id']}/export",
                  params={"exportType": "JSON", "download_all_tasks": "true"}, timeout=120)
        r.raise_for_status()
        tasks = r.json()
        (OUT / f"ls_export_{product}.json").write_text(json.dumps(tasks, indent=1))

        flat, status_count, per_label = [], Counter(), Counter()
        print(f"\n== {product}: {len(tasks)} tasks")
        for t in tasks:
            path = t["data"]["image"]                       # /data/local-files/?d=raw/<product>/<label>/<file>
            file = path.rsplit("/", 1)[-1]
            folder = path.rsplit("/", 2)[-2]
            anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
            skipped = any(a.get("was_cancelled") for a in t.get("annotations", []))
            if not anns:
                status = "skipped" if skipped else "unlabelled"
                status_count[status] += 1
                if status == "unlabelled":
                    print(f"  UNLABELLED  {folder}/{file}")
                    problems += 1
                flat.append({"file": file, "folder_label": folder, "status": status, "boxes": []})
                continue
            ann = max(anns, key=lambda a: a.get("updated_at") or "")   # latest annotation wins
            boxes = []
            for res in ann.get("result", []):
                if res.get("type") != "rectanglelabels":
                    continue
                v = res["value"]
                W, H = res["original_width"], res["original_height"]
                label = v["rectanglelabels"][0]
                box = {"label": label, "x": v["x"] / 100 * W, "y": v["y"] / 100 * H,
                       "w": v["width"] / 100 * W, "h": v["height"] / 100 * H, "img_w": W, "img_h": H}
                if box["w"] < 4 or box["h"] < 4:
                    print(f"  TINY BOX    {folder}/{file}: {box['w']:.0f}x{box['h']:.0f}px")
                    problems += 1
                if label != folder:
                    print(f"  LABEL≠DIR   {folder}/{file}: box labelled '{label}'")
                    problems += 1
                per_label[label] += 1
                boxes.append(box)
            if not boxes:
                print(f"  NO BOXES    {folder}/{file} (submitted without a box)")
                problems += 1
            status_count["labelled"] += 1
            flat.append({"file": file, "folder_label": folder, "status": "labelled", "boxes": boxes})
        (OUT / f"boxes_{product}.json").write_text(json.dumps(flat, indent=1))
        print(f"  status: {dict(status_count)}   boxes per label: {dict(per_label)}")
    print(f"\n{'OK - no problems' if problems == 0 else f'{problems} item(s) need attention'}; exports in {OUT}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
