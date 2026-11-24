"""Curate own photos: relabel photos filed under the wrong defect type, and move photos that must
not be used (wrong material, wrong defect, confusing marks) into raw/_excluded/. Every change is
recorded with its reason. Nothing is deleted.

    python training/data/curate_own.py            # dry run: list what would change
    python training/data/curate_own.py --apply    # apply + append to training/data/reports/
                                                  #   own_relabels.csv / own_exclusions.csv

Rules are explicit and time-based (capture times from the sidecars), decided on 2026-11-20 after
reviewing the contact sheets (docs/weekly/week-02-data.md -> C):
  - demo board = the light wood board; demo fabric = plain fabrics only (no busy print)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

RAW = Path.home() / "bharatai-build/data/own/raw"
REPORT = Path(__file__).resolve().parent / "reports/own_exclusions.csv"
RELABEL_REPORT = Path(__file__).resolve().parent / "reports/own_relabels.csv"

# Photos filed under the wrong defect type, confirmed with the user.
# (product, from_label, date, start, end, to_label, new_instance_id, reason)
RELABELS = [
    ("wood", "liquid", "2026-11-21", "18:21:00", "18:21:59", "color", 3,
     "blue ink/marker stroke is a sharp foreign-colour mark (color), not a soaked-in stain (liquid)"),
]

# (product, label, date, start HH:MM:SS, end HH:MM:SS inclusive, reason). Rules are date-specific:
# a time-only rule would also catch later sessions taken at the same time of day (P-42).
RULES = [
    ("fabric", "*", "2026-11-20", "15:00:00", "23:59:59", "printed (busy-pattern) fabric - only plain fabrics are used"),
    ("wood", "good", "2026-11-20", "15:44:00", "15:49:59", "other boards (grey, dark) - demo board is the light board"),
    ("wood", "hole", "2026-11-20", "13:08:30", "13:08:59", "pencil lines around the holes (colour marks) + other board + shadow"),
    ("wood", "hole", "2026-11-20", "15:52:00", "15:53:59", "pencil rectangle around the holes (colour marks) + shadow"),
    ("wood", "color", "2026-11-20", "13:11:00", "13:11:59", "peeled/gouged patch - surface damage, not a colour mark"),
    ("wood", "scratch", "2026-11-20", "00:00:00", "23:59:59", "board edge and gap between boards, not a surface scratch"),
    # decided 2026-11-21: one fabric = one "normal"; the looped fabric from session 3 is the demo fabric
    ("fabric", "*", "2026-11-20", "00:00:00", "14:59:59", "older plain fabrics (white, grey) - demo fabric is the looped fabric"),
    # session 4 (2026-11-21 evening)
    ("wood", "good", "2026-11-21", "18:26:00", "18:27:59", "dark brown board in low light - demo board is the light board"),
    ("wood", "scratch", "2026-11-21", "18:26:00", "18:26:59", "dark brown board in low light; scratch barely visible"),
    ("wood", "color", "2026-11-21", "18:21:00", "18:21:29", "dark shadow + background strip in frame"),
    # session 5 (scratch): user chose to keep only the 4 clean frames
    ("wood", "scratch", "2026-11-21", "18:45:20", "18:45:30", "hand/phone shadow at the frame edge"),
    ("wood", "scratch", "2026-11-21", "18:45:34", "18:45:34", "hand/phone shadow at the frame edge"),
    # session 6 (2026-11-22, close-distance fabric good + thread, P-51): user chose to set aside the
    # 3 bursts held too close to focus (loops 2-3x defect scale, sharpness 31-74 vs 150-900)
    ("fabric", "good", "2026-11-22", "12:46:33", "12:46:38", "out of focus - camera closer than minimum focus distance"),
    ("fabric", "good", "2026-11-22", "12:46:50", "12:46:54", "out of focus - camera closer than minimum focus distance"),
    ("fabric", "good", "2026-11-22", "12:47:16", "12:47:20", "out of focus - camera closer than minimum focus distance"),
]


def relabel(side: Path, meta: dict, to_label: str, instance: int) -> Path:
    """Move a photo + sidecar to another label folder, renaming it and updating its sidecar."""
    old_label = meta["label"]
    old_part = f"_{old_label}_i{meta['instance_id']:03d}_" if meta.get("instance_id") else f"_{old_label}_"
    new_name = side.stem.replace(old_part, f"_{to_label}_i{instance:03d}_", 1)
    dest = RAW / meta["product"] / to_label
    dest.mkdir(parents=True, exist_ok=True)
    meta["relabeled_from"] = {"label": old_label, "instance_id": meta.get("instance_id"), "file": meta.get("file")}
    meta.update(label=to_label, instance_id=instance, file=f"{new_name}.jpg")
    shutil.move(str(side.with_suffix(".jpg")), dest / f"{new_name}.jpg")
    (dest / f"{new_name}.json").write_text(json.dumps(meta, indent=2))
    side.unlink()
    return dest / f"{new_name}.jpg"


def matches(meta: dict, product: str, label: str, date: str, start: str, end: str) -> bool:
    d, t = meta["captured_at"][:10], meta["captured_at"][11:19]
    return (meta["product"] == product and (label == "*" or meta["label"] == label)
            and d == date and start <= t <= end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    # 1. relabels first, so later exclusion rules see the corrected label
    relabels = []
    for side in sorted(RAW.glob("*/*/*.json")):
        meta = json.loads(side.read_text())
        for product, frm, date, start, end, to, inst, reason in RELABELS:
            if matches(meta, product, frm, date, start, end):
                relabels.append((side, meta, to, inst, reason))
                break
    if relabels:
        print(f"{len(relabels):4d}  relabel {relabels[0][1]['product']}/{relabels[0][1]['label']} -> "
              f"{relabels[0][2]} (instance {relabels[0][3]}): {relabels[0][4]}")
    if args.apply and relabels:
        RELABEL_REPORT.parent.mkdir(parents=True, exist_ok=True)
        new = not RELABEL_REPORT.exists()
        with RELABEL_REPORT.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["old_file", "new_file", "product", "old_label", "new_label", "new_instance_id",
                            "captured_at", "reason"])
            for side, meta, to, inst, reason in relabels:
                old_file, old_label, at = side.with_suffix(".jpg").name, meta["label"], meta["captured_at"]
                new_path = relabel(side, meta, to, inst)
                w.writerow([old_file, new_path.name, meta["product"], old_label, to, inst, at, reason])
        print(f"relabels recorded in {RELABEL_REPORT}")

    # 2. exclusions
    moves = []
    for side in sorted(RAW.glob("*/*/*.json")):
        meta = json.loads(side.read_text())
        for product, label, date, start, end, reason in RULES:
            if matches(meta, product, label, date, start, end):
                moves.append((side, meta, reason))
                break

    by_reason: dict[str, int] = {}
    for _, _, reason in moves:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, n in by_reason.items():
        print(f"{n:4d}  {reason}")
    print(f"{len(moves)} photos {'moved' if args.apply else 'would move'} to {RAW / '_excluded'}")
    if not args.apply:
        return

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    new = not REPORT.exists()
    with REPORT.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "product", "label", "instance_id", "captured_at", "reason"])
        for side, meta, reason in moves:
            dest = RAW / "_excluded" / meta["product"] / meta["label"]
            dest.mkdir(parents=True, exist_ok=True)
            for f in (side.with_suffix(".jpg"), side):
                shutil.move(str(f), dest / f.name)
            w.writerow([side.with_suffix(".jpg").name, meta["product"], meta["label"],
                        meta.get("instance_id") or "", meta["captured_at"], reason])
    print(f"recorded in {REPORT}")


if __name__ == "__main__":
    main()
