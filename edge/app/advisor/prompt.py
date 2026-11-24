"""Advisory pipeline, text side (week 5, ADR-007): prompt for Llama 3.2 1B Instruct, output parsing/repair,
grounding check against the knowledge base, and a templated fallback.

Grounding rule: the LLM may only recommend actions from config/kb/<rules>.toml, cited by their id (e.g. "W-H1").
Output = JSON {summary, probable_cause, recommended_actions[ids], log_note}. If the output is not valid JSON
(after a small repair: Llama often stops before the closing brace), or cites no known action id, the templated
fallback is used, so an alert always has advice.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[2] / "config/kb"
MAX_DEFECTS = 3
# The answer is pre-filled with the start of the JSON object: without it Llama 3.2 1B answered in Markdown
# (week-5 phone test). The generated text continues from here, so callers prepend PREFILL before parsing.
PREFILL = '{"summary": "'
SYSTEM = ("You are a quality-control assistant on a factory inspection station. You receive a structured defect "
          "record from a vision system and notes from the plant knowledge base. Explain the defect to the operator in "
          "plain English and recommend corrective actions. Use ONLY actions listed in the knowledge base notes and "
          "cite them by their id. Do not change the severity or disposition. Reply with a single JSON object with "
          'keys: "summary" (one sentence), "probable_cause" (one sentence), "recommended_actions" (list of 1-3 '
          'action ids, e.g. ["W-H1"]), "log_note" (max 20 words).')


def load_kb(rules: str) -> dict:
    with open(KB_DIR / f"{rules}.toml", "rb") as fh:
        return tomllib.load(fh)


def _kb_entry(kb: dict, dtype: str) -> tuple[str, dict]:
    return (dtype, kb[dtype]) if dtype in kb else ("unknown_anomaly", kb["unknown_anomaly"])


def relevant(record: dict, kb: dict) -> list[tuple[dict, str, dict]]:
    """Defects that drive the verdict (NOTE-only YOLO detections excluded), worst first, at most MAX_DEFECTS,
    one per KB type."""
    out, seen = [], set()
    for d in sorted(record["defects"], key=lambda x: -x["level"]):
        if d.get("severity") == "NOTE":
            continue
        t, e = _kb_entry(kb, d["type"])
        if t not in seen:
            out.append((d, t, e)); seen.add(t)
        if len(out) == MAX_DEFECTS:
            break
    return out


def build_prompt(record: dict, kb: dict, product_label: str) -> str:
    items = relevant(record, kb)
    rec = {"inspection_id": record["id"], "product": product_label, "disposition": record["disposition"],
           "severity": record["severity"],
           "defects": [{"type": d["type"], "severity": d["severity"], "area_percent": round(100 * d["area_frac"], 2),
                        "detected_by": d["source"]} for d, _, _ in items]}
    notes = []
    for _, t, e in items:
        acts = " ".join(f"({a['id']}) {a['text']}" for a in e["actions"])
        notes.append(f"KNOWLEDGE BASE NOTES ({t}):\n- Typical causes: {'; '.join(e['causes'])}.\n- Actions: {acts}")
    user = "DEFECT RECORD:\n" + json.dumps(rec) + "\n\n" + "\n\n".join(notes)
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + PREFILL)


def extract_markdown(text: str) -> dict | None:
    """Safety net for a Markdown-style answer (**Summary:** ... **Probable Cause:** ... (ID) actions ...)."""
    text = text.split("<|eot_id|>")[0]

    def field(name: str) -> str:
        m = re.search(rf"{name}\W*:?\**\s*(.+)", text, re.I)
        return m.group(1).strip(" *") if m else ""
    ids = re.findall(r"[A-Z]-[A-Z]\d+", text)
    summary = field("summary")
    if not summary or not ids:
        return None
    return {"summary": summary, "probable_cause": field(r"probable\s+cause"), "recommended_actions": ids,
            "log_note": field(r"log\s+note")}


def extract_json(text: str) -> dict | None:
    """First JSON object in the generated text; repairs a missing closing brace/bracket (seen in W1)."""
    text = text.split("<|eot_id|>")[0]
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:].strip()
    for candidate in (body, body + "}", body + "]}", body.rstrip(",") + "}", body + '"}'):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", body, re.S)          # trailing chatter after a complete object
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def ground(obj: dict, record: dict, kb: dict) -> dict | None:
    """Validated advice or None. Action ids must exist in the KB entries of this record's defects."""
    allowed = {a["id"]: a["text"] for _, _, e in relevant(record, kb) for a in e["actions"]}
    ids = []
    for a in obj.get("recommended_actions") or []:
        for i in re.findall(r"[A-Z]-[A-Z]\d+", str(a)):
            if i in allowed and i not in ids:
                ids.append(i)
    summary, cause = str(obj.get("summary", "")).strip(), str(obj.get("probable_cause", "")).strip()
    if not ids or not summary:
        return None
    return {"summary": summary[:300], "probable_cause": cause[:300],
            "actions": [{"id": i, "text": allowed[i]} for i in ids[:3]],
            "log_note": " ".join(str(obj.get("log_note", "")).split()[:20])}


def fallback(record: dict, kb: dict, product_label: str) -> dict:
    """Templated advice straight from the KB (instant, always valid)."""
    items = relevant(record, kb)
    if not items:
        return {"summary": f"No defect found on the {product_label}.", "probable_cause": "",
                "actions": [], "log_note": "PASS"}
    d, t, e = items[0]
    names = ", ".join(f"{x['type'].replace('_', ' ')} ({x['severity']})" for x, _, _ in items)
    actions = [a for _, _, ee in items for a in ee["actions"][:1]] + [a for a in e["actions"][1:2]]
    return {"summary": f"{product_label}: {names} found; disposition {record['disposition']}.",
            "probable_cause": f"Typical cause of {t.replace('_', ' ')}: {e['causes'][0]}.",
            "actions": actions[:3],
            "log_note": f"{record['disposition']}: {names}"[:120]}
