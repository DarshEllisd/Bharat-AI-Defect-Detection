"""Summarise llama_main logs into CSV: load time, TTFT, prefill/decode tok/s, peak RSS.

The ExecuTorch LLM runner prints a line containing "PyTorchObserver {json}" with
millisecond timestamps. If that line is missing, the raw log is still kept for
manual reading and the row shows blanks.
"""
import csv
import json
import re
import sys
from pathlib import Path

FIELDS = ["log", "exit_code", "prompt_tokens", "generated_tokens", "load_s", "ttft_s",
          "prefill_tok_s", "decode_tok_s", "peak_rss_mb"]


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {k: "" for k in FIELDS}
    row["log"] = path.name

    m = re.search(r"# exit_code=(\d+)", text)
    if m:
        row["exit_code"] = m.group(1)
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if m:
        row["peak_rss_mb"] = f"{int(m.group(1)) / 1024:.0f}"

    m = re.search(r"PyTorchObserver\s*(\{.*\})", text)
    if not m:
        return row
    try:
        s = json.loads(m.group(1))
    except json.JSONDecodeError:
        return row

    unit = float(s.get("SCALING_FACTOR_UNITS_PER_SECOND", 1000))
    prompt_toks = int(s.get("prompt_tokens", 0))
    gen_toks = int(s.get("generated_tokens", 0))
    row["prompt_tokens"], row["generated_tokens"] = prompt_toks, gen_toks

    def secs(a: str, b: str):
        if a in s and b in s:
            return (float(s[b]) - float(s[a])) / unit
        return None

    load = secs("model_load_start_ms", "model_load_end_ms")
    ttft = secs("inference_start_ms", "first_token_ms")
    prefill = secs("inference_start_ms", "prompt_eval_end_ms")
    decode = secs("first_token_ms", "inference_end_ms")
    if load is not None:
        row["load_s"] = f"{load:.2f}"
    if ttft is not None:
        row["ttft_s"] = f"{ttft:.2f}"
    if prefill:
        row["prefill_tok_s"] = f"{prompt_toks / prefill:.1f}"
    if decode and gen_toks > 1:
        row["decode_tok_s"] = f"{(gen_toks - 1) / decode:.1f}"
    return row


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for log in sorted(out_dir.glob("*.log")):
        writer.writerow(parse(log))


if __name__ == "__main__":
    main()
