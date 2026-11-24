"""Advisory worker (week 5): Llama 3.2 1B Instruct via ExecuTorch's llama_main on the Arm CPU (ADR-012),
in a background thread fed by a queue, so it never blocks an inspection (the vision pipeline answers at once;
advice follows).

Per non-PASS inspection:
  1. templated advice from the KB is stored immediately (status "fallback", instant, always valid);
  2. Llama generates; if its output parses and is grounded in the KB (prompt.py), it replaces the template
     (status "llm"); otherwise the template stays and the reason is stored (status "fallback").
"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .prompt import PREFILL, build_prompt, extract_json, extract_markdown, fallback, ground, load_kb

ASSISTANT_TAG = "<|start_header_id|>assistant<|end_header_id|>"


class AdviceWorker:
    def __init__(self, db, llama_bin: Path, model: Path, tokenizer: Path, threads: int = 4,
                 max_new_tokens: int = 140, timeout_s: float = 120.0) -> None:   # JSON answer ~100 tokens
        self.db, self.bin, self.model, self.tok = db, Path(llama_bin), Path(model), Path(tokenizer)
        self.threads, self.max_new, self.timeout = threads, max_new_tokens, timeout_s
        self.q: queue.Queue = queue.Queue()
        self.available = self.bin.exists() and self.model.exists() and self.tok.exists()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, record: dict, product_label: str, rules: str) -> dict:
        kb = load_kb(rules)
        adv = fallback(record, kb, product_label)
        status = "pending" if self.available and record["disposition"] != "PASS" else "fallback"
        self.db.set_advice(record["id"], status, "template", adv,
                           note="" if self.available else "llama_main or model missing on this device")
        if status == "pending":
            self.q.put((record, product_label, rules))
        return {"status": status, "source": "template", **adv}

    def _loop(self) -> None:
        while True:
            record, label, rules = self.q.get()
            if not self.q.empty():
                # a newer inspection is waiting: only the latest gets LLM advice; this one keeps its instant template
                # advice (live session: 50-68 s per answer, so the queue fell further and further behind)
                self.db.set_advice(record["id"], "fallback", "template", fallback(record, load_kb(rules), label),
                                   note="LLM skipped: a newer inspection was waiting")
                continue
            try:
                self._run(record, label, rules)
            except Exception as e:  # noqa: BLE001 - advice must never take the server down
                kb = load_kb(rules)
                self.db.set_advice(record["id"], "fallback", "template", fallback(record, kb, label), note=f"error: {e}")

    def generate(self, prompt: str) -> tuple[str, float]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(prompt)
            pfile = fh.name
        t0 = time.time()
        try:
            out = subprocess.run([str(self.bin), "--model_path", str(self.model), "--tokenizer_path", str(self.tok),
                                  "--prompt_file", pfile, "--temperature", "0", "--max_new_tokens", str(self.max_new),
                                  "--cpu_threads", str(self.threads)],
                                 capture_output=True, text=True, timeout=self.timeout, stdin=subprocess.DEVNULL,
                                 preexec_fn=lambda: os.nice(10))   # lower priority: inspections come first
        finally:
            Path(pfile).unlink(missing_ok=True)
        text = out.stdout
        if ASSISTANT_TAG in text:
            text = text.rsplit(ASSISTANT_TAG, 1)[1]
        return text, time.time() - t0

    def _run(self, record: dict, label: str, rules: str) -> None:
        kb = load_kb(rules)
        text, secs = self.generate(build_prompt(record, kb, label))
        # llama_main echoes the prompt incl. the pre-fill, so the text after the assistant tag usually already
        # starts with '{"summary": "'; only prepend it when the runner didn't echo it.
        obj = extract_json(text) or extract_json(PREFILL + text.lstrip()) or extract_markdown(text)
        adv = ground(obj, record, kb) if obj else None
        if adv:
            self.db.set_advice(record["id"], "llm", "llama-3.2-1b", adv, seconds=secs, raw=text[:4000])
        else:
            why = "no JSON in output" if obj is None else "no valid knowledge-base action ids"
            self.db.set_advice(record["id"], "fallback", "template", fallback(record, kb, label),
                               seconds=secs, raw=text[:4000], note=f"LLM output rejected: {why}")
