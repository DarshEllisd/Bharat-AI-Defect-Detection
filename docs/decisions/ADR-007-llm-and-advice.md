# ADR-007: Llama 3.2 1B Instruct (SpinQuant INT4) via ExecuTorch; knowledge-base-grounded JSON advice

- **Status:** Accepted
- **Date:** 2026-11-01
- **Week:** 0, validated in week 1

## Context
Llama and ExecuTorch are **mandatory** for the advisory reasoning. The brief also says: give Llama structured findings, not pixels. The phone has 8 GB of RAM, shared with Chrome and the vision models.

## Options considered
| Option | Pros | Cons |
|---|---|---|
| **Llama 3.2 1B Instruct, 4-bit** | Fits easily; fast | Weaker reasoning; JSON not always valid |
| Llama 3.2 3B Instruct, 4-bit (~2.5-3 GB) | Better quality | Slower; tighter memory; exporting it ourselves needs ~24 GB RAM |
| Model source: pre-exported `.pte` (executorch-community, SpinQuant INT4) | Ready now | Built with an older ExecuTorch, so compatibility had to be checked |
| Model source: export ourselves (8da4w via `export_llm`) | Matches our runtime version exactly | Needs a gated Meta checkpoint and ~10-12 GB RAM |

Language: English only / English + Hindi / multiple Indian languages.

## Decision
- **Llama 3.2 1B Instruct, SpinQuant INT4**, from the pre-exported executorch-community `.pte`, run by ExecuTorch's `llama_main` (XNNPACK) on the **CPU with 4 threads**.
- **English only.**
- Advice is **grounded**: Llama gets the JSON defect record plus the category's knowledge-base entries, and must return JSON (`summary`, `probable_cause`, `recommended_actions` taken from the knowledge base, `log_note`). Its output is validated, and a **templated fallback** is used if it's invalid.
- 3B stays an optional upgrade, benchmarked only if needed.

## Why
Measured in week 1 (354-token defect prompt, 107 tokens generated):

| Threads | TTFT | Prefill | Decode | Peak RSS |
|---|---|---|---|---|
| 2 | 5.65 s | 63 tok/s | 12.4 tok/s | 1377 MB |
| **4** | **3.13 s** | 113 tok/s | **13.3 tok/s** | 1376 MB |
| 8 | 2.78 s | 127 tok/s | 12.5 tok/s | 1377 MB |

- Decode ≥ 10 tok/s ✅; RSS ≤ 2 GB ✅; TTFT about 3 s (borderline). Four threads is the sweet spot and leaves cores for everything else.
- The pre-exported `.pte` loads in the v1.5.0 runtime (sha256 verified), so no self-export was needed.
- **Grounding works:** the actions and cause were copied verbatim from the knowledge-base notes, and severity was left unchanged.

## Trade-offs and consequences
- **Imperfect JSON:** Llama added a "Here is the response:" prefix and **left off the closing `}`**. The week-5 parser must extract and repair JSON, with a template fallback.
- **Model load takes ~3 s**, so the advisory worker must stay running.
- **TTFT is dominated by prefill.** Shorten the static system prompt or cache its prefill.
- English only keeps 1B output reliable. Hindi would need templated translations.

## Evidence / references
- `spike/results/W1_RESULTS.md` → advisory pipeline
- `spike/phone/prompts/advisory_prompt.txt`
- `spike/results/phone_live/llama/`
