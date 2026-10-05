# Phase 1: untrained baselines (2026-10-04)

NanoBEIR (13 tasks, nDCG@10). Encoder: `src/olmo3_embed/model.py` (`OlmoEmbedder`), bf16, max 512 tokens.
OLMo-2-1B ran locally (RTX 5070 Ti Laptop); OLMo-3-7B ran on Colab L4 (`notebooks/phase1_olmo3_7b_baseline.ipynb`).

| Task | bge-base (trained) | Qwen3-0.6B (trained) | OLMo-2-1B mean | **OLMo-3-7B mean** | OLMo-2-1B EOS-last |
|---|---:|---:|---:|---:|---:|
| ArguAna | 64.02 | 62.69 | 30.98 | **41.31** | 0.00 |
| ClimateFever | 40.01 | 48.78 | 3.85 | 8.08 | 0.59 |
| DBPedia | 60.00 | 62.07 | 10.22 | 8.45 | 0.17 |
| FEVER | 94.15 | 92.47 | 10.70 | 15.38 | 0.00 |
| FiQA2018 | 46.05 | 65.36 | 3.97 | 3.58 | 0.00 |
| HotpotQA | 86.79 | 80.67 | 11.68 | 11.17 | 0.00 |
| MSMARCO | 63.61 | 65.97 | 4.06 | 1.38 | 0.00 |
| NFCorpus | 39.28 | 35.41 | 4.79 | 3.48 | 1.07 |
| NQ | 66.72 | 64.75 | 0.00 | 1.65 | 0.00 |
| Quora | 95.67 | 94.91 | 70.21 | 71.32 | 0.00 |
| SCIDOCS | 39.08 | 47.22 | 0.56 | 5.83 | 0.20 |
| SciFact | 81.72 | 71.41 | 27.00 | 37.13 | 0.00 |
| Touche2020 | 51.77 | 61.88 | 21.10 | 16.75 | 0.27 |
| **Average** | **63.76** | **65.66** | **15.32** | **17.35** | **0.18** |

## Reading it
- **Untrained LLMs are not retrievers.** The 7B gains only +2 over the 1B. The gains are on the symmetric,
  lexical-overlap tasks (ArguAna +10, SciFact +10, Quora), and there's nothing on asymmetric QA (NQ, MSMARCO, FiQA ≈ 0–4).
  Everything from 17 → 60+ has to come from contrastive training. That's expected; LLM2Vec reports the same shape.
- **EOS last-token pooling is random untrained** (0.18): the EOS hidden state is dominated by a few massive activations,
  so all texts embed nearly identically. It only becomes useful after training (Qwen3, NV-Embed use it *after* training).
  Phase 2 keeps it as an ablation arm.
- **The proxy tracks the target.** OLMo-2-1B and OLMo-3-7B have the same per-task profile, which supports doing the recipe
  ablations on the 1B locally (Phase 2) before spending Colab units on the 7B.

## Cost / ops notes
- Colab L4: ~1 hour total for the 7B NanoBEIR run (after one OOM fix and one empty-doc fix), ≈ 2 CU.
- Fixes found by this run: length-sorted batching with automatic OOM halving; `expandable_segments`; empty documents
  (NanoFiQA2018 has 27) encoded as `[EOS]`; the notebook pulls on re-run; mteb's `only-missing` lets a crashed run resume.
