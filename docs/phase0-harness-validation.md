# Phase 0: harness validation (2026-10-04)

Stack: Windows 11, RTX 5070 Ti Laptop (12 GB, sm_120), torch 2.11+cu128, transformers 5.18, sentence-transformers 6.1,
**mteb 2.22.3**. Run with `uv run python -m olmo3_embed.eval ...`. Compare with `uv run python scripts/compare_published.py`.

## Encoder path: bge-base-en-v1.5, PASS

| Task | Ours | Published | Δ |
|---|---:|---:|---:|
| ArguAna | 63.75 | 63.75 | 0.00 |
| FiQA2018 | 40.62 | 40.65 | −0.02 |
| NFCorpus | 37.35 | 37.37 | −0.01 |
| SCIDOCS | 21.73 | 21.73 | 0.00 |
| SciFact | 74.04 | 74.34 | −0.31 |

NanoBEIR (13 tasks) average: **63.76**. This is our inner-loop reference point; no published counterpart exists.

## Decoder / last-token path: Qwen3-Embedding-0.6B, PASS with one known discrepancy

| Task | Ours | Published | Δ |
|---|---:|---:|---:|
| NFCorpus | 36.30 | 36.71 | −0.41 |
| SciFact | 69.93 | 69.72 | +0.21 |
| ArguAna | 69.31 | 70.97 | **−1.65** |
| FiQA2018 | 47.26 | 46.61 | +0.65 |

The deltas go both ways (−1.65 to +0.65), while bge's are all ≈0. That fits differences in Qwen's published run
setup, not a bias in our harness.

NanoBEIR (13 tasks) average: **65.66**, vs bge-base at 63.76. NanoTouche2020 was run with `--max-seq-length 8192
--batch-size 4` (see the sysmem note below). The ordering matches full BEIR (Qwen3-0.6B 55.5 avg on the leaderboard).

What we ruled out for the ArguAna gap:
- **Instruction text:** mteb's ArguAna prompt equals Qwen's ("Given a claim, find documents that refute the claim").
- **Precision:** transformers 5 loads bf16 by default (from the model config). bf16 = 69.311 and forced fp32 = 69.337.
- **mteb ArguAna scoring changes:** bge's published ArguAna (mteb 1.18) matches ours (2.22) exactly.
- **EOS / pooling:** the tokenizer appends `<|endoftext|>` (151643); pooling is last-token, normalized.

The likely cause is the Qwen team's own run setup (published under mteb 1.38.9; it isn't recorded in `run_settings`).
It's out of scope for us. Treat the published Qwen3 numbers as **±~1.5 per task** when comparing.

## Lessons for OLMo
- **transformers 5 loads checkpoints in their saved dtype.** Always pass `--dtype` explicitly and record it.
- **Windows CUDA sysmem fallback (important).** Qwen3's ModelMeta allows 32k tokens per text. On NanoTouche2020 (long
  docs) at batch 16, VRAM overflowed and the NVIDIA driver *silently spilled into system RAM* instead of raising OOM:
  one python.exe reached 14 GB RAM and Windows commit hit 44.6/45.9 GB. Mitigations:
  1. Pass `--max-seq-length` (8192 is plenty for BEIR) and a smaller batch for long-doc tasks.
  2. Run `scripts/ram_guard.sh 8 &` during evals; it kills any python.exe above 8 GB working set.
  3. Optional: NVIDIA Control Panel → Manage 3D settings → *CUDA – Sysmem Fallback Policy* → **Prefer No Sysmem
     Fallback**. CUDA then raises a real OOM instead of eating RAM. **Recommended before any OLMo work.**
- **mteb writes non-default-kwarg runs under `.../experiments/exp_<hash>/`.** `compare_published.py` keys on the
  directory name, so pass `--ours` per run folder.
- **NanoBEIR results live under the `train` split.**
