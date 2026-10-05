# Phase 2: contrastive recipe ablations on the proxy (OLMo-2-1B, local)

**Status: DRAFT, awaiting Pete's review. Only the smoke test below has been run; no ablation arm has been trained.**

## Goal
Pick the 1–2 best configurations to scale to OLMo-3-7B on Colab (Phase 4) by training the 1B proxy cheaply on the
laptop and scoring each arm on NanoBEIR. The Phase 1 baselines show the 1B and 7B share a per-task profile, so
rankings should transfer, though absolute numbers won't.

## Setup (same for every arm)
| | |
|---|---|
| Model | `allenai/OLMo-2-0425-1B`, bf16, LoRA r=16/α=32 on all linear layers (~0.5% of params trainable) |
| Data | MS MARCO, 16k queries (fixed seeded slice of `data/msmarco_cocondenser_50k.parquet`), 1 positive + k hard negatives sampled from coCondenser ranks 10–50 |
| Objective | InfoNCE over all in-batch docs (positives + hard negatives of every query), τ = 0.02 |
| Batch | 128 queries per step via GradCache, giving 128 × (1+k) documents scored per query; 125 steps, lr 1e-4, 5% warmup, linear decay |
| Lengths | query ≤ 64 tokens, doc ≤ 256 |
| Eval | NanoBEIR (13 tasks), max 512 tokens; report **avg-13** and **avg-12 (excluding NanoMSMARCO, which is in-domain)** |

## Arms
| Arm | Change vs A0 | Question it answers |
|---|---|---|
| **A0 base** | mean pooling, causal, k=3, instructions | reference |
| A1 last | pooling = last (EOS) | Does EOS pooling (random untrained) overtake mean once trained? (Qwen3/E5 use it) |
| A2 bidir | bidirectional attention, mean | NV-Embed-style mask removal without MNTP: does it help, or hurt without adaptation? |
| A3 bidir+last | bidirectional + EOS | interaction check |
| A4 neg1 | k = 1 | value of hard negatives (cheaper steps) |
| A5 neg7 | k = 7 | more hard negatives (≈2× compute per step) |
| A6 noinstr | no query instructions | do instructions help on out-of-domain tasks? |
| A7 τ0.05 | temperature 0.05 | sensitivity to τ |

Run: `scripts/phase2_ablations.sh` (all arms) or `scripts/phase2_ablations.sh a0-base a2-bidir` (some). Each arm
trains, then evaluates. Both steps resume after an interruption.

## Decision rule
- Primary metric: **NanoBEIR avg-12**. A difference under ~1 point is treated as noise; one seed, 50 queries per
  task.
- Carry forward the best pooling × attention combination plus the best k. If two arms are within noise, prefer the
  cheaper one.
- If A2/A3 (bidirectional) lose clearly, add an **MNTP warm-up arm** (LLM2Vec-style masked next-token training before
  contrastive). That needs a small extra script and isn't in this draft.

## Expected cost (measured, RTX 5070 Ti Laptop)
- Training: **~46 s/step** at batch 128, k=3 (GradCache, gradient checkpointing, chunk 32, peak 4.8 GB VRAM).
  The run is compute-bound: chunk 64 is no faster, chunk 128 OOMs, and turning checkpointing off OOMs even at chunk 16.
- Per arm: 125 steps ≈ **96 min train + ~15 min NanoBEIR ≈ 1 h 50 min**. A5 (k=7) costs ≈1.8× per step (~3 h).
- **All 8 arms ≈ 17 h** of laptop GPU time. All local; no Colab units.

Options for Pete:
1. **All 8 arms (~17 h)**, e.g. two overnight sessions.
2. **Priority 5 (~9 h, one overnight):** A0 base, A1 last, A2 bidir, A4 neg1, A6 noinstr. Run A3/A5/A7 only if
   the first five leave the question open.
3. **Half data (8k examples, 63 steps, ~1 h/arm, all 8 ≈ 9 h):** faster but noisier rankings.

Recommendation: **option 2**. Pooling, attention and instructions are the decisions that matter most for the 7B
recipe. Negatives count and τ are cheap to re-tune later.

## Not in Phase 2 (deliberately)
Synthetic data (Phase 3), multi-dataset mixing, cross-encoder false-negative filtering, Matryoshka, depth truncation,
checkpoint merging, the 7B. Each would confound the arm comparisons.

## Smoke test (2026-10-04): PASS
- `tests/test_gradcache.py`: GradCache LoRA gradients match full-batch backprop (relative error < 1e-4) for all four
  pooling × attention combos. A mutated version (skipping the doc backward pass) gives 92% error and is caught.
- A0 trained 4 steps, saved a checkpoint at step 2, and resumed to step 6. Step 3 after the resume reproduced the
  uninterrupted loss (6.375) and grad norm (12.222) exactly. Loss went 9.77 → 5.31 in 6 steps.
- `eval --run runs/p2-smoke` loads the adapter and scores NanoSciFact (24.1 after 6 steps; meaningless, it's a
  plumbing check).
