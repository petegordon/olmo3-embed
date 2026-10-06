# Phase 3 results: Track B v1 on the 1B proxy (2026-10-06)

**Bottom line:**
- **The gate passes.** With 4× the data (B3: 64k rows, 496 steps), Track B v1 lifts the proxy to **62.28 NanoBEIR
  avg-13** (+4.80 over Phase 2's best, a8) and **49.35 on the held-out tasks** (+3.33).
- **Distance to small reference models:** about 1.5 points behind bge-base on both averages.
- **What carries forward:**
  - The Qwen3 rescore + guard is worth +1.65.
  - Per-task instructions still *hurt* (−4.6).
  - At equal compute (16k rows) the mix only matches MS MARCO-only; its value shows up with more data.

Setup:
- **Model and recipe:** OLMo-2-0425-1B, LoRA r=16, Phase 2's a8 recipe (causal attention, EOS last-token pooling, no
  instructions, τ=0.02, batch 128).
- **Data:** Track B v1, decontaminated and rescored (`docs/data-card.md`).
- **Batching:** one task per batch with T=0.5 sampling, k=3 negatives from the 10 hardest, in-batch false-negative
  masking (`configs/trackb-base.yaml`).
- **Hardware:** Colab A100 40GB (`notebooks/phase3_bruns_colab.ipynb`).
- **Eval:** NanoBEIR nDCG@10, max 512 tokens. Raw results are on the `colab-results` branch.

| Task | a8 (Phase 2, MS MARCO 16k) | **B1** mix 16k | B2 +instr | B4 upstream negs | **B3** mix 64k | bge-base | Qwen3-0.6B |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArguAna *(held out)* | 56.64 | 51.31 | 45.02 | 50.69 | 53.93 | 64.02 | 62.69 |
| ClimateFever *(held out)* | 31.91 | 34.79 | 30.45 | 35.70 | 38.70 | 40.01 | 48.78 |
| DBPedia *(held out)* | 52.23 | 56.60 | 52.05 | 53.14 | 60.41 | 60.00 | 62.07 |
| FEVER | 85.10 | 83.51 | 82.21 | 80.82 | 90.59 | 94.15 | 92.47 |
| FiQA2018 | 51.90 | 46.52 | 43.07 | 46.08 | 57.12 | 46.05 | 65.36 |
| HotpotQA | 75.73 | 79.20 | 67.59 | 75.70 | 84.69 | 86.79 | 80.67 |
| MSMARCO | 57.10 | 46.88 | 41.42 | 49.84 | 55.13 | 63.61 | 65.97 |
| NFCorpus | 32.33 | 34.86 | 31.17 | 32.32 | 35.91 | 39.28 | 35.41 |
| NQ | 55.92 | 61.02 | 53.23 | 57.09 | 65.09 | 66.72 | 64.75 |
| Quora | 90.35 | 93.81 | 89.99 | 93.49 | 95.66 | 95.67 | 94.91 |
| SCIDOCS *(held out)* | 36.02 | 38.54 | 30.48 | 36.27 | 39.59 | 39.08 | 47.22 |
| SciFact | 68.71 | 71.44 | 75.03 | 67.93 | 78.70 | 81.72 | 71.41 |
| Touche2020 *(held out)* | 53.28 | 52.61 | 49.94 | 50.65 | 54.12 | 51.77 | 61.88¹ |
| **avg-13** | **57.48** | **57.78** | **53.20** | **56.13** | **62.28** | **63.76** | **65.66** |
| **avg-in-8** | **64.64** | **64.66** | **60.46** | **62.91** | **70.36** | **71.75** | **71.37** |
| **avg-OOD-5** (held out) | **46.02** | **46.77** | **41.59** | **45.29** | **49.35** | **50.98** | **56.53** |

¹ From `docs/phase1-baselines.md`; that file is missing from the `colab-results` archive.

## Reading it
- **B1, mix vs MS MARCO-only at equal compute (+0.30 avg-13, +0.75 held-out): a wash.**
  - The mix spreads the same 16k rows over 8 tasks, so MS MARCO gets only about 27% of them. MS MARCO drops 10.2
    while most other tasks gain (NQ +5.1, DBPedia +4.4, HotpotQA +3.5).
  - At this budget, diversity just moves gains around.
- **B3, 4× the data (+4.50 avg-13 and +2.58 held-out over B1): the real win.**
  - Best of all runs on 11 of 13 tasks. The exceptions are ArguAna (see below) and MS MARCO, where it recovers to 55.1
    but stays just under a8's 57.1, which trained on 16k MS MARCO rows (B3 had roughly 17k).
  - The training loss on Quora reached 0.07 at 96% batch accuracy, so the easy tasks are close to saturated.
  - B3 used only 64k of the 374k rows (17%), so the scaling curve hasn't flattened yet.
- **B2, per-task MTEB instructions (−4.58 avg-13, −5.18 held-out vs B1).**
  - Phase 2 blamed its instruction loss on a train/eval mismatch. Here train and eval use the same MTEB prompts and
    instructions still lose on 12 of 13 tasks (SciFact +3.6 is the exception).
  - At this scale (a 1B model with LoRA, 125 steps) the instruction prefix costs more than it adds. The 7B models that
    benefit from instructions (Qwen3, E5-Mistral) train on millions of pairs. **Keep no instructions for now; re-test
    once at 7B.**
- **B4, upstream negatives instead of rescored (B1 wins by +1.65 avg-13, +1.48 held-out, 11 of 13 tasks).** The 3.5 h
  Qwen3-Embedding rescore and the 95% guard (~24 units) paid off.
- **Noise:** SciFact and NFCorpus use the *same* mined negatives in B1 and B4, yet differ by 3.5 and 2.5 points. So
  single-task gaps of 2–3 points are noise at 50 queries per task. Only the averages (13, 8 and 5 tasks) are reliable.
- **ArguAna is the one regression vs a8** (−2.7 in B3, −5.3 in B1).
  - Counter-argument retrieval has no training data in Track B, since ArguAna was excluded for decontamination.
  - The MS MARCO-only model happened to transfer better.
  - Expected to improve with Track A synthetic counter-argument data.

## Gate to Phase 4: ✅ passed
- Data card written; zero BEIR test-query overlap verified (`docs/data-card.md`).
- The best Track B run beats a8 on avg-13: B3 **62.28** vs 57.48.
- No regression on held-out avg-OOD-5: B3 **49.35** vs 46.02.

## Carried into Phase 4 (7B)
- **Recipe:** causal attention, EOS last-token pooling, **no instructions** (one instruction ablation at 7B), τ=0.02,
  k=3 from the 10 hardest rescored negatives, one task per batch at T=0.5, false-negative masking.
- **Data:** all of Track B v1 (373,762 rows, rescored). B3 says more data still pays, and the 7B has far more capacity.
  One epoch at batch 128 is about 2,900 steps. Size the first 7B run from a measured s/step (Phase 4 timing check),
  against the ~330 units left.
- **Optional cheap check before the 7B:** the proxy on the full 374k rows (~2,900 steps, about 10–11 h on an A100,
  ~55 units) would show where the 1B curve flattens. It's probably not worth the units; the 7B run answers the more
  important question.

## Cost and ops
- **B-runs:** about 3.4 h on an A100 40GB at 5.3 units/h ≈ 18 units.
  - 13.4 s/step on Track B vs 9.8 s on MS MARCO, because the passages are longer.
  - B3 ran faster than projected (9.3 s/step late in the run).
- **Phase 3 so far:** rescore ≈ 24, judge pilot ≈ 5, B-runs ≈ 18, total **≈ 47 units**.
- **Bug found:** `scripts/push_results.sh` used `git add -f results`, which also stages deletions. Every push from a
  runtime that didn't have the full `results/` folder dropped earlier results from the `colab-results` tip. Two pushes
  did this (Phase 2 round 2 and the rescore push).
  - Fixed with `--ignore-removal`.
  - The 110 dropped files were restored as a union of all earlier pushes (`colab-results` commit `83bcfe3`; no
    history rewritten).
