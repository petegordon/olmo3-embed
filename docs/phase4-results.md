# Phase 4 run #1 results: OLMo-3-7B on Track B v1 (2026-10-07)

**Bottom line:**
- **The 7B lifts NanoBEIR to 65.93 avg-13** (+3.65 over the 1B proxy on the same data and recipe). That now beats
  bge-base (63.76) and edges past Qwen3-Embedding-0.6B (65.66).
- **Gate check A fails:** Qwen3-Embedding-4B scores **69.97** on our harness.
- **The gap is almost entirely held-out generalization:**
  - In-domain we're level with Qwen3-4B (75.65 vs 76.20).
  - On the five held-out tasks we're 9.7 points behind (50.37 vs 60.02).
  - The 7B barely improved on held-out over the 1B (+1.02).
- More of the same 8 training tasks isn't the main lever; **task diversity** is.

**Run:** `allenai/Olmo-3-1025-7B` with `configs/7b-run1.yaml`.
- Exactly Phase 3 B3's recipe and data budget: 64k Track B v1 rows, 496 steps at batch 128, EOS pooling, no
  instructions, LoRA r=16 (39.98M trainable parameters).
- Colab A100 80GB, no gradient checkpointing, chunk 8, 43.9 s/step average, 6.1 h of training.
- `notebooks/phase4_7b_colab.ipynb`. Results are on `colab-results` (`8862c07`, `671c70b`).

| Task | 1B proxy B3 | **7B run #1** | Δ 7B vs 1B | bge-base | Qwen3-0.6B | **Qwen3-4B** | Δ 7B vs Qwen3-4B |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArguAna *(held out)* | 53.93 | 57.28 | +3.35 | 64.02 | 62.69 | 61.54 | −4.26 |
| ClimateFever *(held out)* | 38.70 | 40.16 | +1.46 | 40.01 | 48.78 | 50.51 | −10.35 |
| DBPedia *(held out)* | 60.41 | 60.14 | −0.27 | 60.00 | 62.07 | 69.90 | −9.76 |
| SCIDOCS *(held out)* | 39.59 | 44.34 | +4.75 | 39.08 | 47.22 | 52.80 | −8.46 |
| Touche2020 *(held out)* | 54.12 | 49.92 | −4.20 | 51.77 | 61.88 | 65.34 | −15.42 |
| FEVER | 90.59 | 94.47 | +3.88 | 94.15 | 92.47 | 94.26 | +0.21 |
| FiQA2018 | 57.12 | 66.35 | +9.23 | 46.05 | 65.36 | 74.72 | −8.37 |
| HotpotQA | 84.69 | 89.14 | +4.45 | 86.79 | 80.67 | 87.07 | +2.07 |
| MSMARCO | 55.13 | 61.88 | +6.75 | 63.61 | 65.97 | 67.97 | −6.09 |
| NFCorpus | 35.91 | 40.84 | +4.93 | 39.28 | 35.41 | 38.20 | +2.64 |
| NQ | 65.09 | 74.73 | +9.64 | 66.72 | 64.75 | 70.36 | +4.37 |
| Quora | 95.66 | 97.39 | +1.73 | 95.67 | 94.91 | 95.61 | +1.78 |
| SciFact | 78.70 | 80.40 | +1.70 | 81.72 | 71.41 | 81.40 | −1.00 |
| **avg-13** | **62.28** | **65.93** | **+3.65** | **63.76** | **65.66** | **69.97** | **−4.04** |
| **avg-in-8** | **70.36** | **75.65** | **+5.29** | **71.75** | **71.37** | **76.20** | **−0.55** |
| **avg-OOD-5** (held out) | **49.35** | **50.37** | **+1.02** | **50.98** | **56.53** | **60.02** | **−9.65** |

Qwen3-0.6B's Touche (61.88) comes from `docs/phase1-baselines.md`.

## Reading it
- **Scale helps where there's training signal.** In-domain, the 7B gains +5.3 over the 1B: NQ +9.6, FiQA +9.2, MS MARCO
  +6.8. It's now level with Qwen3-4B in-domain, and ahead on NQ, HotpotQA, NFCorpus and Quora.
- **Scale barely moves the held-out tasks (+1.0).** Gains on SCIDOCS (+4.8) and ArguAna (+3.4) are offset by Touche
  (−4.2) and a flat DBPedia.
  - Qwen3-4B's held-out lead (+9.7) is the whole story of the gap: argument retrieval (Touche −15.4, ArguAna −4.3),
    climate claims (−10.4), entity search (DBPedia −9.8) and scientific citation (SCIDOCS −8.5).
  - Track B has no training data shaped like any of those. Qwen3 trained on ~150M pairs, including large synthetic
    sets covering many task types.
- **Within noise:** single-task gaps of 2–3 points are noise at 50 queries per task (Phase 3). The held-out average's
  −9.7 vs Qwen3-4B, and its +1.0 over the 1B, aren't.
- **Projection to full BEIR:** on our harness, small models have scored about 10 points lower on full BEIR than on
  NanoBEIR. That would put run #1 around **56** on full BEIR, right at T1 (56.6). Only the full eval can say whether it
  clears it. It also calibrates the Nano-to-BEIR mapping for our own model.

## Gate status
- **A, NanoBEIR ≥ Qwen3-Embedding-4B: ✗** 65.93 vs 69.97.
- **B, full BEIR ≥ 56.6 (T1): not yet measured.** The projection says it's borderline.

## What this says to do next (options for Pete)
1. **Full 15-task BEIR eval of run #1** (~4 h on an A100 ≈ 25 units, or ~10 h on an L4 ≈ 17 units). It settles T1 and
   calibrates Nano vs BEIR. Cheap, and worth knowing before spending more.
2. **Task diversity: Track A / synthetic data aimed at the held-out task types** (PLAN §4.4, `docs/phase3-plan.md` §7).
   - Generate queries for argument / counter-argument, climate-claim, entity-search and citation-style retrieval from
     non-BEIR corpora (Wikipedia, S2ORC, args.me-style debate text), with the four generators already wired up:
     Claude, Codex, Gemini, and OLMo-3.1-32B on Colab.
   - Also, cheaply: public non-BEIR pairs (StackExchange, S2ORC citation pairs, ELI5, NLI) from `cfli/bge-full-data` /
     `bge-e5data`, decontaminated like Track B.
   - The 1B proxy measures each addition by **avg-OOD-5** before it touches the 7B.
3. **Full Track B epoch on the 7B** (~240 units). This mainly lifts in-domain tasks, which are already level with
   Qwen3-4B, so it's the weakest lever for the gap. Hold it until the data mix is broader.

**Recommendation:** do 1 now (cheap, needed for T1 either way), and start 2 in parallel on the laptop and subscriptions
(the synthetic-data pilot), proving it on the proxy first. Then a 7B run #2 on Track B + the new data.

## Cost
- **Run #1:** about 6.1 h training, plus ~10 min of timing trials and ~1.3 h of evals (7B and Qwen3-4B) on an A100 80GB
  at 6.77 units/h. **≈ 52 units.**
- **Units left:** roughly 255 (check the Resources panel).
