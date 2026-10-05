# Phase 2 results: recipe ablations on the 1B proxy (2026-10-05)

Round 1: the 5 priority arms from `docs/phase2-plan.md`. Each arm is `configs/proxy-base.yaml` with a single override:
OLMo-2-0425-1B, LoRA r=16, 16k MS MARCO queries, 125 steps at batch 128, τ=0.02. Trained and evaluated on a Colab
A100 40GB (`notebooks/phase2_ablations_colab.ipynb`). Scores are NanoBEIR nDCG@10 (×100), max 512 tokens.
Raw results and run logs are on the `colab-results` branch (`results/results/olmo3-embed__p2-*`,
`results/run_logs/p2-*`).

| Task | 1B untrained (mean) | **a0 base** (mean, causal, k=3, instr) | a1 last | a2 bidir | a4 neg1 | a6 noinstr |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 30.98 | 44.12 | **54.71** | 31.38 | 44.17 | 50.63 |
| ClimateFever | 3.85 | 25.92 | 28.99 | 26.15 | 25.39 | **37.00** |
| DBPedia | 10.22 | 38.72 | 51.50 | **52.00** | 39.51 | 45.91 |
| FEVER | 10.70 | 76.05 | 77.35 | 65.96 | 73.21 | **79.20** |
| FiQA2018 | 3.97 | 18.09 | **46.25** | 35.09 | 20.45 | 41.88 |
| HotpotQA | 11.68 | 57.02 | **66.53** | 57.57 | 54.66 | 63.24 |
| MSMARCO *(in-domain)* | 4.06 | 37.34 | 47.19 | **56.99** | 36.80 | 51.07 |
| NFCorpus | 4.79 | 16.46 | 30.13 | 26.73 | 17.37 | **31.60** |
| NQ | 0.00 | 40.48 | **56.73** | 53.59 | 38.62 | 49.93 |
| Quora | 70.21 | 67.31 | **93.36** | 84.71 | 66.89 | 79.27 |
| SCIDOCS | 0.56 | 14.88 | **33.41** | 28.54 | 16.47 | 28.73 |
| SciFact | 27.00 | 59.45 | **69.96** | 62.07 | 60.69 | 66.40 |
| Touche2020 | 21.10 | 44.47 | 48.19 | 46.48 | 46.66 | **48.94** |
| **avg-13** | 15.32 | 41.56 | **54.18** | 48.25 | 41.61 | 51.83 |
| **avg-12 (no MSMARCO, primary)** | 16.25 | 41.91 | **54.76** | 47.52 | 42.01 | 51.89 |
| Δ avg-12 vs a0 | | – | **+12.85** | +5.61 | +0.10 | +9.98 |
| A100 s/step | | 9.5 | 9.5 | 9.4 | **5.3** | 9.5 |

References on the same harness (Phase 1): bge-base 63.76 / 63.77 (avg-13 / avg-12), Qwen3-Embedding-0.6B 65.66 / 65.98.

## Reading it
- **The gate passes.** Training lifts the proxy from 16.25 to 41.9–54.8 avg-12 in 125 steps on MS MARCO alone.
- **Last-token (EOS) pooling wins by a wide margin (+12.9),** and it's the best arm on 7 of the 12 out-of-domain tasks. The biggest
  gains are on tasks where mean pooling is weak: FiQA +28, Quora +26, SCIDOCS +19. Untrained EOS was random (0.18,
  Phase 1), so this is entirely learned. Mean pooling even *lowered* Quora below untrained (70.2 → 67.3).
- **No instructions beat instructions (+10.0).** This is probably a train/eval mismatch rather than evidence against
  instructions: training uses one fixed instruction (`train.py` default, MS MARCO web search), while eval feeds
  MTEB's per-task instructions, which the model never saw. Without instructions, train and eval inputs match. Revisit
  once Phase 3 trains on many tasks with varied instructions, which is how Qwen3/E5 use them.
- **Bidirectional attention helps without MNTP (+5.6)** but is uneven. It is best on in-domain MSMARCO (57.0) and on
  DBPedia, but loses on ArguAna (−12.7) and FEVER (−10.1). So it fits MS MARCO better and generalizes worse, and so
  far it isn't better than a causal model with EOS pooling. The planned MNTP arm is not needed yet; a3 (bidir + last)
  tells us whether to pursue it.
- **1 hard negative ≈ 3 hard negatives (+0.1, within noise) at 56% of the step cost.** At this data scale the extra
  negatives buy nothing. Keep k small for proxy work; revisit at 7B and with false-negative-filtered mining (Phase 3),
  where hard negatives usually matter more.
- **Caveats:** one seed, 50 queries per task, 16k MS MARCO examples. Each arm changed one thing from a0, so the
  effects may not add up. Round 2 tests that.

## Round 2 (queued)
Added to the notebook (`RUN_ARMS`):
- **a8-last-noinstr** (`pooling=last instructions=false`): do the two biggest wins stack?
- **a3-bidir-last** (`bidirectional=true pooling=last`): does bidirectional attention help on top of EOS pooling?

**Provisional 7B recipe** (until round 2 lands): causal attention, EOS last-token pooling, k=1–3 hard negatives,
and query instructions only once training covers many tasks with varied instructions.

## Cost / ops notes
- A100 40GB: ~25 min per k=3 arm (16 min for k=1), with eval taking ~5 min of that; 5 arms ≈ 1 h 50 min ≈ 10 CU.
  That's ~4× faster than the laptop estimate (1 h 50 min per arm).
- Fixes found by this run: the `data/` gitignore pattern was swallowing `src/olmo3_embed/data/`; IPython skips `{VAR}`
  expansion on a `!` line that also has `$var`; Colab's preinstalled torchao 0.10 blocks importing peft ≥ 0.21; a split
  `print(f'…` line in the notebook; `compare_published.py` crashed on `results/run_logs/*.json`.
