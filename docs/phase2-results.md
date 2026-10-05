# Phase 2 results: recipe ablations on the 1B proxy (2026-10-05)

**Bottom line:** EOS last-token pooling + no instructions (**a8: 57.51 NanoBEIR avg-12**, up from 16.25 untrained and
41.91 for the mean-pooling base) is the recipe to scale to the 7B. Bidirectional attention is dropped. See
[Phase 2 decision](#phase-2-decision).

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

## Round 2 (2026-10-05): do the winners stack?
Same setup and A100. a8 = `pooling=last instructions=false`; a3 = `bidirectional=true pooling=last`.

| Task | a1 last (round 1 best) | **a8 last + noinstr** | a3 bidir + last |
|---|---:|---:|---:|
| ArguAna | 54.71 | **56.64** | 28.71 |
| ClimateFever | 28.99 | **31.91** | 22.60 |
| DBPedia | 51.50 | **52.23** | 48.31 |
| FEVER | 77.35 | **85.10** | 55.99 |
| FiQA2018 | 46.25 | **51.90** | 35.25 |
| HotpotQA | 66.53 | **75.73** | 54.55 |
| MSMARCO *(in-domain)* | 47.19 | **57.10** | 51.83 |
| NFCorpus | 30.13 | **32.33** | 25.19 |
| NQ | **56.73** | 55.92 | 56.12 |
| Quora | **93.36** | 90.35 | 79.42 |
| SCIDOCS | 33.41 | **36.02** | 26.12 |
| SciFact | **69.96** | 68.71 | 56.72 |
| Touche2020 | 48.19 | **53.28** | 42.91 |
| **avg-13** | 54.18 | **57.48** | 44.90 |
| **avg-12 (primary)** | 54.76 | **57.51** | 44.32 |
| Δ avg-12 vs a1 | – | **+2.75** | −10.44 |

- **The two wins stack.** a8 is the new best at **57.51 avg-12** (+15.6 over a0). It beats a1 on 9 of 12 out-of-domain
  tasks and loses only slightly on Quora, SciFact and NQ. The gain is smaller than the sum of the two single effects
  (+12.9 and +10.0), so they partly overlap.
- **Bidirectional attention clearly hurts with EOS pooling (−10.4 vs a1).** It loses on all 12 out-of-domain tasks
  (only narrowly on NQ, −0.6). With mean pooling it helped (+5.6), which fits the picture: removing the causal mask without MNTP adaptation
  disrupts the final-token state, while mean pooling averages over it. Pursuing bidirectional attention would mean
  adding the MNTP warm-up and going back to mean pooling, and even then it would need to beat a8's 57.5. That isn't
  worth it at this stage, so drop it.

## Phase 2 decision
**7B recipe:** causal attention, EOS last-token pooling, no query instructions while training on MS MARCO alone, k=1–3
hard negatives (k=1 is ~45% cheaper per step at equal quality on the proxy), τ=0.02.

- **Instructions:** reintroduce them as an ablation in Phase 3, once training covers many tasks with varied instructions
  (the train/eval mismatch explanation above).
- **Runner-up config:** a1 (last + instructions), as the control for that Phase 3 ablation.
- **Distance to the reference models:** a8 is still 8.5 points below Qwen3-Embedding-0.6B (65.98) on the same harness.
  Expected after 16k MS MARCO examples; Phase 3 data and the 7B are the levers.
- **Optional before Phase 3:** a7 (τ=0.05) and a5 (k=7) remain untested, but they are cheap to re-tune later and
  unlikely to change the recipe.

## Cost / ops notes
- A100 40GB: ~25 min per k=3 arm (16 min for k=1), with eval taking ~5 min of that; 5 arms ≈ 1 h 50 min ≈ 10 CU.
  That's ~4× faster than the laptop estimate (1 h 50 min per arm). Round 2 (2 arms on a fresh runtime, including
  setup) took ≈ 1 h ≈ 5–6 CU. All of Phase 2 cost ≈ 16 CU (≈ $1.60).
- Fixes found by this run: the `data/` gitignore pattern was swallowing `src/olmo3_embed/data/`; IPython skips `{VAR}`
  expansion on a `!` line that also has `$var`; Colab's preinstalled torchao 0.10 blocks importing peft ≥ 0.21; a split
  `print(f'…` line in the notebook; `compare_published.py` crashed on `results/run_logs/*.json`.
