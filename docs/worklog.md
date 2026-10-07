# Work log: everything done so far, how long it took, what we learned

The plans (`PLAN.md`, `docs/phase*-plan.md`, `docs/task-variety-plan.md`) say what we intended. The results docs say
what each phase scored. **This file is the running diary:** each step in order, wall-clock time and Colab cost, the
iterations it took to get there (bugs, retries, wrong turns), and what we learned. Newest phase at the bottom; I append
to it as work happens.

Times are laptop local time. Colab pushes are stamped in UTC and converted here. Durations come from commit and push
timestamps, so they include waiting and debugging, not just compute. Scores are NanoBEIR nDCG@10 (×100):
**avg-13** = all 13 tasks, **avg-in-8** = the 8 task types we train on, **avg-OOD-5** = the 5 held-out task types
(ArguAna, ClimateFEVER, DBPedia, SCIDOCS, Touche2020).

## At a glance

| Phase | When | Wall clock | Colab units | Outcome |
|---|---|---|---|---|
| Setup + plan | Oct 4, 14:22–15:00 | ~40 min | 0 | Repo, plan, BEIR leaderboard snapshot, decisions |
| 0. Harness validation | Oct 4, 15:00–17:11 | ~2 h | 0 | Our eval reproduces bge-base exactly, Qwen3-0.6B within ±1.5 |
| 1. Untrained baselines | Oct 4, 17:11–20:12 | ~3 h | ~2 | OLMo-3-7B untrained = 17.35: everything must come from training |
| 2. Recipe ablations (1B) | Oct 4, 20:12 – Oct 5, 04:20 | ~8 h (overnight) | ~16 | EOS pooling + no instructions: 57.51 avg-12 (from 16) |
| 3. Track B data + proxy runs | Oct 5, 04:43 – Oct 6, 13:20 | ~32 h, incl. a ~14 h pause | ~47 | B3: 62.28 avg-13, 49.35 held-out; gate passed |
| 4. 7B run #1 | Oct 6, 13:33–21:53 | ~8.5 h | ~55 | 65.93 avg-13: level with Qwen3-4B in-domain, −9.7 held-out |
| 5. Task variety (in progress) | Oct 6, 21:39 → | ~16 h so far | ~28 so far | V1 +0.6, V2 +0.1, V3 +0.1 held-out: none clears the bar; effects ≈ Nano noise |

**Total so far:** about 3 days of calendar time and ~150 Colab units. About 232 units are left.

---

## Setup and plan (Oct 4, 14:22–15:00, ~40 min)
- Scaffolded the repo (uv project), wrote `PLAN.md`, snapshotted the BEIR leaderboard (`docs/beir_snapshot_2026-10-04.csv`).
- **Decisions (Pete):** private repo; both tracks (A = strictly clean data, B = standard public training data); Colab
  Pro for GPU; synthetic data from several generators on existing subscriptions (Claude Max, Codex, Antigravity)
  instead of paid API calls.

## Phase 0: harness validation (Oct 4, 15:00–17:11, ~2 h, laptop)
**Goal:** prove the eval harness gives published numbers before trusting it on our own model.
- bge-base-en-v1.5 (encoder path): matches published scores to within 0.3 on all 5 tasks. **Pass.**
- Qwen3-Embedding-0.6B (decoder, last-token path): within ±1.65 per task, deltas in both directions. **Pass**, with the
  note that published Qwen3 numbers are ±~1.5 per task on our harness.

**Iterations:**
- NanoTouche2020's long documents overflowed the laptop GPU, and the Windows driver *silently spilled into system RAM*
  (one process reached 14 GB; Windows commit hit 44.6/45.9 GB). Fixed by Pete changing the NVIDIA "CUDA Sysmem
  Fallback Policy" to *Prefer No Sysmem Fallback*. CUDA now raises a real out-of-memory error.

**Learned:** transformers 5 loads models in their saved dtype, so always set it; mteb files non-default runs under
`experiments/`; NanoBEIR results live under the `train` split.

## Phase 1: untrained baselines (Oct 4, 17:11–20:12, ~3 h)
- Built `OlmoEmbedder` (mean or EOS last-token pooling). OLMo-2-1B ran on the laptop, OLMo-3-7B on a Colab L4 (~1 h, ~2 units).
- **Results:** OLMo-2-1B 15.32, OLMo-3-7B **17.35** (mean pooling); EOS pooling untrained is random (0.18).

**Iterations:** Colab out-of-memory → batch 8 with automatic halving; length-sorted batching; NanoFiQA has 27 empty
documents that crashed a batch → encode them as `[EOS]`.

**Learned:** untrained LLMs aren't retrievers, and the 7B is barely better than the 1B untrained. The 1B and 7B have
the same per-task profile, which justified doing recipe experiments on the cheap 1B "proxy".

## Phase 2: recipe ablations on the 1B proxy (Oct 4, 20:12 – Oct 5, 04:20, ~8 h, ~16 units)
- Wrote `train.py` (LoRA r=16, InfoNCE, GradCache, resumable), the ablation plan, the Colab notebook, and a runbook so
  a browser-driving Claude could operate Colab.
- **Round 1** (5 arms, A100 40GB, ~25 min each, Oct 4 21:54–23:27): last-token pooling +12.9, no instructions +10.0,
  bidirectional attention +5.6, 1 vs 3 hard negatives ≈ equal.
- **Round 2** (2 arms, ~1 h, done 04:16): last-token + no instructions stack → **57.51 avg-12** (from 16.25 untrained).
  Bidirectional + last-token: −10.4 → dropped.

**Iterations (first Colab session, ~1 h of fixes):** the `data/` gitignore rule was hiding `src/olmo3_embed/data/`;
IPython doesn't expand `{VAR}` on a line that also has `$var`; Colab's preinstalled torchao 0.10 blocks peft; a broken
f-string in the notebook; `compare_published.py` crashed on run-log JSON files.

**Learned:** the A100 was ~4× faster than the laptop estimate (25 min vs 1 h 50 min per arm), so GPU work moved to
Colab for good. Without MNTP adaptation, bidirectional attention damages the EOS state.

## Phase 3: Track B data + proxy runs (Oct 5, 04:43 – Oct 6, 13:20)
**Plan** (04:43–04:59, ~20 min): Track B v1 = RLHN-680K backbone + bge Quora + BEIR SciFact/NFCorpus train sets;
decontaminate; rescore negatives; LLM judge cascade. Approved with Pete's additions: frontier models (Claude, Codex,
Gemini) as judges, and OLMo-3.1-32B on Colab.

| Step | When | Took | Where | Result |
|---|---|---|---|---|
| 1. Build (`build_trackb.py`) | Oct 5, 05:07–05:22 | ~15 min | laptop | 389k rows |
| 2. Decontaminate (`decontam.py`) vs 39 test sets | 05:22–08:07 | ~2.75 h | laptop | 373,762 rows kept; **Quora 26% leak** removed (15,499 rows) |
| 3a. Rescore code + notebook | 08:07–08:41 | ~35 min | laptop | Qwen3-0.6B rescoring, 95% guard |
| — pause (benched at 437 docs/s on A100 80GB, documented state) | 09:04 → 23:25 | ~14 h | | |
| 5. Multi-task training (`train.py`) | Oct 5, 23:25 | — | laptop | task-homogeneous batching T=0.5, false-negative masking |
| 3b. Rescore run | Oct 5 night → Oct 6, 02:50 | 3 h 05 min embedding | Colab A100 80GB | 4.86M passages; ~24 units |
| 4. LLM judge pilot | Oct 6, 01:48–09:37 | ~8 h incl. CLI setup | Colab + laptop | Claude/Codex agree (κ 0.65); OLMo-32B doesn't (κ 0.15); consensus false-negatives only 1.2% → bulk judging skipped; ~5 units |
| 6. B-runs (B1, B2, B4, B3) | Oct 6, 10:00–13:15 | ~3.4 h | Colab A100 40GB | **B3 62.28 avg-13, 49.35 held-out**; ~18 units |

**Iterations:**
- Parquet written as one huge row group → `row_group_size=20_000`; SciFact scores stored as strings; CQADupStack uses
  `id` not `_id`; empty FiQA passages.
- Rescore shards went stale after a code change → fingerprint in the shard file names.
- Codex and Antigravity CLIs: installed and signed in by Pete; they didn't work from the VS Code terminal until a
  restart; `agy` needs the prompt as an argument, not stdin. The Gemini judge pilot stopped on an Antigravity 503 at
  1,607 verdicts.
- **`push_results.sh` was deleting earlier results** from the `colab-results` branch (`git add` staged deletions from a
  fresh runtime) → fixed with `--ignore-removal`; 110 files restored.
- Colab friction: per-notebook secret grants, Drive mounts and their timeouts. Pete asked for a better way → wrote
  `docs/colab-hf-storage-plan.md` (HF Hub storage), deferred because the HF token is read-only.

**Learned:**
- 4× the data (B3) is the real win (+4.5); mixing tasks at equal compute just moves gains around.
- Rescoring + guard is worth +1.65. **Instructions hurt even when train and eval match (−4.6).**
- Single-task gaps of 2–3 points are noise at 50 queries per task; only the averages count.

## Phase 4: 7B run #1 (Oct 6, 13:33–21:53)
| Step | When | Took | Result |
|---|---|---|---|
| Plan + config + notebook | 13:33–13:49 | ~15 min | B3's exact recipe and 64k rows on OLMo-3-7B |
| Timing trials, plan approved | –15:00 | ~10 min GPU | 38–44 s/step on A100 80GB; run ≈ 42 units |
| Training | afternoon → evening | **6.1 h** (43.9 s/step) | 496 steps |
| Eval: 7B + Qwen3-4B reference | → 20:52 (pushed) | ~1.3 h | **65.93 avg-13**; Qwen3-4B 69.97 |
| Eval speed test | 21:33–21:53 | ~20 min | 7B eval is compute-bound, ~9–11k tokens/s |

**Result:** beats bge-base and Qwen3-0.6B. Level with Qwen3-4B on the 8 trained task types (75.65 vs 76.20), but
**9.7 points behind on the 5 held-out ones** (50.37 vs 60.02), and only +1.0 over the 1B there. Cost ≈ 52 units + ~3
for the speed test.

**Iterations:**
- My first full-BEIR estimate (~4 h, ~25 units) was wrong. Pete challenged it; the measured Nano eval speed (≈47
  passages/s) put full BEIR at ~60 h / ~400 units, and the speed test refined it to ~580–720 units. Corrected in the
  docs. BEIR-15 lite (~50–60 units) was then skipped for run #1 (Pete).

**Learned:** scale helps where there's training signal and barely helps elsewhere. **The gap is task-shape variety, not
model size or more of the same data.** Measure costs before quoting them.

## Phase 5: task-variety data (Oct 6, 21:39 → in progress)
**Plan** (21:39–21:57): add decontaminated data shaped like the held-out tasks, prove each source on the 1B proxy
(keep it only if avg-OOD-5 rises by more than ~1 without avg-in-8 losing more than ~1), then 7B run #2. Approved; Gemini
added as a generator alongside Claude, Codex and OLMo; Hansard + Wikipedia for argument text (Kialo and Reddit CMV
excluded for unclear reuse terms).

| Step | When | Took | Result |
|---|---|---|---|
| Document-level decontam (`doc_decontam.py`) + source vetting | 21:57–22:00 (code earlier) | — | full test corpora + all test positives indexed |
| V1 ingest: StackExchange + S2ORC citations | –22:12 | ~15 min code; streaming ingest | 139,953 rows after query decontam |
| Synthetic generator (`synth.py`) + smoke tests (Claude, Codex, Gemini) | 22:12–22:52 | ~40 min | 5 shapes, disjoint quarters per generator |
| Passage sampling (5 shapes × 8,000) | overnight | — | Hansard, Wikipedia (local 11.6 GB), arXiv abstracts |
| Synthetic generation, Claude | Oct 6 ~23:00 → Oct 7, done by 06:21 | ~7 h | 9,992 queries, all 5 shapes |
| Synthetic generation, Codex | ~23:00 → 01:26, then 06:23 → ~07:30 | ~3.5 h active | 10,000 queries; stopped at its 5-hour usage window, resumed |
| Synthetic generation, Gemini | ~23:00 → still running | 12 h+ | slowest: counterargument alone took ~5 h |
| V1 data to Colab | Oct 7, ~06:00 | — | private GitHub release asset `data-variety-v1` (49 MB) |
| **V1 proxy run (p5-v1)** | 06:05–07:59 | **~2 h** (mining 138k docs, 1 h 20 min training, eval) | **+0.61 held-out, −0.82 in-domain**: below the bar; ~10 units |
| V2/V3 collect + decontam + release `data-variety-synth-v1` | 09:40–09:47 | ~10 min | V2 7,989 rows, V3 7,981 rows |
| **V2 + V3 proxy runs** | 09:50–13:16 | **~3.4 h** (mining ~5 min each, ~1.4 h training + 8 min eval per run) | V2 +0.13 held-out, V3 +0.12: both below the bar; ~18 units |

**Iterations:**
- StackExchange's stackoverflow file is 7.8 GB → streaming with reservoir sampling.
- The climate-passage regex broke (`\b` turned into backspace characters inside a shell heredoc); the filter was then
  too loose, then too strict → strong-phrase filter, up to 3 paragraphs per article.
- **Decontam caught 44 StackExchange questions identical to FiQA test queries:** money.stackexchange is FiQA's source.
  The site was excluded (956 rows).
- **Background tool tasks are killed after 2 h** → generators relaunched as detached processes.
- I misread Gemini as "stuck for 20 min" when it had been ~2 min; added a print timeout and fail-soft CLI calls anyway
  (a failed call is retried on the next run; 5 failures in a row stop the generator).
- Decontam dropped 3 synthetic argument questions that exactly equal Touche2020 test queries (e.g. generic "Should X be
  banned?" questions). The query check matters for synthetic data too.
- Colab: Mount Drive clicked from the Files panel inserted an extra cell (deleted); a "Leave site?" dialog blocks
  reusing a tab → open a new tab.

**Learned so far:**
- **V1:** cheap public pairs barely move held-out. SCIDOCS (+1.45) moved in the expected direction from S2ORC
  citations, and ArguAna rose +2.3. StackExchange taught almost nothing: its negatives came from other forums, so
  batches sat at accuracy 1.0. Carry S2ORC into the combined run; drop StackExchange unless negatives come from the
  same site.
- **V2/V3:** synthetic data was learned (argument batches 4% → 75% accuracy) but didn't transfer to its targets beyond
  noise (ClimateFEVER +1.4 and DBPedia +1.4 for V3; nothing on ArguAna/Touche for V2). Two reasons stand out:
  entity and climate batches were trivially easy (own-pool negatives from 4k random Wikipedia passages), and Hansard
  speeches don't read like debate-site arguments.
- **Measurement:** run-to-run noise on NanoBEIR (in-domain tasks swing 1–3 points between near-identical runs; Touche
  was below B3 in every run) is as large as the effects we're trying to see. The 5 held-out Nano tasks are too small
  to judge +1-point data changes on the 1B proxy.
- **Generators:** Claude and Codex are fast enough for pilot volumes (~2,000 queries per shape in a few hours); Gemini
  via Antigravity is several times slower; Codex is bounded by its usage window.
- **Storage:** Pete is considering making Google Drive the shared store between the Windows laptop, the Mac (Colab CLI)
  and Colab; deferred for now. Data reaches Colab via private GitHub release assets in the meantime.

---

## Recurring lessons
- **Measure before quoting a cost.** The full-BEIR estimate was off by ~25×; every later estimate came from a timed run.
- **Decontaminate everything, including synthetic data.** Leaks came from places we didn't expect: Quora (26%),
  money.stackexchange (FiQA's source), and LLM-written questions that match Touche's.
- **Trust averages, not single tasks.** Nano tasks have ~50 queries; 2–3 point single-task moves are noise.
- **Colab:** an A100 is ~4× the laptop; delete the runtime after each run; per-notebook secret and Drive grants are the
  main friction; anything that must survive a disconnect goes on Drive.
- **Long laptop jobs:** run detached, make them resumable, and make failures soft so one bad call can't kill hours of
  work.
