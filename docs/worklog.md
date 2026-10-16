# Work log: everything done so far, how long it took, what we learned

The plans (`PLAN.md`, `docs/phase*-plan.md`, `docs/task-variety-plan.md`) say what we intended. The results docs say
what each phase scored. **This file is the running diary:** each step in order, wall-clock time and Colab cost, the
iterations it took to get there (bugs, retries, wrong turns), and what we learned. Newest phase at the bottom; I append
to it as work happens.

Terms and concepts explained along the way (Olmo-core vs OLMo, causal attention, post-training, ...) live in
[`docs/petes-learning.md`](petes-learning.md).

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
| 5. Task variety (A and B done) | Oct 6, 21:39 → Oct 8, 02:30 | ~29 h | ~79 | Full-size held-out: V1 +1.20, V2 +0.66, V3 +0.59, V3b +0.27, V4 (all) +0.92: gains don't stack at 64k; carry V1 + V2 to 7B run #2; A (instructions) kept, B (stance negatives) dropped |

**Total so far:** about 3.5 days of calendar time and ~201 Colab units. About 174 units are left.

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
| Pause Gemini (classifier blocked my kill; Pete ran the command) | 13:30 | — | Gemini kept: argument 1,775, counterargument 1,724, citation 2,000, entity 752 |
| Full-size held-out suite + Wikipedia mining pool (`synth.py pool`, `rescore.py --pool`) | 13:20–14:05 | ~45 min | pools: 500k entity leads, 223k climate paragraphs |
| **Full-size held-out eval of B3, V1, V2, V3** | ~15:00–16:03 | **~12 min each** | V1 37.86 (+1.20 vs B3's 36.66): first source to clear the bar; V2 +0.66, V3 +0.59 |
| **V3b proxy run** (V3 + pool negatives) | 16:03–18:25 | ~2.4 h (mining ~25 min, training ~1.4 h, both evals) | 36.93: no better than V3; dropped. Session ~18 units |
| **V4 proxy run** (V1 + V2 + V3 combined) | 19:13–21:08 | ~1.9 h (training 492 steps ~1.5 h, Nano 7 min, held-out 12 min) | heldout-4 37.58 (+0.92): below V1 alone; best ArguAna (56.0) and best Nano avg-13 (62.35); ~10 units |
| Outside review of the pipeline → experiments A (instructions + dropout) and B (filter + stance negatives) | 21:15–21:35 | ~20 min | kept 3 ideas, rejected 4 with reasons (task-variety plan) |
| A: `instruction_dropout`, instructions for new tasks, held-out suite on `.v2` tasks; launch `p5-v2i` | 21:35–21:47 | ~12 min + Colab | training from 21:47 |
| B: round-trip filter (laptop GPU), stance-negative generator, attach, rescore support | 21:40–22:05 | ~25 min | filter kept 6,382 / 7,989; Claude + Codex generating 1,600 negatives |
| **A result (`p5-v2i`)** | 21:47–23:37 | ~1.9 h, ~10 units | heldout-4 37.92 (+0.60 vs p5-v2), Nano avg-in-8 70.70 (+0.63), avg-13 63.08 (best proxy yet): instructions kept for 7B run #2 |
| Hand-off prompt for a Mac Claude Code instance (Colab CLI); B notebook `variety_b_stance_colab.ipynb` | 22:30 | ~15 min | Mac runs B once release `data-variety-v2b` exists |
| B data: stance-flipped negatives (Claude 21:45–23:25, Codex 21:45–00:26), attach, release `data-variety-v2b` | 21:45–00:30 | Claude ~1.7 h, Codex ~2.7 h for 800 each | 1,600 negatives (median 1,307 chars), 0 test-document hits; 6,382 rows; ready for the Mac to run B |
| B first attempt (Mac, Colab CLI): runtime + Drive consent, waiting cell for the release | 23:22–23:49 | ~0.5 h, ~2.6 units lost | the Mac idle-slept at 23:49, the CLI's websocket dropped, and Colab reclaimed the runtime before the data existed |
| **B result (`p5-v2b`)**, headless, no Drive: Track B parquet uploaded from the Mac's Drive copy | 00:27–02:28 | ~2 h, ~10.5 units | heldout-4 37.12 (**−0.20 vs p5-v2**), **ArguAna −2.01**, avg-in-8 70.02 (−0.05): B not adopted |
| 7B run #2 recommendation + cost (task-variety plan) | 02:30 | — | Track B + V1 + V2 at 96k, instructions + dropout, ≈ 75 units (70–85), timing trial first |
| **7B run #2 (`p4-run2-7b`), session 1**, headless (Colab CLI, A100 80GB via `--high-mem`, 6.77 u/h) | Oct 8, 06:17–12:13 | ~6 h, ~39 units | timing trial 58 s/step (4 steps; ckpt-c16 69) → actual ~47 s/step avg; **paused at checkpoint step 410 / 743** for Pete to disconnect; resumes tonight from Drive |

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
- **Full-size held-out eval was the missing yardstick.** At ~12 min and ~1 unit per 1B adapter, it showed V1 at
  +1.20 (Nano said +0.61) and V2 helping ArguAna by +2.65 (Nano said 0). Use it for every proxy run from now on.
- **Harder negatives aren't enough on their own:** a 500k-doc Wikipedia mining pool made V3's batches harder but didn't
  improve DBPedia.
- **Combining sources at a fixed budget dilutes them (V4).** The held-out gains overlap on ArguAna instead of adding
  up, and the Wikipedia-style tasks (ClimateFEVER, DBPedia, FEVER) lose ground as Track B's share shrinks. The fix is
  a bigger budget, not fewer sources.
- **Instructions flipped from −4.6 (Phase 3, 8 search-like tasks) to positive (+0.6 held-out, +0.6 in-domain)** once
  the training data had task shapes that differ in what "relevant" means, and with 20% instruction dropout. An early
  negative result was specific to its data, not general.
- **The teacher can't see stance.** Qwen3-0.6B ranks a passage that agrees with an argument above the real rebuttal in
  31 of 32 rows, so guard rules built on the teacher would have thrown away every stance negative. That's also the
  clearest picture yet of why ArguAna is hard for similarity-trained embedders.
- **Iterations:** pool build ran out of memory (exploding paragraphs copied each article's text onto every row); the
  doc-decontam index only unpickled when its own script ran (`__main__.DocIndex`); Chrome disconnected, so Pete ran
  Run all and granted access. On V4's restart one of my clicks added an empty text cell; deleted before rerunning.
- **Generators:** Claude and Codex are fast enough for pilot volumes (~2,000 queries per shape in a few hours); Gemini
  via Antigravity is several times slower; Codex is bounded by its usage window.
- **Storage:** Pete is considering making Google Drive the shared store between the Windows laptop, the Mac (Colab CLI)
  and Colab; deferred for now. Data reaches Colab via private GitHub release assets in the meantime.

### Colab CLI on the Mac (Oct 7, 22:34–23:05, ~30 min, < 0.5 units)
From here on, a second Claude Code instance on Pete's Mac drives Colab with Google's Colab CLI (`colab`, v0.7.4,
`uv tool install google-colab-cli`), with no browser. The CLI only supports Linux and macOS, which is why the Windows
laptop used Chrome.

| Step | Took | Result |
|---|---|---|
| Install + sign in (oauth2, personal Google account) | ~5 min | Its own token in `~/.config/colab-cli/`; gcloud/ADC untouched |
| CPU probes: secrets, `--env`, upload, Drive | ~15 min | Secrets don't work headless (see below) |
| Drive wrapper + token script | ~10 min | `scripts/colab_drivemount.py`, `scripts/colab_secrets.sh` |
| **A100 smoke** (`notebooks/colab_cli_smoke.ipynb`) | ~4 min GPU | A100-SXM4-40GB, token file read, Drive listed, clone OK; runtime deleted |

**What's different under the CLI:**
- **Runtimes:** `colab new --gpu A100 -s NAME` gave an A100 in ~3 s. `colab exec -s NAME -f nb.ipynb` runs a notebook
  cell by cell and streams output, then saves `nb_output.ipynb` next to it (now gitignored). `colab stop -s NAME`
  deletes the runtime.
  - `colab exec --timeout` is an inactivity timeout (default 30 s).
  - The CLI only sees runtimes it created, so a browser runtime (like A's) shows as `[?]` in `colab sessions` and
    can't be attached to.
- **Secrets:** `userdata.get` fails ("Secrets can only be fetched when running from the Colab UI"). The fix:
  - `colab exec --env KEY=VALUE` works, but **writes the value in plain text** to `~/.config/colab-cli/history/*.jsonl`
    and `colab.log`, so it's not used for secrets.
  - `scripts/colab_secrets.sh` uploads the GitHub token from the macOS Keychain (`olmo3-colab-gh`, a fine-grained PAT
    for the Colab repos) to `/root/.olmo3_GH_TOKEN`; uploads aren't logged.
  - Cell 1's `secret(k)` reads that file, else `userdata`, so browser runs are unchanged. Only B's notebook has it
    so far.
- **Drive:** `colab drivemount` relays Colab's usual consent link, which is needed **again for every runtime** (it
  isn't remembered). Under the CLI it waits for Enter on `/dev/tty`, while `drive.mount()` times out after 120 s.
  `scripts/colab_drivemount.py` waits for a signal file instead and gives the consent 20 min. Once mounted, the
  notebook's own `drive.mount()` just says "already mounted".
- **Mac side:** Drive for desktop has the personal Google account, so `runs/` and `data/` are readable on the Mac (e.g. A's
  final adapter appeared at 23:16). The VM can't see the Mac's copy.
- `uv sync` failed on macOS (cu128 torch has no Mac wheels) → the CUDA index now applies only off macOS.

**Iterations:** the first two consents timed out: the first from the 120 s mount limit, the second from my own 10-min
wait, 24 s before Pete's "done". `colab upload` paths are relative to `/`, and the target folder must already exist
(500 otherwise). The CLI's `colab.log` holds OAuth tokens at debug level and was world-readable → `chmod 600`.

**Running B headless (Oct 8, 00:27–02:28):**
- **The Mac must stay awake.** Claude Code holds `caffeinate` only for 300 s per tool call, so between turns the Mac
  idle-slept (23:49) and the CLI's websocket dropped. With no client connected, Colab reclaimed the runtime (first
  attempt, ~2.6 units). The rerun ran under `caffeinate -ims` and stayed connected for 2 h.
- **Without Drive:** `colab upload` takes one base64 request per file and rejects ~100 MB (50 MB works, ~7 MB/s). The
  2.5 GB Track B parquet went up from the Mac's Drive for desktop copy in 48 × 50 MB parts (~7 min) and was joined
  and checked by sha256 on the VM. The adapter and V2b data came back as 40 MB parts into the Mac's Drive folder,
  which syncs to Drive. No consent click was needed, but checkpoints don't survive a lost runtime.
- **The exec ended with `TimeoutError: Timeout waiting for reply`** right after the push (the last cell never
  started), so the auto-cleanup didn't run. I packed, downloaded and deleted the runtime by hand ~2 min later.
  Cleanup should check the result on `colab-results`, not the exec's exit status.

**7B run #2, session 1 (Oct 8, 06:17–12:13):**
- `colab new --gpu A100 --high-mem` gives the A100-SXM4-80GB High-RAM shape (6.77 u/h), same as run #1's browser runtime.
- **I lost a runtime to my own bug** (~2.5 units, plus one more Drive consent): the train-only notebook skipped the
  setup cell, so it ran in `/content` (a later `colab exec` doesn't keep the earlier one's `%cd`) and couldn't find
  the config. `colab exec -f` also **keeps going after a failed cell**, and my done marker printed unconditionally,
  so the script deleted the runtime. Fix: the run notebook always includes the setup cells, and the done marker
  prints only if `runs/<run>/.pushed` exists.
- **A network drop doesn't kill the runtime while the Mac is awake.** The websocket dropped at 07:42, and the CLI
  noticed at 07:51. Training kept going (Drive checkpoints at 07:48 and later), and the next `colab exec` queued its
  cells behind the running one. Progress was then read from the checkpoint step on Drive, through the Mac's Drive
  copy.
- **Pausing for a disconnect:** wait for the next checkpoint (every 10 steps ≈ 8 min) to land in the Mac's Drive
  copy, then `colab stop`. Done at step 410 with nothing lost.

**7B run #2, session 2 (Oct 8, 20:04 – Oct 9, 10:18): finished and pushed.**
- **Resume from step 410 worked:** a new runtime plus one Drive consent, and the train cell printed "resumed from step 410/743".
- **I lost a runtime to a `--timeout` change** (~8 units). I raised `colab exec --timeout` from 3600 to 36000 so the
  training cell wouldn't time out. But the CLI's client sleeps on one event until a message arrives or the deadline
  passes, and only then checks the connection. The websocket dropped at 22:26, and the client didn't notice until 06:12.
  Training ran on to a step 610 checkpoint (22:45), and then Colab reclaimed the runtime (with no client, after ~20 min).
- **A short `--timeout` isn't the fix either:** once the deadline passes, the reply wait gets 0 s, so any cell longer
  than the timeout fails with "Timeout waiting for reply". With 60 s, setup cell 1 failed on four attempts in a row.
- **Fix: `scripts/colab_exec_watch.py`** runs `colab exec` with each internal wait capped at 30 s, so a long
  `--timeout` (36000) still notices a dropped connection within about 30 s, and the driver can re-exec. It ran
  610 → 743 plus both evals (07:18–10:18) with no drop to test it on, and a 70 s smoke cell used almost no CPU.
- **Cost:** session 2 was ~38.7 units (135.03 → 96.29), so run #2 came to ~78 units, within the ~95 approved.

**Run #2 result** (Nano = NanoBEIR nDCG@10, max 512 tokens; raw results on `colab-results`):

| | run #1 | run #2 |
|---|---|---|
| NanoBEIR avg (13) | 65.93 | 66.08 |
| avg-in-8 | 75.65 | 75.67 |
| avg-OOD-5 | 50.37 | 50.73 |
| heldout-4, full size | 40.14 | **42.13** (+1.99) |
| ArguAna (full) | 57.57 | 60.45 (+2.88) |
| SCIDOCS (full) | 21.96 | 23.81 (+1.85) |
| ClimateFEVER HardNeg (full) | 34.62 | 34.44 (−0.18) |
| DBPedia HardNeg (full) | 46.42 | 49.81 (+3.39) |

Nano moves are within noise (+0.15 avg). **On the full-size held-out tasks, run #2 is +1.99** (3 of 4 tasks up, the
fourth flat). These tasks have hundreds to thousands of queries each, so this is the signal Nano couldn't give. Run #1's
held-out eval ran Oct 9, 10:41–11:36 (eval only, ~6.2 units). Caveat: run #2 changed the data (+ V1 + V2) and added
per-task instructions at once, so the gain is the recipe's, not the data's alone. On the 1B proxy, both moved held-out
by similar amounts (V1 +1.20, V2 +0.66, instructions +0.60 on heldout-4).

**Run #2 on 8 BEIR tasks (Oct 9, 13:06–17:51, ~32 units):** full BEIR-15 is out of budget. Measured throughput is
~35–50k chars/s for the 7B on an A100, and BEIR-15 is ~14B chars (33M docs) → ~80–100 A100-h ≈ 550–700 units. The
plan's 6–12 GPU-h was ~10× low. Instead: the 8 tasks with small or medium corpora (ArguAna, SCIDOCS + NFCorpus, SciFact, FiQA,
Quora, TREC-COVID, CQADupstack), ranked against the 139 15-task models in `docs/beir_snapshot_2026-10-04.csv`.

| | ArguAna | SCIDOCS | NFCorpus | SciFact | FiQA | Quora | TREC-COVID | CQADup | 8-avg | BEIR-15 |
|---|---|---|---|---|---|---|---|---|---|---|
| **run #2** | 60.5 | 23.8 | 39.7 | 79.5 | 52.9 | 89.8 | 65.9 | 46.2 | **57.27** | ~55 (est.) |
| LLM2Vec-Llama-3-8B (T1) | 62.8 | 23.0 | 41.8 | 78.2 | 55.3 | 87.2 | 80.3 | 48.2 | 59.61 | 56.6 |
| Qwen3-Embedding-0.6B | 71.0 | 24.4 | 36.7 | 69.7 | 46.6 | 87.8 | 90.5 | 46.0 | 59.09 | 55.5 |
| Qwen3-Embedding-4B | 75.6 | 31.4 | 41.1 | 78.3 | 62.6 | 88.1 | 92.9 | 50.3 | 65.05 | 61.6 |

- **Rank 30 of 139 on the 8-task average.**
- **BEIR-15 estimate ~55 (rank ~33), so T1 (56.6) is probably ~1.5 short.** Two methods agree: (a) the 8-avg →
  BEIR-15 fit across the 139 models (R² 0.98, residual sd 2.0) gives 55.1; (b) the 8 measured tasks plus the other 7
  projected from our NanoBEIR deltas vs the Qwen3 references give 54.7. On the 6 measured tasks, (b) was within
  ~2 points except ArguAna (+8, Nano overstates).
- **Biggest gaps:** TREC-COVID (−14 vs LLM2Vec, −25 vs Qwen3-4B) and, by projection, Touche and ClimateFEVER.
  In-domain (NQ, HotpotQA, FEVER) projects at or above Qwen3-4B.

**Phase 6 data: PubMedQA biomedical slice, built and decontaminated on the Mac (Oct 9, 20:56–21:30):**
- **Source:** `qiaojin/PubMedQA` `pqa_artificial` (MIT), 30,000 rows sampled (seed 0) of 211,269, one per PubMed id:
  research question (generated from the title) → full abstract (context paragraphs + conclusion, median 1,633 chars).
  `build_variety.py --sources pubmedqa`.
- **Zero-shot protocol for TREC-COVID** (Pete, 2026-10-09): no TREC-COVID queries or qrels, and no document overlap
  with its 171,332 CORD-19 docs (nor NFCorpus / SciFact, which are PubMed too).
- **Test cache:** the Mac had no `data/raw/beir_test/` and only 22 GB free disk. The laptop's 39 per-task JSONs
  (queries + relevant-doc openings, 47.9 MB) came over as release `beir-test-cache-v1`, all repo/revision pairs equal
  to the current mteb tasks. With them the index build only downloads the 8 small full corpora + CQADupStack.

| Step | Result |
|---|---|
| Query decontam (`decontam.py`, 72,055 test queries, 39 tasks) | 0 / 30,000 dropped |
| Doc index rebuild (`doc_decontam.py --build`, 1,853,796 full-corpus docs, 139,199 test positives) | ~3.5 min, 3 GB RAM |
| Old rules (exact, prefix, near-dup of first 300 chars) | 5 dropped: 2 TREC-COVID prefix, 3 NanoNFCorpus positives |
| **New `doc_contain` rule** (≥5% of word 8-grams in a TREC-COVID / NFCorpus / SciFact corpus) | **186 more dropped**: 159 TREC-COVID, 15 NFCorpus, 12 SciFact |
| **Kept** | **29,809** |

- **The old rules missed ~140 near-verbatim copies of test documents** (≥50% of 8-grams; 118 CORD-19). CORD-19
  stores "title + section label + abstract" and PubMedQA the bare abstract: the same text, offset, so exact/prefix
  matching and the first-300-char MinHash both fail. Found by an independent 8-gram containment check run because 2
  hits looked low for 299 coronavirus-related rows.
- **Threshold:** overlap is bimodal (139 rows ≥50%, 1 at 20–50%, 46 at 5–20%, 1,408 at 2–5%). 5–20% looked like
  related papers sharing sentences, dropped anyway under zero tolerance; below 5% is shared boilerplate.
- **Verified:** after the rule, the independent check finds 0 rows at ≥5% overlap with any of the three corpora.
- **Learned:** document decontamination needs a containment check against the whole test corpus, not just a prefix
  or near-duplicate match, whenever the source and the test corpus format the same text differently. `doc_contain`
  is now permanent in `doc_decontam.py` (index v2), so every future source gets it.

**Phase 6 cleanup: the training data re-checked against index v2 (Oct 9, 21:50–23:45, Mac CPU, no units):**
Pete's zero-contamination commitment covers every source run #3 trains on, not just PubMedQA. So V1, V2 and Track B
were re-checked against the new index (`doc_near` + `doc_contain` over TREC-COVID, NFCorpus, SciFact and now SCIDOCS).

| Data | Rows with a flagged positive | Flagged negatives | Mostly |
|---|---:|---:|---|
| V1 (StackExchange + S2ORC), **used in run #2** | 520 / 138,962 | ~6,800 | S2ORC cited titles that are CORD-19 / SciFact / NFCorpus / SCIDOCS paper titles |
| V2 (synthetic debate) | 0 / 7,989 | 0 | |
| PubMedQA (+ SCIDOCS checks) | 2 / 29,809 | 24 | SCIDOCS |
| SciRepEval search (new fallback source) | 988 / 29,997 | n/a | TREC-COVID, SCIDOCS |
| Track B, zero-shot test sets only | 528 / 373,762 | ~4,460 | TREC-COVID (494 positives), SCIDOCS (107) |

- **Why V1 leaked:** index v1 matched whole documents only. Index v2 also indexes each TREC-COVID / NFCorpus /
  SciFact / SCIDOCS document's title alone, and S2ORC's positives are bare cited titles: 157 equal a CORD-19 title.
- **Track B policy:** only overlap with the test sets that have **no** BEIR train split was removed (TREC-COVID,
  SCIDOCS, ArguAna, Touche2020, CQADupStack; `doc_decontam --only-tasks`). Overlap with the tasks Track B trains on
  (Quora, NFCorpus, FiQA, HotpotQA, ...) is by design and documented in the data card. Dropped rows: MS MARCO 443,
  HotpotQA 68, NQ 17. Short passages need only ~3 shared 8-grams to reach 5%, so some of these are likely common
  phrasing; dropping them (0.14%) is the conservative side.
- **Cleaned copies on Drive** (new folders; the originals stay for provenance): `trackb_v1c` (373,234 rows),
  `variety_v1c` (138,442), `variety_bioc` (29,807). Negatives keep their `neg_scores` aligned. Release
  `data-variety-bio-v2` supersedes `-v1`. `configs/7b-run3.yaml` and its notebook now read the cleaned copies.
- **Caveat on run #2 (and run #1 for Track B):** their training data contained TREC-COVID / SCIDOCS documents, mostly
  as *negatives* (pushed away from queries) and as titles. That more likely hurt than helped those tasks, but run #2's
  TREC-COVID (65.9), SCIDOCS (23.8), SciFact and NFCorpus scores are not strictly zero-overlap. Run #3 is the first
  run trained without it.
- **Also found:** `decontam.py` wrote its manifest to Track B's committed path whatever `--data` was. A PubMedQA run
  emptied it. Restored before any commit, and the default now follows `--data`.
- **p6-bio** (the proxy) trains on PubMedQA v1, so it includes the 2 SCIDOCS rows: negligible for the gate.
- **Learned:** re-check *all* training sources whenever the decontamination rules get stronger. A rule that is new
  for one source is new for the others too.

**Phase 6 step 2: p6-bio proxy gate (Oct 9, 21:38 – Oct 10, 00:13; personal account, A100 40GB, ~13.9 units):**
p5-v2i's exact recipe plus the PubMedQA slice (`notebooks/phase6_bio_proxy_colab.ipynb`), against p5-v2i scored on the
same full-size tasks.

| Task (full size unless Nano) | p5-v2i (1B) | p6-bio (1B + PubMedQA) | Δ | 7B run #2 |
|---|---:|---:|---:|---:|
| TREC-COVID (50 queries) | 79.80 | 80.56 | +0.76 | **65.90** |
| NFCorpus | 36.39 | 36.58 | +0.19 | 39.66 |
| SciFact | 75.22 | 75.89 | +0.67 | 79.55 |
| heldout-4 | 37.92 | 38.10 | +0.18 | 42.13 |
| Nano avg-in-8 | 70.70 | 71.36 | +0.66 | 75.67 |
| Nano avg-13 | 63.08 | 63.54 | +0.46 | 66.08 |

- **Gate: fails on its main criterion.** +0.76 on 50 queries isn't a measurable TREC-COVID gain. SciFact and NFCorpus
  didn't drop, and in-domain is within ±1. Per the agreed rule, PubMedQA isn't the fix: it's harmless and slightly
  positive everywhere, but too small to carry a 7B run on its own.
- **The real finding is the baseline:** the 1B p5-v2i scores **79.8 on TREC-COVID, 14 points above the 7B run #2
  (65.9)**, while the 7B leads almost everywhere else. So TREC-COVID was never a biomedical-domain gap. Something
  specific to run #2 depresses it. Prime suspect: V1, which run #2 trained on and p5-v2i didn't, and whose negatives
  include ~4,700 CORD-19 titles (found by the index-v2 cleanup), teaching the model to push TREC-COVID documents
  away. Diagnostic running: B3 (Track B only) vs p5-v1 (Track B + V1) on full TREC-COVID.
- **Diagnostic (Oct 10, 04:21–04:57, personal account, ~3 units): V1 is not the cause.** Full TREC-COVID: B3 (Track B)
  72.80, p5-v1 (Track B + V1) 74.02, p5-v2i (Track B + V2, instructions) 79.80, p6-bio 80.56, 7B run #2 65.90. V1's
  CORD-19 negatives didn't depress TREC-COVID at 1B (+1.2). Instructions + V2 lift the 1B by ~7. The 7B, on the same
  kinds of data with instructions, is 14 below the 1B: the cause is specific to the 7B. Next suspects: the 7B's
  handling of the unseen TREC-COVID instruction, and TREC-COVID's shallow judgments (a model that retrieves different,
  unjudged CORD-19 papers is scored as wrong). `eval.py` now has `--run-instructions on|off`, `--name-suffix` and
  `--predictions` (saved rankings, for a judged@10 check).

**Phase 6 diagnostic 2: what holds the 7B down on TREC-COVID (Oct 10, 06:05–08:47; personal account, A100 40GB, ~14.2 units):**
Run #2 on full TREC-COVID twice, rankings saved (`notebooks/phase6_treccovid_7b_colab.ipynb`, `eval.py
--run-instructions/--name-suffix/--predictions`), then `scripts/judged_at_k.py` on the Mac.

| 7B run #2 | MTEB nDCG@10 | judged@10 | nDCG@10, unjudged removed |
|---|---:|---:|---:|
| as trained (instructions on) | 65.90 (reproduced exactly) | 0.742 | 83.9 |
| instructions off | 56.99 | 0.666 | 78.5 |

- **Instructions help (+8.9); they aren't the cause.**
- **The cause: a strong pull toward title-only CORD-19 entries.** 98% of the 129 unjudged top-10 documents have no
  abstract, against 24.6% of the corpus and 15.6% of all judged documents. TREC-COVID's 2020 pools rarely judged
  those entries, so they count as misses even when they look relevant ("Is reinfection possible after recovery from
  COVID-19?" for the post-recovery immunity query). On judged documents only, run #2 scores 83.9.
- **No embedding collapse:** 485 distinct documents fill the 500 top-10 slots, and no document recurs in more than 3
  queries' top 10.
- **Suspect:** V1's S2ORC pairs are bare title → bare title (~60k rows), which reward matching a query to a short
  title. At 1B, V1 didn't hurt TREC-COVID (+1.2), but the 7B may learn the shortcut harder. Possible run #3 fix: use
  `sentence-transformers/s2orc` title → abstract pairs instead.
- **Next checks (proposed, ~9 units):** rankings from the 1B p5-v2i (is the title-only pull 7B-specific?) and from 7B
  run #1, which had no V1 (does V1 cause it?).
- **Also learned:** an argparse value starting with `-` must be passed as `--opt=value`. A notebook's "done" marker
  must check the work's own output, not the push's exit code: a failed eval followed by "nothing new to push" wrote
  both markers, and the driver deleted the runtime. The driver now stops instead of retrying after a cell fails
  without a dropped connection.

**Phase 6 diagnostic 3: the title-only pull comes from the 7B base, not V1 (Oct 10, 09:14–10:50; personal account, A100 40GB, ~9 units):**

Full TREC-COVID rankings for the 1B p5-v2i and 7B run #1 (`notebooks/phase6_treccovid_rank_colab.ipynb`),
compared with run #2's saved rankings.

| Model | Training data | nDCG@10 | judged@10 | nDCG@10, unjudged removed | top-10 with no abstract |
|---|---|---:|---:|---:|---:|
| 1B p5-v2i | Track B + V2 | 79.80 | 0.900 | 83.6 | 24.0% |
| 7B run #1 | Track B only, no instructions | 58.12 | 0.684 | 80.5 | 76.6% |
| 7B run #2 | Track B + V1 + V2 | 65.90 | 0.742 | 83.9 | 73.4% |
| 7B run #2, instructions off | | 56.99 | 0.666 | 78.5 | 82.6% |

The corpus is 24.6% title-only, so the 1B ranks title-only entries at their base rate. Both 7Bs fill about three
quarters of their top 10 with them.

- **V1 isn't the cause.** Run #1 never saw V1 and shows the same pull (99% of its unjudged top-10 hits have no
  abstract). Swapping S2ORC title → title for title → abstract wouldn't fix it.
- **It follows the base model.** The two 7B runs share only the base model (OLMo-3-7B against OLMo-2-1B) and the
  recipe settings, which the 1B shares too.
- **On judged documents the 7B is as good as the 1B** (83.9 against 83.6). The 14-point gap on the official score
  is almost all unjudged title-only entries counted as misses. TREC-COVID's 2020 pools rarely judged them.
- **Implication:** recovering TREC-COVID means teaching the 7B to rank a full abstract above a bare title on the same
  topic. That optimizes against a pool artefact more than it improves retrieval.

**7B run #3: run #2's recipe on the cleaned data (Oct 10, 15:12 – Oct 11, 09:10; cybercontext, A100 80GB, ~121 units):**
`configs/7b-run3.yaml`, `notebooks/phase6_7b_run3_colab.ipynb`: Track B + V1 + V2 + PubMedQA, all cleaned copies (no
document overlap with the zero-shot test corpora), 115k rows, 892 steps at ~47 s/step (11.7 h), then 4.5 h of evals.

| Task (full size) | Run #3 (clean) | Run #2 | Δ |
|---|---:|---:|---:|
| ArguAna | 61.38 | 60.45 | +0.93 |
| SCIDOCS | 23.29 | 23.81 | −0.53 |
| NFCorpus | 37.46 | 39.66 | −2.20 |
| SciFact | 80.21 | 79.55 | +0.66 |
| FiQA | 53.18 | 52.88 | +0.30 |
| Quora | 89.59 | 89.76 | −0.17 |
| TREC-COVID | 60.16 | 65.90 | **−5.74** |
| CQADupstack | 45.45 | 46.16 | −0.71 |
| **8-task average** | **56.34** (rank 36/139) | 57.27 (rank 30) | −0.93 |
| BEIR-15 estimate (fit) | ~54.1 (rank ~42) | ~55.1 (rank ~33) | |
| ClimateFEVER-HN / DBPedia-HN | 34.52 / 49.95 | 34.44 / 49.81 | +0.08 / +0.15 |

- **Most of the drop is TREC-COVID, where run #2 trained on test documents.** Run #2's uncleaned Track B had 494 rows
  whose positive is a TREC-COVID test document. Without them, TREC-COVID falls 5.7 points. So part of run #2's 65.90
  was overlap, not zero-shot skill. Run #3's numbers are the honest ones and the submission candidate.
- **Elsewhere the two runs are close** (7-task average without TREC-COVID: 55.80 vs 56.04). The PubMedQA slice didn't
  lift NFCorpus or SCIDOCS at 7B.
- **Ops:** the Mac's websocket dropped twice during the evals; attempt 3's `colab exec` hung at "Parsing notebook"
  from 05:20 while the attempt-2 eval kept running in the kernel (status showed IDLE). I stopped the driver, waited
  for the result files with `colab ls`, fetched them with `colab download`, and pushed from the Mac. Lesson: a
  re-exec while the kernel still runs the old cell blocks silently; check `colab ls` for progress before re-execing.

**Phase 7 step 1: merging two 1B runs doesn't help (Oct 11, 09:58–12:16; cybercontext, A100 40GB, ~12.9 units):**
`notebooks/phase7_merge_proxy_colab.ipynb`. p7-s1 = p5-v2i's recipe with seed 1 (different rows and order); p7-merge = the
exact average of the two runs' LoRA updates (`olmo3_embed.merge`, rank concatenation, checked to 1e-7).

| | p5-v2i (seed 0) | p7-s1 (seed 1) | p7-merge |
|---|---:|---:|---:|
| heldout-4 (full size) | 37.92 | 37.42 | 37.94 |
| Nano avg-13 | 63.08 | 63.37 | 62.93 |
| Nano avg-in-8 | 70.70 | 71.70 | 70.22 |
| NFCorpus / SciFact | 36.39 / 75.22 | 37.16 / 75.97 | 36.74 / 75.03 |
| ArguAna / SCIDOCS | 56.11 / 19.73 | 54.18 / 19.60 | 54.93 / 19.70 |
| ClimateFEVER-HN / DBPedia-HN | 31.14 / 44.68 | 31.61 / 44.30 | 32.43 / 44.68 |

- **Gate fails.** The merge ties the better run on heldout-4 (+0.02) and is 0.44 below it on Nano avg-13. It lands at
  about the average of the two runs, not above them.
- **Seed noise is about 0.5 points** (heldout-4 37.92 vs 37.42; Nano avg-in-8 70.70 vs 71.70). Recipe differences
  smaller than that, measured on one seed, aren't real.
- **Decision:** no second 7B run just to merge with run #3.

**Phase 7 step 2: 7B run #4, run #3 without V2 (Oct 11, 14:48 – Oct 13, 02:03; cybercontext, A100 80GB, ~120 units):**
`configs/7b-run4.yaml`, `notebooks/phase7_7b_run4_colab.ipynb`. Drops V2, the ~8k Claude/Codex-written Hansard rows, so
every training row is human-written public data. 115k rows, 893 steps, then Nano + held-out + 8-task BEIR.

| Task (full size) | Run #4 (no V2) | Run #3 | Δ |
|---|---:|---:|---:|
| ArguAna | 57.82 | 61.38 | −3.56 |
| SCIDOCS | 23.65 | 23.29 | +0.36 |
| NFCorpus | 38.98 | 37.46 | +1.52 |
| SciFact | 79.47 | 80.21 | −0.74 |
| FiQA | 53.39 | 53.18 | +0.21 |
| Quora | 89.66 | 89.59 | +0.07 |
| TREC-COVID | 63.10 | 60.16 | +2.94 |
| CQADupstack | 45.76 | 45.45 | +0.31 |
| **8-task average** | **56.48** (rank 35/139) | 56.34 (rank 36) | +0.14 |
| BEIR-15 estimate (fit) | ~54.2 (rank ~39) | ~54.1 (rank ~42) | |
| ClimateFEVER-HN / DBPedia-HN | 36.39 / 49.92 | 34.52 / 49.95 | +1.87 / −0.03 |
| heldout-4 | 41.95 | 42.29 | −0.34 |
| Nano avg-13 | 66.33 | 66.24 | +0.09 |

- **A tie.** Every average moves by less than the ~0.5 seed noise. V2 was worth nothing measurable at 7B.
- **Per task, the swings cancel.** ArguAna drops 3.6 (V2's Hansard questions resemble its argument-retrieval style);
  TREC-COVID gains 2.9, ClimateFEVER-HN 1.9 and NFCorpus 1.5. With one seed per run, single-task moves of this size
  may be noise.
- **Decision: run #4 is the submission candidate.** Same quality, and no model-written training data, so no question
  about the providers' terms.
- **Ops:** the Mac lost DNS at 21:20 on Oct 11 (step ~495). The driver's `colab sessions` check failed on the token
  refresh and it exited as "runtime gone", while the runtime trained on alone until ~22:25 (step ~575) and was then
  reclaimed. Resumed Oct 12, 15:58 on a new runtime from the step-570 Drive checkpoint; finished in one attempt.
  Three consent links expired or broke before that (~8 units of idle runtime; one link was mistyped in transcription).
  Fix: the driver now retries the sessions check for 30 min before deciding the runtime is gone. Lessons: create the
  runtime only when Pete is ready to click; keep the Mac on power and Wi-Fi overnight.
- Balance after run #4: ~32.6 units (cybercontext).

**Phase 8: full BEIR-15 for run #4 (launched Oct 13, 10:42; personal account, Google AI Ultra, 3× G4):**
The leaderboard needs all 15 tasks; run #4 has 8. The other 7 (MSMARCO, FEVER, ClimateFEVER, HotpotQA, DBPedia, NQ,
Touche2020) are ~30M documents.

- **Account:** the personal account was upgraded to Google AI Ultra (896.55 units). Background execution checked first
  on CPU: a busy cell ran 95+ min after the CLI client was killed (99 log lines, no gaps).
- **Adapter published** as release `model-p7-run4-7b` (zip with adapter, configs, README), so runtimes fetch it
  without Drive: no consent click.
- **Faster eval** (`model.py`, off by default): token-budget batching (fill batches to 32k padded tokens), a float16
  per-document embedding cache (FEVER and ClimateFEVER share one 5.4M-passage corpus, embedded once), cosine
  similarity on the GPU. `scripts/bench_encode.py`, 20k real docs per corpus:

| chars/s (7B, ≤512 tokens) | A100 fixed 64 | A100 budget 32k | G4 budget 32k |
|---|---:|---:|---:|
| MSMARCO | 33k | 52k | 72k |
| FEVER | 40k | 52k | 72k |
| Touche2020 | 83k | 90k | 125k |

  Budget vs fixed batching: same embeddings up to bf16 noise (min cosine 0.996 over 20k docs, typically ~1.0); cache
  round trip max diff 1e-4; GPU similarity matches mteb's cos_sim (diff ≤1e-6). G4 = RTX PRO 6000 Blackwell 96 GB,
  ~8.9 units/h (A100 80GB 6.77), 192 GB disk, 190 GB RAM. H100: "Service Unavailable" four times today.
- **One cell, no Drive** (`notebooks/phase8_beir15_colab.ipynb`, generated by `scripts/make_beir15_notebook.py`):
  bootstrap from GitHub, evals, push to `colab-results` (with retry, parallel jobs), then `runtime.unassign()`, so the
  laptop can close. Colab allows **3 concurrent GPU runtimes** (a 4th was refused), so 3 jobs:
  A = MSMARCO + DBPedia (~17 h), B = FEVER + ClimateFEVER + NQ (~14 h), C = HotpotQA + Touche2020 (~7.5 h).
  Estimate ~340 units. The SciFact/NFCorpus re-check with the new batching failed to start (argparse read
  `-tbcheck` as a flag; fixed) and runs after the jobs.
- **Results, day 1:** job C finished in 7.8 h: **HotpotQA 75.41**, **Touche2020 18.52**. Job B finished
  FEVER + ClimateFEVER in 11.2 h (**FEVER 89.09**, **ClimateFEVER 37.00**; ClimateFEVER took 5 min thanks to the
  cache) and pushed them at 21:59. Then **both A and B stopped at ~11.3 h** (~22:00, within minutes of each other, no
  error pushed): A was still in MSMARCO, B had just started NQ. Two runtimes dying together at the same age points to a
  Colab runtime-lifetime limit (~11–12 h), not to our code. ~267 units used (623 left).
- **Relaunch (23:00):** D = NQ + the SciFact/NFCorpus re-check, E = DBPedia (each well under 11 h). MSMARCO needs
  ~11.5 h on one G4, so it now runs as **3 corpus shards** (`MSMARCO@i/3`, `src/olmo3_embed/shards.py`). Each shard
  searches a contiguous third of the corpus and uploads its per-query top-1000 to the release
  `msmarco-shards-p7-run4-7b`. Then `scripts/merge_shards.py` scores the union with mteb, which equals a single
  full-corpus search: on SciFact (MiniLM, 3 shards) all 149 metrics were identical. S0 started at once; S1 and S2
  start when D and E free their slots (`queue_shards.sh` on the Mac).
- **Night 2:** D = NQ **62.12** (4.9 h), E = DBPedia **48.28** (6.0 h). MSMARCO shards took 3.9 h each (S1 and S2
  started at 03:06 and 04:08, as D and E freed their slots). `merge_shards.py` on the Mac (2 min): **MSMARCO 38.57**
  (dev). D's SciFact/NFCorpus re-check ran but wasn't pushed: the log loop read `model_meta.json` (no scores) and its
  KeyError ended the cell (fixed). Rerun on a G4 (~10 min): **SciFact 79.43, NFCorpus 38.94** vs 79.47 / 38.98 with
  fixed batching, so token-budget batching doesn't move scores. Phase 8 total ~468 units (890 → 422).

**Run #4 BEIR-15 = 54.72** (estimate was ~54.2): **rank 37 of 140** 15-task models in the 2026-10-04 snapshot (between
lightonai/GTE-ModernColBERT-v1 54.75 and manveertamber/cadet-embed-base-v1 54.32). T1 (56.6) missed by 1.88.

| TRECCOVID | NFCorpus | NQ | HotpotQA | FiQA | ArguAna | Touche | CQADup | Quora | DBPedia | SCIDOCS | FEVER | ClimateFEVER | SciFact | MSMARCO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63.10 | 38.98 | 62.12 | 75.41 | 53.39 | 57.82 | 18.52 | 45.76 | 89.66 | 48.28 | 23.65 | 89.09 | 37.00 | 79.47 | 38.57 |

**Learned:** plan every Colab job to finish within ~10 h. Job A lost ~11 h (~100 units) of MSMARCO to the runtime limit;
splitting large corpora up front (or checkpointing embeddings) would have saved it.

**Phase 9: Ship (Oct 14).**
- **HF model:** [`petegordon/olmo3-embed-7b`](https://huggingface.co/petegordon/olmo3-embed-7b), public, Apache-2.0,
  revision `ad1008e`. `scripts/export_hf.py` merges the LoRA (bf16, 13.8 GB) and saves a Sentence Transformers model:
  EOS baked into the tokenizer's post-processor (same ids as `model.py`, truncation included), last-token pooling,
  Normalize. Card: `hf/README.md`.
- **Verification** through the proposed mteb loader (`hf/model_meta.py`, `InstructSentenceTransformerModel`): cosine
  vs our encoder ≥ 0.9998 over all 323 NFCorpus queries and 3,000 docs; SciFact 79.44 (79.47), NFCorpus 38.78 (38.98,
  bf16 batching noise: the 0.15 gate I set flagged it, the embedding check cleared it).
- First export runtime vanished after ~28 min (KEEP set, no unassign; cause unknown, logs lost); the second attempt,
  apparently on the same VM, found the export already on disk. ~6 units for Phase 9 so far (416 left).
- **mteb ModelMeta** drafted (`mteb/models/model_implementations/olmo3_embed_models.py` in a clone): training_datasets =
  MSMARCO, HotpotQA, NQ, FEVER, FiQA2018, NFCorpus, SciFact, QuoraRetrieval, so mteb rates it **46% zero-shot on
  BEIR**. mteb's model-meta tests pass (3,723). Results folder for embeddings-benchmark/results re-labelled to
  `petegordon__olmo3-embed-7b/ad1008e…` (26 task files + model_meta.json); mteb's ResultCache loads it as 15 BEIR tasks,
  avg 54.72.

- **PRs opened (Oct 15):** [mteb#5677](https://github.com/embeddings-benchmark/mteb/pull/5677) (ModelMeta) and
  [results#775](https://github.com/embeddings-benchmark/results/pull/775) (BEIR-15), after both branches were rebased
  on upstream main and re-checked: mteb ruff + model tests 3,731 passed; the results suite against the mteb branch
  6,198 passed. Fixes from the review pass: the results are labelled BEIR, not MTEB(eng, v2) (a different 10-task
  set); `contacts=["petegordon"]` added; the loader check is described as run on the export before upload, and the
  "reference implementation" box is left unchecked with the reproduction numbers. results#775's
  implementation check passes only once mteb#5677 is merged and released. Texts: `docs/ship-pr-drafts.md`.

Also on Oct 11: PLAN.md status section (pre-public checklist, BEIR-15 blocker) and an Apache-2.0 LICENSE.

**Learned:** headless Colab works end to end and needs no tab, but each runtime still costs one human consent click
for Drive. The HF storage plan (`docs/colab-hf-storage-plan.md`) would remove that: an HF token goes up like the GitHub
token, with no consent.

### BEIR test cache to the Mac (Oct 9, ~5 min, laptop, no units)
The Mac needed the laptop's per-task BEIR test JSONs to rebuild the doc-decontam index for the new near-duplicate
check. That's the 39 top-level `data/raw/beir_test/*.json` files, 47.9 MB, which hold each test set's queries plus the
opening of every relevant document. The corpus subfolders and `doc_index_v1.pkl` were skipped.
- Neither planned route was usable. Drive for desktop isn't installed on the laptop. The USB drive Pete attached is a
  Seagate Backup Plus formatted HFS+ (MBR type `0xAF`), which Windows can't mount.
- So the files went as a private release asset instead: `beir-test-cache-v1` / `beir_test_cache.zip` (19.6 MB). A local
  copy is in `data/beir_test_cache/` (gitignored).
- On the Mac: `gh release download beir-test-cache-v1 -p beir_test_cache.zip`, then unzip into `data/beir_test_cache/`.

### Laptop-only data to Drive (Oct 9, ~20 min, laptop + Mac, no units)
For the long term, Drive (`MyDrive/olmo3-embed/data/`) should hold everything that costs time or units to make. Raw
public downloads (`data/raw/`, ~33 GB) stay local.
- **Inventory:** the Mac listed Drive. It had 6.7 GB in `data/`, all Colab-built: the `rescored` parquets for
  trackb_v1, v1, v2, v2b, v3 and v3b, plus MS MARCO and two judge files. `runs/` held another 4.4 GB.
- **Missing from Drive:** only the laptop's unreproducible LLM output and its provenance files.
- **Release `drive-data-v1`** (27 MB zip, 85 MB unzipped, 45 files plus `MANIFEST.sha256`), at Drive-relative paths:
  - `synth/`
  - `variety_v2b/neg_{claude,codex}.jsonl`
  - `trackb_v1/judge/verdicts_pilot_{claude,codex,gemini}.jsonl`
  - the small json reports and logs from `trackb_v1/` and `variety_v1`–`v3`
- **Placement:** the Mac unzips it into Drive. It doesn't overwrite anything, because no file in the zip already
  exists there.
- **Not uploaded:**
  - `trackb_v1/decontam.parquet` (3.76 GB): it can be rebuilt, and Drive has the rescored files training reads.
  - The variety pre-rescore parquets and the v3 pools: these are already release assets.
  - `beir_test_cache/`: release `beir-test-cache-v1`.
- **MS MARCO differs:** the laptop's copy is 339,934,842 bytes (Oct 4 local build), and Drive's is 340,159,918 bytes
  (Colab-built, the one training used). Drive's copy stays as it is.
- **On Drive (Mac, Oct 9 21:40):** all 45 files passed `shasum -c` after download, were copied with `rsync -a
  --ignore-existing` (none already existed, so nothing was skipped), and passed `shasum -c` again on Drive. The
  manifest sits at `data/MANIFEST_drive-data-v1.sha256`. 84.4 MB in total, including 82 MB of `synth/`.
- **Learned:** Git Bash's GNU `tar -a` silently writes a plain tar named `.zip`. Use `C:\Windows\System32\tar.exe` for
  real zips.

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
