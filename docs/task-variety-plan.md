# Task-variety data (Track B+): close the held-out gap (plan)

**Status: APPROVED 2026-10-07 (decisions at the end). In progress: source vetting + document-level decontamination.**

## Why
Run #1 (OLMo-3-7B on Track B v1, `docs/phase4-results.md`):
- **Level with Qwen3-Embedding-4B in-domain** (75.65 vs 76.20).
- **9.7 points behind on the five held-out NanoBEIR tasks** (50.37 vs 60.02).
- **Scaling 1B → 7B moved the held-out average by only +1.0.**

The held-out losses line up with task *shapes* that Track B never shows the model:

| Held-out task | Gap vs Qwen3-4B | Task shape | Closest thing in Track B |
|---|---:|---|---|
| Touche2020 | −15.4 | Controversial question → argumentative passages (pro/con) | none |
| ClimateFEVER | −10.4 | Real-world climate claim → Wikipedia evidence | FEVER (synthetic claims, general topics) |
| DBPedia | −9.8 | Keyword/entity query ("szechwan dish food") → entity description | none |
| SCIDOCS | −8.5 | Paper title → abstracts of papers it cites | none (SciFact is claim → evidence) |
| ArguAna | −4.3 | Argument → its best counter-argument | none |

The goal is to add **diverse, decontaminated training data for these shapes**, and to prove each source on the 1B proxy
by its effect on **avg-OOD-5**, before the 7B sees it.

## Ground rules
- **Never train on a test corpus:**
  - Touche's corpus *is* args.me (~382k arguments), and ArguAna's is idebate.org. Neither, nor anything scraped from
    them, is usable.
  - SCIDOCS is Semantic Scholar papers, and DBPedia-entity is DBpedia abstracts. Citation or entity data built from the
    same sources must be decontaminated by document, not just by query.
- **Decontaminate every new source** with `decontam.py`, extended as follows:
  - Query exact match and near-duplicates against all 39 test sets, as for Track B.
  - **Document-level checks** against the corpora of the task the source targets: drop any training passage that
    exactly or nearly matches a test-corpus document (MinHash over passage shingles). This is new and needed here,
    because shared corpora are the main leak risk for these shapes.
- **Per-row provenance:** every row records its `source`, `generator` and `model`, as PLAN §4.4 requires.
- **License check before use:** listed per source below. A source whose license is unclear gets dropped, not used.

## Candidate sources (to be verified; licenses checked before any download)

### A. Public non-BEIR pairs: broad diversity, cheap (no generation)
From `cfli/bge-e5data` / `cfli/bge-full-data` (already proven loadable) and sentence-transformers sets:
- **StackExchange** title→body and duplicate questions: CQADupStack-like shape.
  - CQADupStack's 12 forums are *test sets*, so those sites' rows are excluded and the rest decontaminated.
- **ELI5, SQuAD, TriviaQA, NLI:** general QA and entailment.
  - RLHN reported these *hurt* BEIR slightly for E5-base, so each gets measured, not assumed.
- **S2ORC / SPECTER citation pairs** (paper → cited paper): **the SCIDOCS shape.**
  - SPECTER and SCIDOCS come from the same group and the same source, so expect overlap. Decontaminate by SCIDOCS
    test titles and abstracts before use.

### B. Synthetic, passage → query (the main lever; PLAN §4.4)
Real non-BEIR passages, with an LLM writing a query in the target *shape*. The four generators are already wired
(`src/olmo3_embed/data/judge/` backends): **OLMo-3.1-32B on Colab** for bulk (fast, about 3.7 prompts/s via vLLM, no
usage cap), with **Claude, Codex and Gemini** for smaller, more diverse slices.

| Target shape | Passage source (not a test corpus) | What the LLM writes |
|---|---|---|
| Argument retrieval (Touche) | Argumentative text that isn't args.me or idebate: candidates are UK Hansard debates (Open Parliament Licence), Wikipedia "Criticism"/"Controversy" sections | A controversial yes/no question the passage argues for or against |
| Counter-argument (ArguAna) | Same sources: pairs of opposing passages | The opposing argument as the query, with the passage as its counter |
| Claim verification (ClimateFEVER) | Wikipedia climate, environment and science articles | A short real-world-style claim the passage supports or refutes |
| Entity search (DBPedia) | Wikipedia lead paragraphs (entity descriptions) | Keyword / list / attribute queries ("female astronauts", "dish from sichuan with peppers") |
| Citation-style (SCIDOCS) | arXiv or PubMed abstracts outside SCIDOCS | A title of a hypothetical paper that would cite this abstract |

- **Hard negatives** come from mining with the Qwen3-Embedding-0.6B rescorer (`rescore.py`, unchanged), not from the
  LLM.
- **Pilot first:** 2k examples per shape per generator. Pick the generator mix by **downstream avg-OOD-5 on the
  proxy**, not by reading samples. The judge pilot showed OLMo-3.1-32B is a poor *judge*; whether it's a good
  *generator* is exactly what the pilot measures.

## Vetted sources (checked live on HF, 2026-10-07)
Each source's license is as declared on its card; there's been no separate legal review. Every row still goes through
`decontam.py` (queries) and `doc_decontam.py` (passages).

| Use | Source | Rows | License (card) | Notes |
|---|---|---:|---|---|
| StackExchange title→body | `flax-sentence-embeddings/stackexchange_title_body_jsonl` | 5.75M | CC BY-SA 4.0 (card text) | One file per site: **drop the 12 CQADupStack sites' files** |
| Citations (SCIDOCS shape) | `sentence-transformers/s2orc` (`title-citation`, `title-abstract`) | 51M / 42M | ODC-By 1.0 (upstream S2ORC) | Title-dedupe against SCIDOCS; sample, don't use all of it |
| Citations | `allenai/scirepeval` `cite_prediction_new` | 6.37M | ODC-BY (GitHub) | **Exclude `scidocs_*` configs** |
| Argument text | `biglam/hansard_speech` | 2.69M speeches | CC-BY-4.0 | Drop procedural speeches |
| Argument text | `common-pile/uk_hansard_filtered` | 47.9k sessions | Open Parliament Licence | Whole sessions; split into speeches |
| Argument pairs (optional) | `ibm-research/argument_quality_ranking_30k` | 30.5k | CC-BY(-SA) 3.0 (card conflict) | Topic → argument; attribute as CC-BY-SA 3.0 |
| Wikipedia (entity leads, climate, controversy sections) | `wikimedia/wikipedia` `20231101.en` | 6.41M | CC-BY-SA 3.0 / GFDL | Lead paragraphs overlap DBPedia-entity *corpus text*: `doc_decontam` drops test positives |
| Abstracts (citation-style queries) | `common-pile/arxiv_abstracts` | 2.54M | CC0 | |
| Entity queries | `facebook/kilt_tasks` (`structured_zeroshot`, `trex`) | 157k / 2.29M | MIT | Entity + relation queries, Wikipedia-linked |

**Avoid:**

| Source | Why |
|---|---|
| `DebateLabKIT/arguments-and-debates` | Contains idebate's Debatabase, ArguAna's test corpus |
| `KShivendu/dbpedia-entities-openai-1M` (and similar copies) | Built from BEIR DBPedia-entity |
| `sentence-transformers/stackexchange-duplicates` | No site column, so the CQADupStack forums can't be removed |
| `allenai/specter` triples | Same origin as SCIDOCS |
| ORCAS | Non-commercial terms |
| Kialo, Reddit ChangeMyView | Unclear terms |
| `flax-sentence-embeddings/stackexchange_*best_voted_answer*` | Card says CC-BY-NC-SA, which conflicts with upstream |

## Proxy experiments (Colab A100, OLMo-2-1B, B3 recipe)
The baseline is **B3** (64k Track B rows, 496 steps): avg-13 62.28, **avg-OOD-5 49.35**. Every run uses the same
compute (64k rows, 496 steps) and adds new tasks to the temperature mix, so any gain is from variety, not more steps.

| Run | Adds | Question |
|---|---|---|
| V1 | A: StackExchange (decontaminated) + S2ORC citations | Do cheap public pairs move held-out? (SCIDOCS especially) |
| V2 | B pilot: synthetic argument + counter-argument (2k each, best generator) | Touche / ArguAna |
| V3 | B pilot: synthetic entity + claim (2k each) | DBPedia / ClimateFEVER |
| V4 | V1 + V2 + V3 combined | Do they stack? |

- **Each run:** about 1 h on an A100 40GB, ~5–6 units.
- **Decision rule:** keep a source if it raises avg-OOD-5 by more than ~1 point without costing avg-in-8 more than ~1.
- **Scale up after:** synthetic generation goes from 2k to ~20–50k per shape for the winners, then **7B run #2**
  (Track B + the winning sources).

## Cost and time
| Item | Where | Estimate |
|---|---|---|
| Source vetting, licenses, decontamination extension | laptop | about a day of work; 0 units |
| Public pairs: ingest + decontam + rescore (~100–200k rows) | laptop + Colab rescore | ~10 units |
| Synthetic pilot: 5 shapes × 2k × (OLMo + 1–2 frontier generators) | Colab vLLM + subscriptions | OLMo ≈ 5 units; frontier: subscription windows |
| Proxy runs V1–V4 | Colab A100 40GB | ~25 units |
| **Subtotal before the 7B** | | **~40 units** |
| 7B run #2 (64k–100k rows) | Colab A100 80GB | ~45–65 units |

That fits the ~255 units left, and keeps room for an eval of run #2.

## Decisions (Pete, 2026-10-07)
1. **Direction approved:** task-variety data proven on the 1B proxy first, then 7B run #2.
2. **Generators:** OLMo-3.1-32B on Colab for bulk, plus **Claude, Codex and Gemini (Antigravity)** for diverse
   slices. All four are in the synthetic pilot, and per-row provenance lets the proxy decide the mix.
3. **Argument text:** Hansard and Wikipedia controversy sections. Kialo and Reddit ChangeMyView are excluded (unclear
   reuse terms).
4. **BEIR-15 lite on run #1: skipped** (~50–60 units). Evaluate run #2 instead.

## Results

### V1: public pairs (p5-v1, 2026-10-07)
B3 recipe and 64k-row budget, with V1's 138,962 rows (StackExchange 78,978 and S2ORCCitation 59,984, after
decontamination) added to the temperature mix. Data: release asset `data-variety-v1`. Negatives mined from each task's
own pool. Cost: about 2 h on A100 40GB (~10 units, including mining 138k docs).

| Task | B3 | p5-v1 | Δ |
|---|---|---|---|
| ArguAna | 53.93 | 56.25 | +2.32 |
| ClimateFEVER | 38.70 | 39.03 | +0.33 |
| DBPedia | 60.41 | 61.50 | +1.09 |
| SCIDOCS | 39.59 | 41.04 | +1.45 |
| Touche2020 | 54.12 | 52.00 | −2.12 |
| **avg-OOD-5** | **49.35** | **49.96** | **+0.61** |
| FEVER | 90.59 | 90.75 | +0.16 |
| FiQA2018 | 57.12 | 56.50 | −0.62 |
| HotpotQA | 84.69 | 83.30 | −1.39 |
| MSMARCO | 55.13 | 53.88 | −1.25 |
| NFCorpus | 35.91 | 35.01 | −0.90 |
| NQ | 65.09 | 66.02 | +0.93 |
| Quora | 95.66 | 94.87 | −0.79 |
| SciFact | 78.70 | 75.99 | −2.71 |
| **avg-in-8** | **70.36** | **69.54** | **−0.82** |
| avg-13 | 62.28 | 62.01 | −0.27 |

**Below the keep bar** (OOD +0.61 vs the +1 threshold; in-domain −0.82). Per-task differences of about 1–2 points are
within Nano noise (~50 queries per task). The SCIDOCS gain from S2ORC citations is the expected direction.
StackExchange added almost no training signal: its batches sat at loss ≈0.005, accuracy 1.0. Own-pool negatives come
from other forums, so they are trivially easy. That budget came out of the in-domain tasks.

**Next:** carry only S2ORCCitation into V4. Drop StackExchange unless its negatives are mined within the same site, and
then only if it earns its place in a combined run.

### V2 and V3: synthetic pilot (p5-v2, p5-v3, 2026-10-07)
Claude + Codex queries (`synth.py`), decontaminated (3 V2 queries were exact Touche2020 test queries; 19 V3 passages
overlapped test positives), release asset `data-variety-synth-v1`. Same recipe, budget and negatives as V1. Both runs
in one Colab session (`notebooks/variety_synth_colab.ipynb`): ~3.4 h on A100 40GB, ~18 units.
- **V2** (7,989 rows): SynthArgument + SynthCounterArgument on Hansard speeches, aimed at Touche2020 / ArguAna.
- **V3** (7,981 rows): SynthEntitySearch + SynthClimateClaim on Wikipedia, aimed at DBPedia / ClimateFEVER.

| Task | B3 | V1 | **V2** | **V3** |
|---|---|---|---|---|
| ArguAna | 53.93 | 56.25 | 53.95 | 54.11 |
| ClimateFEVER | 38.70 | 39.03 | 39.32 | 40.13 |
| DBPedia | 60.41 | 61.50 | 62.29 | 61.78 |
| SCIDOCS | 39.59 | 41.04 | 39.54 | 41.27 |
| Touche2020 | 54.12 | 52.00 | 52.32 | 50.04 |
| **avg-OOD-5** | **49.35** | **49.96** | **49.48** | **49.47** |
| **avg-in-8** | **70.36** | **69.54** | **70.07** | **70.00** |
| avg-13 | 62.28 | 62.01 | 62.15 | 62.10 |

Training logs (every 5th step):

| Task | Share of steps | Batch accuracy, start → end |
|---|---|---|
| SynthArgument | ~5% | 0.04 → ~0.75 |
| SynthCounterArgument | ~8% | 0.07 → ~0.93 |
| SynthClimateClaim | ~6% | ~0.95 throughout |
| SynthEntitySearch | ~3% | ~1.0 throughout |

**Reading it:**
- **None of V1–V3 clears the keep bar.** All three land at +0.1 to +0.6 on avg-OOD-5 and −0.3 to −0.8 on avg-in-8.
- **Run-to-run noise is the same size as the effects.** V1–V3 are small variations of B3, yet in-domain tasks they
  barely target swing by 1–3 points (SciFact 76.0–78.7, NFCorpus 33.0–35.9). Touche2020 is *below* B3 in all three
  runs (50.0–52.3 vs 54.1), so B3's Touche was likely a high draw.
- **Targeted tasks moved slightly in the right direction for V1 and V3,** within noise: SCIDOCS +1.5 from S2ORC
  citations (V1); ClimateFEVER +1.4 and DBPedia +1.4 (V3). V2 didn't move ArguAna or Touche.
- **Negatives decide whether a source teaches anything.** Own-pool mining only gives hard negatives when the pool is
  topically dense. Hansard is (argument batches started at 4% accuracy), but random Wikipedia entities, climate
  paragraphs and StackExchange posts are not: those batches sat at 95–100% accuracy, so they taught little. DBPedia and
  ClimateFEVER retrieve from large Wikipedia corpora, so their synthetic data needs negatives mined from a large
  Wikipedia pool, not from 4k sampled passages.
- **Domain mismatch for V2:** Hansard speeches don't read like the debate-site arguments in Touche (args.me) or ArguAna
  (idebate), even though the model learned the synthetic task.

### Full-size held-out eval + V3b (2026-10-07)
Nano's ~50 queries per task can't resolve +1-point changes, so the saved 1B adapters were re-scored on the held-out
tasks at full size: `eval --suite heldout` = ArguAna (1,406 queries), SCIDOCS (1,000), ClimateFEVER-HardNegatives
(1,000), DBPedia-HardNegatives (400). **heldout-4** is their average. Touche2020 has only 49 queries even in full, so it
stays on Nano. About 12 min per adapter on an A100 40GB (`notebooks/variety_heldout_colab.ipynb`).

The `.v2` versions of the HardNegatives tasks only change the query instruction. We evaluate without instructions, so
the scores are the same.

**V3b:** V3's rows with negatives mined from a Wikipedia pool (`synth.py pool`, `rescore.py --pool`; release
`data-variety-v3-pool`) instead of V3's 4k own pool:
- 500,000 entity leads and 222,778 climate-term paragraphs.
- Articles in the V3 passage sample are excluded.
- Decontaminated against test documents (842 + 60 dropped).

The guard now drops 11.6% (entity) and 5.2% (climate) of candidates, vs 0.5% and 1.1%, so the negatives are much
closer. Entity batches still trained at 96–99% accuracy, though.

| | B3 | V1 | V2 | V3 | V3b |
|---|---|---|---|---|---|
| ArguAna (full) | 52.42 | 54.94 | **55.07** | 53.98 | 52.64 |
| SCIDOCS (full) | 19.54 | **20.72** | 19.22 | 19.52 | 19.79 |
| ClimateFEVER-HN | 32.26 | 32.90 | 32.24 | 32.83 | **33.18** |
| DBPedia-HN | 42.41 | **42.88** | 42.74 | 42.69 | 42.12 |
| **heldout-4 (full size)** | **36.66** | **37.86 (+1.20)** | **37.32 (+0.66)** | **37.25 (+0.59)** | **36.93 (+0.27)** |
| Nano avg-OOD-5 | 49.35 | 49.96 | 49.48 | 49.47 | 50.07 |
| Nano avg-in-8 | 70.36 | 69.54 | 70.07 | 70.00 | 69.51 |

**Reading it:**
- **On the full-size tasks all four variants beat B3, and V1 clears the keep bar (+1.20).** The gain comes from
  ArguAna (+2.5) and SCIDOCS (+1.2), the S2ORC citation target.
- **V2's synthetic counter-arguments do help ArguAna (+2.65 on 1,406 queries),** which Nano couldn't show. Touche
  remains unmeasurable at 49 queries.
- **ArguAna rises 1.5–2.6 with V1, V2 and V3.** Part of the gain may be generic (any non-Track-B data), not
  shape-specific.
- **The large mining pool didn't pay off.** V3b improves ClimateFEVER-HN slightly (+0.9 vs B3, +0.35 vs V3) but not
  DBPedia (−0.3), and loses V3's ArguAna gain. heldout-4 drops to 36.93, below V3. Better negatives alone don't make the
  entity-search data teach DBPedia. The query style, or 4k rows at ~3% of steps, is the limit.
- **In-domain cost:** Nano avg-in-8 is 0.3–0.9 below B3 for every variant. That's within Nano noise, but consistently
  negative: each variant takes ~10–15% of a fixed 64k budget away from the eight in-domain tasks.

**Decision:**
- **Keep V1 (StackExchange + S2ORC) and V2 (argument + counter-argument).** V3's entity/claim data stays a candidate
  for the combined run (small positive).
- **Drop the V3b pool mining** (no gain for its cost).
- **Next, V4:** V1 + V2 + V3 together at the same 64k budget, to see whether the held-out gains add up. Then 7B run #2
  with the kept sources, ideally at a larger budget so the in-domain tasks don't lose their share.

### V4: V1 + V2 + V3 combined (p5-v4, 2026-10-07)
Track B v1 + V1 + V2 + V3 (all own-pool rescored) at B3's recipe and 64k-row budget
(`notebooks/variety_v4_colab.ipynb`). About 1.9 h on an A100 40GB (~10 units) for training and both evals.

| | B3 | V1 | V2 | V3 | **V4** |
|---|---|---|---|---|---|
| ArguAna (full) | 52.42 | 54.94 | 55.07 | 53.98 | **56.00** |
| SCIDOCS (full) | 19.54 | **20.72** | 19.22 | 19.52 | 20.45 |
| ClimateFEVER-HN | 32.26 | **32.90** | 32.24 | 32.83 | 31.84 |
| DBPedia-HN | 42.41 | **42.88** | 42.74 | 42.69 | 42.03 |
| **heldout-4 (full size)** | **36.66** | **37.86** | **37.32** | **37.25** | **37.58 (+0.92)** |
| Nano avg-OOD-5 | 49.35 | 49.96 | 49.48 | 49.47 | **50.39** |
| Nano avg-in-8 | 70.36 | 69.54 | 70.07 | 70.00 | 69.83 |
| Nano avg-13 | 62.28 | 62.01 | 62.15 | 62.10 | **62.35** |

**Reading it:**
- **The gains don't stack at a fixed budget.** V4 is +0.92 on heldout-4, below V1 alone (+1.20). Its gains are the
  best ArguAna yet (+3.6, where V1's and V2's argument-shaped data overlap) and V1's SCIDOCS gain, mostly kept.
- **ClimateFEVER and DBPedia fall below B3.** With four sources sharing the 64k rows, Track B's FEVER/HotpotQA/NQ rows
  get fewer steps, and FEVER drops 3.1 on Nano. Those are the Wikipedia tasks closest to ClimateFEVER/DBPedia.
- **Nano agrees on direction but not on size.** V4 has the best Nano avg-OOD-5 (50.39) and avg-13 (62.35), but only
  the full-size eval separates V1 from V4.

**What carries to 7B run #2:**
- **Sources:** Track B v1 + V1 (StackExchange + S2ORC) + V2 (synthetic argument + counter-argument). V3 is
  dropped: no full-size gain on its targets, and it costs FEVER-style steps.
- **Budget:** more than 64k rows, so the added sources come *on top of* Track B's share instead of out of it. Every
  variant cost avg-in-8 at a fixed 64k, and the 7B's in-domain lead is the asset to keep.

## Experiments A and B (approved 2026-10-07)
Prompted by an outside review of the pipeline. What we took from it, what we didn't, and why:
- **Taken:**
  - Re-test instructions now that the training data has held-out shapes (A).
  - Stance-flipped hard negatives for counter-argument retrieval (B).
  - A round-trip consistency filter for synthetic pairs (B).
- **Not taken:**
  - *Asymmetric instructions:* already in place (queries only).
  - *Causal attention as the ceiling:* Qwen3-Embedding is causal + EOS, and our 7B matches Qwen3-4B in-domain.
  - *MNTP / latent-attention pooling:* new architecture work, and the gap is in the data.
  - *NLI "contradiction = negative" for ClimateFEVER:* ClimateFEVER counts refuting evidence as relevant.

Both are 1B proxy runs at B3's recipe and 64k rows, compared with **p5-v2** (Track B + V2, no instructions:
heldout-4 37.32, Nano avg-in-8 70.07), which has the same data and differs only in what each experiment changes.
- **A, `p5-v2i`** (`notebooks/variety_a_instr_colab.ipynb`): per-task query instructions, 20% instruction dropout.
  - MTEB prompts for Track B tasks; our own for the new tasks (`train.py` `EXTRA_INSTRUCTIONS`).
  - Dropout is decided by a hash of the query (`instruction_dropout`).
  - Phase 3's B2 (−4.6) trained on only 8 search-like tasks.
  - The held-out suite now uses the HardNegatives `.v2` tasks, which carry query prompts; the data is identical to v1.
- **B, `p5-v2b`**: V2 after a round-trip filter, plus stance-flipped negatives.
  - **Filter** (`synth.py filter`, Qwen3-0.6B, own passage in the top 10 among the task's passages):
    - Kept 6,382 of 7,989 rows.
    - SynthCounterArgument: 3,742 / 3,992 kept, median rank 1, top-1 71%.
    - SynthArgument: 2,640 / 3,997 kept, median rank 4. Generic debate questions that many speeches answer are dropped.
  - **Stance-flipped negatives** (`synth.py negatives`): for up to 1,600 counter-argument rows (Claude part 0, Codex
    part 1), a new passage on the same topic with the same key terms and parliamentary register that *agrees* with
    the argument. The prompt has the generator state the argument's premise and its strategy before writing.
  - **Attach** (`synth.py attach`): the negatives become a `gen_negatives` column, are checked against test documents,
    and `rescore.py` puts them first, exempt from the 95% guard.
  - **Smoke test:** the Qwen3-0.6B teacher scores the agreeing passage *above* the real rebuttal in 31 of 32 rows. That
    is the confusion B targets, and why the guard can't be applied to these negatives.

### Result A: instructions + 20% dropout (`p5-v2i`, 2026-10-07)
Pushed 23:37 (Windows instance's browser runtime, ~1.85 h, ~10 units). Compared with p5-v2 (same data, no instructions):

| | ArguAna | SCIDOCS | ClimateFEVER-HN | DBPedia-HN | **heldout-4** | Nano avg-in-8 | Nano avg-OOD-5 |
|---|---|---|---|---|---|---|---|
| p5-v2 | 55.07 | 19.22 | 32.24 | 42.74 | 37.32 | 70.07 | 49.48 |
| **p5-v2i** | 56.11 | 19.73 | 31.14 | 44.68 | **37.92** | 70.70 | 50.88 |
| Δ | +1.04 | +0.51 | −1.10 | +1.94 | **+0.60** | +0.63 | +1.40 |

- **Below the bar** (heldout-4 +0.60, under ~1), with no in-domain loss: avg-in-8 +0.63, best avg-13 so far (63.08).
- **Instructions no longer hurt.** B2 lost 4.6 points with instructions on 8 search-like tasks. With varied task shapes
  and 20% dropout they're neutral to slightly positive. Nano MSMARCO +5.0 and Touche +2.7 are single-task noise
  territory.
- The held-out runs differ in one more way: p5-v2i is scored on the `.v2` HardNegatives tasks with their query prompts
  (same documents and qrels). So ClimateFEVER −1.10 / DBPedia +1.94 mix the instruction effect with the prompt.
- **Verdict:** not adopted on its own merit. It costs nothing at inference, so it can ride along in 7B run #2 if B
  doesn't win outright; at 7B scale the gain may differ either way.
