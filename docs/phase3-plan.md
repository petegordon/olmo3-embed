# Phase 3: training data (Track B first)

**Status: APPROVED 2026-10-05 (decisions at the end). Nothing built yet.**

## Goal
Replace Phase 2's 16k MS MARCO rows with a decontaminated, multi-task **Track B** mix (BEIR train splits, as E5-Mistral,
NV-Embed, bge-en-icl and Qwen3 use) with clean hard negatives. Then prove on the 1B proxy that it beats the Phase 2
best (a8, 57.51 NanoBEIR avg-12) before spending 7B hours. Track A (zero-shot-clean) and the synthetic-data pilot come
after Track B works (§7).

**Recipe carried in from Phase 2:** causal attention, EOS last-token pooling, τ=0.02, k=1–3 hard negatives.
Instructions get re-tested here (§5), because the multi-task mix finally gives them something to do.

## 1. What's available (verified 2026-10-05 on HF; details and URLs in the research notes below)

| Source | What it is | Use |
|---|---|---|
| **`rlhn/rlhn-680K`** (CC-BY-SA-4.0, 648,766 rows) | BGE training data for MS MARCO, NQ, HotpotQA, FEVER, FiQA, ArguAna, SciDocsRR, with **false negatives relabeled as positives** by a GPT-4o-mini → GPT-4o cascade (Thakur et al. 2025, arXiv 2505.16967; up to 56% of MS MARCO pairs had false negatives; +2–3 nDCG@10 on BEIR) | **Backbone** |
| `rlhn/default-680K` (680,223 rows) | Same rows *before* relabeling | Ablation: is the cleaning worth it on our model? |
| `cfli/bge-full-data` (no license declared, ~57 GB) | Full bge-en-icl data: also Quora 60k, NLI 276k, ELI5, SQuAD, TriviaQA, StackOverflowDup, plus classification and clustering. Includes `pos_scores`/`neg_scores` | Quora source; later mixes |
| BEIR native train qrels (SciFact, NFCorpus) | Small official train splits; tiny corpora (5k and 3.6k docs) | Mine our own negatives locally |
| `KaLM-Embedding/...-finetuning-data` (MIT, 1.05M) | Includes TREC-COVID, DBPedia and NFCorpus rows **generated from the BEIR test corpora** | **Avoid**: contamination |
| `nomic-ai/nomic-embed-supervised-data`, `lightonai/nv-embed-supervised` | No license, third-party uploads | Not needed |

## 2. Track B v1 composition (proposal)

| Task | Source | Rows (cap) | Notes |
|---|---|---:|---|
| MS MARCO passage | RLHN-680K | 150k (cap) | Biggest source; cap so it doesn't dominate the mix |
| NQ | RLHN-680K | all (~58k) | |
| HotpotQA | RLHN-680K | all (~84k) | Often 2 positives per row; sample 1 per epoch |
| FEVER | RLHN-680K | all (~29k) | |
| FiQA-2018 | RLHN-680K | all (~5.5k) | |
| Quora duplicates | bge-full-data `quora` | ~60k | BEIR has no Quora train split, so strict decontamination against the test set is required (§3) |
| SciFact | BEIR train qrels | ~800 | Negatives mined locally (§4) |
| NFCorpus | BEIR train qrels | ~2.6k queries | Negatives mined locally (§4) |
| **Total** | | **≈ 390k → ~300k after caps/decontam** | Matches the PLAN §5 budget for 7B run #1 |

Row counts are the `cfli/bge-full-data` subset sizes. RLHN-680K's per-subset counts aren't published, so step 1 of
§8 measures them, and the caps get adjusted then.

**Excluded from v1, on purpose:**
- **ArguAna and SciDocsRR** (both in RLHN and BGE): BEIR has *no* train split for either, so these rows overlap the
  test corpora. bge-en-icl and NV-Embed train on ArguAna anyway, but we skip it for the honesty story. It can be a
  later ablation.
- **NLI, ELI5, SQuAD, TriviaQA:** RLHN found that pruning these *raised* BEIR (+1.0 for E5-base). Revisit in Track A.

## 3. Decontamination (gate: zero test-query overlap)
Build a test index from all 15 BEIR test sets plus NanoBEIR and CQADupStack: queries, and qrel-positive doc ids and
texts.

1. **Queries:** drop any training row whose normalized query (NFKC, lowercase, collapsed whitespace and punctuation)
   exactly matches a test query, or is a near-duplicate (MinHash over word 5-grams, Jaccard ≥ 0.8). Run this across
   all tasks, not just the matching one (FEVER claims vs Climate-FEVER, Quora vs CQADupStack).
2. **Positive documents:** for each task, count training positives that are also test-qrel positives.
   - Track B: shared corpora are standard practice (NQ, HotpotQA and FEVER all draw on Wikipedia), so **report** this
     overlap but don't drop the rows, except for Quora, where a shared duplicate pair *is* a test example, so drop it.
   - Track A: drop these rows.
3. **Output:** `docs/data-card.md` gets a per-source table: rows in, dropped by each rule, and rows out. The
   contamination script stays in the repo so anyone can reproduce the numbers.

## 4. Hard negatives (decided 2026-10-05: embedder for retrieval, LLMs as judges)
The "teacher" does two separate jobs:
1. **Retrieval:** ranking a corpus to find candidate negatives. Only an embedder can do this at scale.
2. **Judging:** deciding whether a candidate is really irrelevant (false-negative filtering). LLMs do this far
   better than embedding scores, which is RLHN's whole result.

**4a. Embedder (retrieval + cheap guard), laptop GPU.** Qwen3-Embedding-0.6B (Apache-2.0, 65.98 NanoBEIR on our
harness, fits 12 GB in bf16).
- Re-score every row's candidates and drop any negative above **95% of the positive's score** (NV-Retriever's
  TopK-PercPos). Keep the hardest remaining 7; training samples k from those.
- **SciFact and NFCorpus:** full mining. Embed the whole corpus (under 10k docs, minutes), take the top 100, and keep
  ranks 10–100 before the guard.
- **Cost:** about 300k queries × ~25 candidates, roughly 2–4M unique passages after dedup. A few hours overnight,
  stored as fp16 shards. Candidate lists only, so no FAISS index is needed.

**4b. LLM judge cascade, on the rows nobody has LLM-judged.** RLHN rows (~80% of the mix) were already judged by
GPT-4o-mini → GPT-4o, so re-judging them would mostly repeat that work. The cascade covers **Quora, SciFact and
NFCorpus**, on the negatives most likely to be false: the top 3 per query with embedder score between 80% and 95% of
the positive. That's about 190k (query, positive, candidate) judgments.

| Stage | Judge | Where | Covers |
|---|---|---|---|
| 1. Bulk | **`allenai/Olmo-3.1-32B-Instruct`** (Apache-2.0, latest OLMo instruct) | Colab H100 80GB or A100 80GB, vLLM, bf16 | **All** ~190k candidates. Fully-open labels for the bulk |
| 2. Verify | Claude Opus 5.5 (Max) | `claude -p` | Every pair OLMo flags as relevant, plus a 2% random sample of OLMo's "irrelevant" calls (calibration) |
| 3. Tie-break | Codex (Plus) and Gemini (AI Pro) | `codex exec`, `agy -p` | Pairs where OLMo and Claude disagree |

- **Decision rule:** a candidate counts as a false negative if 2 of the judges who saw it agree (with Codex and
  Gemini voting only on disagreements). False negatives are dropped as negatives. On SciFact and NFCorpus, where
  queries have few labeled positives, they're also added as extra positives (RLHN's "relabel" variant).
- **Batching:** about 25 pairs per call. On average, a pair is a short query plus two ~150-token passages.
  - The OLMo bulk pass is roughly 35M input tokens. That's a few H100-hours with vLLM prefix caching (the
    instructions and query repeat across a call's pairs).
  - The frontier calls depend on OLMo's flag rate. At 10%, that's ~19k pairs ≈ 800 `claude -p` calls, which fits
    comfortably in the Max windows.
- **Provenance:** every label records the judge, the model id and version, the prompt hash and the raw verdict. The
  data card reports per-judge agreement.
- **Pilot first:** 500 Quora queries (~1.5k pairs) judged by **all four** models.
  - Measure pairwise agreement, the false-negative rate, and whether OLMo-3.1-32B tracks the frontier judges well
    enough to be the bulk judge.
  - If OLMo agrees poorly (under ~0.8 with Claude), swap the roles: Claude becomes the bulk judge on a smaller slice,
    and OLMo becomes a secondary vote.
  - If the false-negative rate is tiny, the embedder guard alone is enough, and stages 2–3 are skipped.
- **Colab cost:** an H100 burn rate is unpublished, so measure it in the pilot. On an A100 80GB (~7.5 CU/h), 4 h ≈
  30 CU ≈ $3. vLLM gets **its own notebook and runtime**, since its torch pin can clash with the training stack.
- **Fallback:** `Olmo-3-7B-Instruct` fits an A100 40GB, if the 32B is too slow or no 80GB GPU is available.

## 5. Training-code changes (small, testable)
1. **Schema:** `query, positives[], negatives[] (ranked), task, instruction, source`. Phase 2 parquet still loads
   (`task=msmarco`).
2. **Task-homogeneous batches** (PLAN §4.5): each step's 128 queries come from one task, so in-batch negatives are
   meaningful. Tasks are sampled with temperature T over dataset sizes (default T=0.5; caps from §2).
3. **In-batch false-negative masking:** if another row's document text equals a query's own positive (common in
   Quora, HotpotQA and FEVER, whose corpora are shared), mask that logit.
4. **Instructions:** `instructions: none | fixed | per_task`, with `per_task` using the MTEB task instruction for the
   matching BEIR task, so train and eval match. This is the explanation Phase 2 suggested for the no-instruction win.
5. **Tests:** batching is deterministic and resumable; masking is checked against a hand-built batch; the GradCache
   gradient test still passes.

## 6. Proxy validation (Colab A100, OLMo-2-1B, a8 recipe)

**Eval split changes:** Track B makes most NanoBEIR tasks in-domain. Report three numbers:

| Metric | Tasks |
|---|---|
| **avg-13** | All NanoBEIR tasks |
| **avg-in-8** | Tasks with training data: MSMARCO, NQ, HotpotQA, FEVER, FiQA, SciFact, NFCorpus, Quora |
| **avg-OOD-5** | Held-out tasks: ArguAna, ClimateFEVER, DBPedia, SCIDOCS, Touche2020 |

ClimateFEVER is only partly held out because FEVER is related, so it's also reported separately.

| Run | Data | Steps | Question | ≈ A100 time |
|---|---|---|---|---|
| B1 | Track B v1, 16k rows | 125 | Mix vs MS MARCO-only, same compute | 26 min |
| B2 | B1 + `per_task` instructions | 125 | Do instructions help once they match eval? | 26 min |
| B3 | Track B v1, 64k rows | 500 | Does more data keep paying on the proxy? | ~1.5 h |
| B4 | B1 with `rlhn/default` (uncleaned) | 125 | Is RLHN's relabeling worth it for us? | 26 min |
| B5 | B1 without the §4b LLM judging (embedder guard only) | 125 | Is the judge cascade worth it on Quora, SciFact and NFCorpus? | 26 min |

Total is about 3.5 h ≈ 20 CU ≈ $2. Same single-seed caveats as Phase 2.

**Gate to Phase 4:**
- Data card written, with zero BEIR test-query overlap verified.
- The best Track B run beats a8 on avg-13.
- No regression on avg-OOD-5 vs a8 (a8's avg-OOD-5 gets recomputed from its existing results).

## 7. Later in Phase 3 (after the gate)
- **Track A (zero-shot-clean):** drop every BEIR-derived source. Use non-BEIR pairs (StackExchange, S2ORC, PAQ, ELI5,
  NLI from `cfli/bge-full-data` / `bge-e5data`) plus synthetic data.
- **Synthetic pilot (PLAN §4.4):** passage → query generation, with 2k examples per backend. **Olmo-3.1-32B-Instruct
  on Colab joins Claude, Codex and Gemini as a fourth backend** (no usage windows, fully open, the same vLLM notebook
  as §4b). Pick the mix by downstream NanoBEIR avg-OOD-5, not by reading samples. Because OLMo has no usage cap,
  it's the natural source for scaling up if its pilot score is competitive.
- **Dolma 3 contamination report** for the data card. This is heavy; defer to Phase 6 unless an existing index
  (infini-gram / OLMoTrace) makes it cheap.

## 8. Order of work
1. `data/build_trackb.py`: ingest RLHN, bge `quora`, and BEIR SciFact/NFCorpus train → unified parquet (laptop, CPU).
2. `data/decontam.py` + test index → drops + data card numbers.
3. `data/rescore.py`: Qwen3-0.6B re-score + 95% guard; mine SciFact/NFCorpus (laptop GPU, overnight).
4. `data/judge/`: one shared prompt and JSON verdict schema, with one adapter per backend (`vllm` for OLMo,
   `claude -p`, `codex exec`, `agy -p`), plus resumable sharded outputs. Also
   `notebooks/phase3_olmo_judge_colab.ipynb`. Run the 500-query pilot, then the cascade (§4b).
5. `train.py` changes + tests (§5).
6. Proxy runs B1–B5 on Colab → `docs/phase3-results.md`.

Steps 1–3 and 5 don't depend on the judges, so the judge work (step 4) can run in parallel.

## Decisions (Pete, 2026-10-05)
1. **Exclude ArguAna and SciDocsRR** from Track B v1 (no BEIR train split exists for either).
2. **Include Quora**, with strict decontamination; report the overlap.
3. **Qwen3-Embedding-0.6B** for retrieval and the 95% guard, disclosed in the data card.
4. **RLHN's CC-BY-SA-4.0 is fine.** A processed mix we publish has to be CC-BY-SA too. The terms of the frontier
   models' outputs (judge labels, synthetic data) were also reviewed and judged fine.
5. **LLM judges:** **Olmo-3.1-32B-Instruct on Colab** is the bulk judge and a synthetic-data backend. Claude Opus 5.5
   verifies, and Codex and Gemini break ties (§4b).

## Research notes (2026-10-05)
- **BEIR train splits exist for:** MS MARCO, NQ, HotpotQA, FiQA, FEVER, SciFact, NFCorpus (BEIR README; `mteb/nq` only
  ships test qrels).
  - Dev/test only: Quora, DBPedia.
  - Test only: ArguAna, Touche, SCIDOCS, Climate-FEVER, TREC-COVID, CQADupStack.
- **What the papers trained on:**
  - E5-Mistral: 13 public sets plus synthetic data.
  - NV-Embed-v2: MS MARCO, HotpotQA, NQ, PAQ, StackExchange, NLI, SQuAD, ArguAna, BioASQ, FiQA, FEVER, HoVer,
    SciFact, NFCorpus and more, with E5-Mistral-mined negatives and TopK-PercPos at 95%.
  - bge-en-icl: the E5 data plus ArguAna, FiQA, SciDocsRR and StackOverflowDup.
  - Qwen3-Embedding: MS MARCO, NQ, HotpotQA, NLI and multilingual sets, plus ~12M synthetic pairs.
- **E5 public mix:** not on HF. It's on Google Drive (3.6 GB) via `jakespringer/echo-embeddings`; `cfli/bge-e5data` is
  a 2.09M-row copy in BGE format.
- **Sources with no license declared:** bge-full-data, nomic, lightonai and most sentence-transformers sets. Prefer
  RLHN (CC-BY-SA) and BEIR native splits for anything we redistribute.
