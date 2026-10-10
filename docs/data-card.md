# Data card: olmo3-embed training data

**Status: Track B v1 (2026-10-05); task-variety V1, V2 and the Phase 6 PubMedQA slice added 2026-10-09.** Ingest and decontamination are done. Hard-negative rescoring and
the LLM judge pass are still to come (`docs/phase3-plan.md` §4); this card gets updated after each step.

## Track B v1: sources
Built by `python -m olmo3_embed.data.build_trackb` and decontaminated by `python -m olmo3_embed.data.decontam --docs`.

| Task | Source | License | Rows in | Rows out | Positives/row | Negatives/row (source) |
|---|---|---|---:|---:|---:|---:|
| MS MARCO | `rlhn/rlhn-680K` `msmarco_passage`, seeded sample of 150k from 455,748 | CC-BY-SA-4.0 | 150,000 | 149,949 | 3.0 | 23.1 |
| HotpotQA | `rlhn/rlhn-680K` | CC-BY-SA-4.0 | 84,390 | 84,387 | 2.1 | 19.9 |
| NQ | `rlhn/rlhn-680K` | CC-BY-SA-4.0 | 57,535 | 57,520 | 1.8 | 97.7 |
| Quora duplicates | `cfli/bge-full-data` `quora` | none declared | 60,201 | **44,702** | 1.0 | 29.2 |
| FEVER | `rlhn/rlhn-680K` | CC-BY-SA-4.0 | 28,327 | 28,325 | 1.4 | 19.8 |
| FiQA-2018 | `rlhn/rlhn-680K` | CC-BY-SA-4.0 | 5,496 | 5,495 | 3.2 | 14.3 |
| NFCorpus | `mteb/nfcorpus` train qrels (highest relevance grade) | BEIR | 2,590 | 2,582 | 42.7 | mined later |
| SciFact | `mteb/scifact` train qrels | BEIR | 809 | 802 | 1.1 | mined later |
| **Total** | | | **389,348** | **373,762** | | |

- **RLHN** (Thakur et al. 2025) is the BGE training data with false negatives relabeled as positives by a GPT-4o-mini →
  GPT-4o cascade. That's why MS MARCO averages 3.0 positives per row.
- **Removed before decontamination:**
  - RLHN's ArguAna (4,065) and SciDocsRR (12,654) subsets, because BEIR has no train split for either, so they
    overlap the test sets.
  - 1,105 duplicate queries.
  - Empty passages (FiQA's corpus has empty documents).

## Decontamination
**Test sets:** 72,055 queries from 39 mteb tasks, using the same dataset paths and revisions the eval harness scores
on.
- The 14 BEIR tasks, the 12 CQADupStack subforums and the 13 NanoBEIR tasks.
- Each task's eval split plus every other non-train split; for example, MS MARCO dev + test, and HotpotQA and FEVER
  dev + test.
- The full list is in `decontam_report.json`.

| Rule | What it catches | Action | Rows |
|---|---|---|---:|
| Query, exact | Normalized (NFKC, lowercase, punctuation stripped) train query equals any test query, in *any* task | drop | 5,783 |
| Query, near-duplicate | MinHash LSH over word 3-grams, confirmed Jaccard ≥ 0.8 | drop | 129 |
| Quora pair | Quora row whose query or positive equals a Quora/NanoQuora test query or test-positive document: a test duplicate pair | drop | 9,674 |
| **Total dropped** | | | **15,586** |

- **Quora is the leak, as expected:** 15,499 of 60,201 rows (26%) collide with the Quora test set. Every other
  source loses only 87 rows combined. These are mostly MS MARCO and NQ queries that repeat their own dev/test sets,
  plus a few across tasks (for example "what is bitcoin ??" in MS MARCO vs "What is bitcoin?" in Quora test).
- **Near-duplicate precision:** a spot check found mostly true copies ("…filmed at" vs "…filmed"). It also found a few
  template collisions that aren't real leaks ("a sentence using the word *discretion*" vs "*myriad*"). Those still get
  dropped: about a hundred rows, an acceptable cost for a contamination gate.
- **Gate status: zero train queries match a test query** (exact or near-duplicate) after decontamination.

### Positive-document overlap (reported, not dropped, for Track B)
Rows whose positive passage is also a test-qrel positive document:

| Train task | Overlapping test task: rows |
|---|---|
| HotpotQA | HotpotQA 28,256 · FEVER 184 · DBPedia 154 · NQ 110 · ClimateFEVER 12 · MSMARCO 4 |
| FEVER | FEVER 3,442 · ClimateFEVER 1,438 · HotpotQA 232 · DBPedia 24 · NQ 1 |
| NFCorpus | NFCorpus 2,555 (of 2,582) |
| FiQA2018 | FiQA2018 406 · NanoFiQA2018 24 |
| SciFact | SciFact 265 · NFCorpus 1 |
| MSMARCO | MSMARCO 155 · DBPedia 25 · HotpotQA 22 · NQ 9 · FEVER 3 · NanoMSMARCO 1 |

**Why these rows stay:** BEIR's train and test splits share one corpus per dataset. Wikipedia serves NQ, HotpotQA,
FEVER, ClimateFEVER and DBPedia; PubMed serves NFCorpus. So a test-relevant passage also being a train positive *for a
different query* is how every Track B model (E5, NV-Embed, bge-en-icl, Qwen3) trains.

**What it means for results:** read in-domain Nano scores for HotpotQA, FEVER and NFCorpus with this in mind, and lean
on the held-out avg-OOD-5 (ArguAna, ClimateFEVER, DBPedia, SCIDOCS, Touche2020; plan §6). FEVER → ClimateFEVER (1,438
rows) is why ClimateFEVER counts as only partly held out. **Track A will drop all of these rows.**

## Hard-negative rescoring (2026-10-06)
`python -m olmo3_embed.data.rescore` on a Colab A100 80GB (`notebooks/phase3_rescore_colab.ipynb`).
- **Teacher:** Qwen/Qwen3-Embedding-0.6B, bf16, cosine similarity. Queries use Qwen3's instruction format with each
  task's MTEB instruction; documents get no instruction.
- **Candidates:** the upstream negatives, capped at 30 per row in source rank order.
- **SciFact and NFCorpus:** mined from the full corpus (top 100, ranks 10–39 kept).
- **Scale:** 4,863,468 unique passages and 373,762 queries, embedded in 3 h 05 min.
- **Guard:** drop a candidate scoring above 95% of the best positive's score (NV-Retriever's TopK-PercPos).
  Survivors are stored hardest-first with their scores.

| Task | Rows | Candidates/row | Guarded (dropped) | Judge band/row (80–95% of positive) | Rows with <3 negatives left | Mean positive score |
|---|---:|---:|---:|---:|---:|---:|
| MS MARCO | 149,949 | 23.1 | 8.0% | 5.45 | 443 | 0.73 |
| HotpotQA | 84,387 | 19.9 | 1.7% | 1.34 | 113 | 0.69 |
| NQ | 57,520 | 30.0 | 12.7% | 8.67 | 750 | 0.68 |
| Quora | 44,702 | 29.2 | 1.6% | 5.33 | 4 | 0.85 |
| FEVER | 28,325 | 19.8 | 0.2% | 0.36 | 3 | 0.78 |
| FiQA-2018 | 5,495 | 14.3 | 13.6% | 3.17 | 167 | 0.64 |
| NFCorpus | 2,582 | 29.9 | 27.4% | 7.61 | 603 | 0.53 |
| SciFact | 802 | 30.0 | 15.0% | 7.13 | 97 | 0.66 |

- **The guard barely touches RLHN's LLM-cleaned sets** (FEVER 0.2%, HotpotQA 1.7%, which matches upstream cleaning).
  It bites hardest where nobody has cleaned the data or where relevance is graded and broad: NFCorpus 27%, SciFact
  15%, FiQA 14%, NQ 13%.
- **NFCorpus is the weak spot:**
  - 603 of 2,582 rows keep fewer than 3 negatives, so k=3 training skips them.
  - Its mean positive score (0.53) is the lowest, because the highest-relevance grade covers many loosely related
    documents.
- **Judge band size**, with top 3 per query: about 134k Quora, 7.7k NFCorpus and 2.4k SciFact pairs, so **~144k
  pairs** for the §4b cascade. That's below the plan's ~190k estimate.
- **Output:** `MyDrive/olmo3-embed/data/trackb_v1/rescored.parquet` (2.5 GB). The stats are on `colab-results` in
  `results/data/trackb_v1/rescored_stats.json`.

## LLM judge pilot (2026-10-06)
**Setup:** `python -m olmo3_embed.data.judge.run`, run on 500 random Quora queries.
- **Pairs:** every candidate scoring 80–95% of the positive, 2,650 pairs in all (366 queries had at least one).
- **Prompt:** one shared prompt for all judges, with the query's known positive as an example of "relevant".
- **Raw verdicts:** on the `colab-results` branch in `results/judge/`.

| Judge | Where | Judged | Flags as false negative | Kappa vs Claude |
|---|---|---:|---:|---:|
| Claude Opus 5.5 (Max) | laptop, `claude -p` | 2,650 | 50 (1.9%) | — |
| Codex (ChatGPT Plus) | laptop, `codex exec` | 2,650 | 48 (1.8%) | **0.65** |
| OLMo-3.1-32B-Instruct | Colab A100 80GB, vLLM | 2,645 | 94 (3.6%) | 0.15 |
| Gemini via Antigravity (AI Pro) | laptop, `agy -p` | 933 (partial) | 1 (0.1%) | 0.14 |

**Speed:**
- OLMo judged all 369 prompts in 1 min 39 s.
- The subscription CLIs manage about 3.7 prompts/min each (Gemini about 2.5/min, with batches of 10).

**Findings:**
- **Real false negatives are rare.** Claude and Codex agree on just 32 pairs (1.2%), touching 28 of 366 queries.
- **OLMo-3.1-32B isn't a usable bulk judge here.** It flags topically related but different questions ("How can I
  sex for long time?" vs "What is the average duration of sex?"), finds only 8 of the 32 agreed false negatives, and
  only 8 of its 94 flags are right.
- **Gemini (the Antigravity default model) almost never flags anything.**
- **Embedding scores can't isolate the false negatives.** Lowering the guard from 0.95 to 0.90 drops 14.7% of band
  candidates but catches only 6 of the 32; at 0.85 it drops 46.5% to catch 20.

**Consequence for Track B v1:** see the plan's progress notes (§4b decision).

## Task-variety sources (Phase 5, 2026-10-07)
Built by `python -m olmo3_embed.data.build_variety` (V1) and `synth.py` (V2); same schema as Track B. Queries are
checked by `decontam.py` and passages by `doc_decontam.py`. Negatives are mined by `rescore.py` (Qwen3-Embedding-0.6B)
within each task's own pool. Plan and results: `docs/task-variety-plan.md`.

| Source | Task | License | Rows in | Query drops | Passage drops | Rows out |
|---|---|---|---:|---:|---:|---:|
| `flax-sentence-embeddings/stackexchange_title_body_jsonl`, 80 sites × 1,000 (12 CQADupStack forums excluded) | StackExchange | CC BY-SA 4.0 | 80,000 | 47: 44 FiQA, 3 Quora | 19 (CQADupStack docs) + 956 (rest of money.stackexchange) | 78,978 |
| `sentence-transformers/s2orc` title-citation pairs (SCIDOCS titles removed at ingest: 320) | S2ORCCitation | ODC-By 1.0 | 60,000 | 0 | 16 (TREC-COVID docs) | 59,984 |
| Hansard speeches + Claude/Codex questions | SynthArgument | CC-BY-4.0 / Open Parliament Licence | 4,000 | 3 (Touche2020 queries) | 0 | 3,997 |
| Hansard speeches + Claude/Codex counter-arguments | SynthCounterArgument | as above | 3,992 | 0 | 0 | 3,992 |

- **What leaked:** money.stackexchange questions are FiQA-2018's source (44 exact test queries), so the site's other
  956 rows were removed as well, and `build_variety.py` now excludes it at ingest. 16 S2ORC citation titles are CORD-19 papers in TREC-COVID's corpus. 3 LLM-written debate
  questions matched Touche2020 test queries word for word.
- **Used in training:** run #2 (7B) used Track B v1 + V1 + V2 at 96k rows. V2b (stance negatives), V3 (entity and
  climate) and V3b were tested on the 1B proxy and not adopted.

## Phase 6: PubMedQA biomedical slice (2026-10-09)
- **Source:** `qiaojin/PubMedQA` `pqa_artificial` (MIT; 211,269 rows). 30,000 sampled (seed 0), one row per PubMed
  id. Query: a research question generated from the paper's title. Positive: the paper's full abstract, the context
  paragraphs plus the conclusion (median 1,633 characters). Built by `build_variety.py --sources pubmedqa`.
- **Target:** TREC-COVID-shaped retrieval (question → biomedical abstract), the largest gap in run #2's 8-task BEIR
  subset (65.9 vs 80.3 for LLM2Vec-Llama-3-8B).
- **Zero-shot protocol for TREC-COVID** (decided 2026-10-09): no TREC-COVID queries or qrels, and no document overlap
  with its 171,332 CORD-19 documents, nor with NFCorpus or SciFact, which are PubMed-derived too.

| Check | Rows dropped |
|---|---:|
| Query exact / near-duplicate vs 72,055 test queries (39 tasks) | 0 |
| `doc_prefix` (TREC-COVID) | 2 |
| `pos_near` (NanoNFCorpus positives) | 3 |
| **`doc_contain`** (≥5% of word 8-grams in a full corpus): TREC-COVID 159, NFCorpus 15, SciFact 12 | **186** |
| **Rows out** | **29,809** |

- **Independent re-check:** after these rules, 0 rows have ≥5% of their 8-grams in any of the three corpora.
- **Release:** `data-variety-bio-v1` (`variety_bio_clean.parquet` and its reports). Negatives are mined on Colab from
  the PubMedQA abstract pool (`data/variety_bio/rescored.parquet` on Drive).
- **Status:** on trial in the 1B proxy `p6-bio`, compared with `p5-v2i`. 7B run #3 uses it only if the proxy gate
  passes.

## Document-level decontamination rules (`doc_decontam.py`, index v2)
| Rule | Matches | Against |
|---|---|---|
| `doc_exact` / `doc_prefix` | Same normalized text, or the same first 200 normalized chars | Every document of the full test corpora: Touche-2020, ArguAna, SCIDOCS, CQADupStack (12), Quora, TREC-COVID, FiQA, SciFact, NFCorpus. For TREC-COVID, NFCorpus and SciFact also the title alone and the abstract alone |
| `pos_near` | MinHash over word 3-grams of the first 300 chars, Jaccard ≥ 0.8 | Test-qrel positive documents of all 39 test sets (covers the Wikipedia-based tasks) |
| `doc_near` | Same MinHash rule | Every document of TREC-COVID, NFCorpus and SciFact |
| `doc_contain` | ≥5% of the passage's word 8-grams occur anywhere in the corpus | Every document of TREC-COVID, NFCorpus and SciFact |

- **Why `doc_contain` exists:** CORD-19 stores "title + section label + abstract" while PubMed sources carry the bare
  abstract. The text is the same but offset, so exact, prefix and first-300-character MinHash matching all miss it.
  On PubMedQA, the first three rules caught 5 rows. `doc_contain` caught 186 more, 139 of them with ≥50% overlap.
- **Not yet applied retroactively:** V1 and V2 were checked with the first two rules only (index v1). See "Still to add".

## Still to add
- LLM judge pass on Quora, SciFact and NFCorpus: OLMo-3.1-32B-Instruct for the bulk, Claude to verify, Codex and Gemini
  to break ties. Per-judge agreement goes here.
- Final per-task row counts and the sampling temperature used in training.
- A Dolma 3 contamination check (Phase 6).
- Re-check V1 and V2 with index v2 (`doc_near`, `doc_contain`) before 7B run #3, which trains on them.
- PubMedQA rescoring stats (negatives per row, guard rate) once `p6-bio` has mined them.
