# Data card: olmo3-embed training data

**Status: Track B v1, in progress (2026-10-05).** Ingest and decontamination are done. Hard-negative rescoring and
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

## Still to add
- LLM judge pass on Quora, SciFact and NFCorpus: OLMo-3.1-32B-Instruct for the bulk, Claude to verify, Codex and Gemini
  to break ties. Per-judge agreement goes here.
- Final per-task row counts and the sampling temperature used in training.
- A Dolma 3 contamination check (Phase 6).
