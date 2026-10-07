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
