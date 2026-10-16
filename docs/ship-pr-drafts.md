# Ship PRs (Phase 9)

**Opened Oct 15:** [mteb#5677](https://github.com/embeddings-benchmark/mteb/pull/5677) and [results#775](https://github.com/embeddings-benchmark/results/pull/775). The text below is what was posted, with the cross-links as placeholders.

Branches are rebased on upstream main (Oct 15) and pushed to the forks. Open mteb first: the results repo's CI
checks that every model has an mteb implementation, so it passes only once the mteb PR is merged and released.

| Repo | Fork branch | Checks run locally (Oct 15) |
|---|---|---|
| embeddings-benchmark/mteb | [`petegordon:add-olmo3-embed-7b`](https://github.com/embeddings-benchmark/mteb/compare/main...petegordon:add-olmo3-embed-7b) (3fb60c50) | ruff check + format clean; model-meta, import-hygiene, no-torch-at-import, task-to-prompt tests: 3,731 passed |
| embeddings-benchmark/results | [`petegordon:add-olmo3-embed-7b`](https://github.com/embeddings-benchmark/results/compare/main...petegordon:add-olmo3-embed-7b) (5e792f273) | full suite against the mteb branch: 6,198 passed, 725 skipped; `model_meta.json` equals the ModelMeta field for field |

History: mteb #5654 was opened before review on Oct 14 and closed (its description was 2,419 chars, over the bot's
2,000 limit). It can't be reopened after the force-push, so the mteb PR is a new one.

---

## mteb PR

**Title:** `model: add petegordon/olmo3-embed-7b`

```markdown
Adds [petegordon/olmo3-embed-7b](https://huggingface.co/petegordon/olmo3-embed-7b): a LoRA fine-tune of [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) for English retrieval, merged and saved as a Sentence Transformers model (4096-d, last-token pooling on an EOS the tokenizer appends, 512 tokens, bf16).

- **Loader:** `InstructSentenceTransformerModel`; queries get `Instruct: {instruction}\nQuery: `, documents get no instruction.
- **Training data:** the train splits of MSMARCO, HotpotQA, NQ, FEVER, FiQA2018, NFCorpus, SciFact and QuoraRetrieval (all in `training_datasets`, so it is 46% zero-shot on BEIR), plus StackExchange (CQADupStack forums excluded), S2ORC and PubMedQA. Hard negatives were filtered with Qwen3-Embedding-0.6B as a scorer (no distillation). [Data card](https://github.com/petegordon/olmo3-embed/blob/main/docs/data-card.md)
- **Results:** RESULTS_PR_LINK (BEIR, 15 tasks, avg nDCG@10 54.72)

- [x] I have filled out the ModelMeta object to the extent possible
- [x] I have ensured that my model can be loaded using
  - [x] `mteb.get_model(model_name, revision)` and
  - [x] `mteb.get_model_meta(model_name, revision)`
- [x] I have tested the implementation works on a representative set of tasks.
  - Through this loader, the export (uploaded unchanged as `ad1008e`) scores SciFact 79.44 / NFCorpus 38.78 vs 79.47 / 38.98 for the training-time encoder; embedding cosine ≥ 0.9998.
- [x] The model is public, i.e., is available either as an API or the weights are publicly available to download
- [ ] I reproduced results from the original paper (if applicable) on at least one benchmark, and I am including the results in the PR description.
  - N/A: no paper.
```

---

## results PR

**Title:** `Add results for petegordon/olmo3-embed-7b (BEIR)`

```markdown
BEIR results (15 tasks, avg nDCG@10 54.72) for [petegordon/olmo3-embed-7b](https://huggingface.co/petegordon/olmo3-embed-7b) at revision `ad1008e`. Model implementation: MTEB_PR_LINK.

MSMARCO was run as 3 corpus shards, and mteb re-scored the union of their top-k (identical to a single full-corpus run on all metrics in a SciFact check), so its `evaluation_time` covers only that last step.

### Checklist

- [x] My model has a model sheet, report, or similar ([model card](https://huggingface.co/petegordon/olmo3-embed-7b))
- [x] My model has a reference implementation in `mteb/models/model_implementations/`
  - [x] No, but there is an existing PR MTEB_PR_LINK
- [ ] The results submitted are obtained using the reference implementation
  - They come from the training-time encoder (same weights, mteb 2.24). The reference implementation reproduces them: SciFact 79.44 vs 79.47, NFCorpus 38.78 vs 38.98, embedding cosine ≥ 0.9998.
- [x] My model is available, either as a publicly accessible API or publicly on e.g., Huggingface
- [x] I *solemnly swear* that for all results submitted I have not trained on the evaluation dataset including training splits. If I have, I have disclosed it clearly.
  - Disclosed: trained on the train splits of MSMARCO, HotpotQA, NQ, FEVER, FiQA2018, NFCorpus, SciFact and QuoraRetrieval (`training_datasets` in the ModelMeta); no test data.
```
