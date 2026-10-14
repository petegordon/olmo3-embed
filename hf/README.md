---
license: apache-2.0
language:
- en
base_model: allenai/Olmo-3-1025-7B
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- mteb
- olmo
- retrieval
---

# olmo3-embed-7b

A text embedding model fine-tuned from [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
for retrieval. It is an independent project, not affiliated with Ai2; training code, configs, data card and logs are
at [github.com/petegordon/olmo3-embed](https://github.com/petegordon/olmo3-embed).

- **Architecture:** OLMo-3-7B (causal attention, unchanged), embedding = final hidden state of an appended EOS token
  (last-token pooling), L2-normalized, 4096 dimensions. Max 512 tokens.
- **Training:** LoRA (r 16, α 32) with an InfoNCE contrastive loss (temperature 0.02, in-batch + 3 mined hard negatives,
  false-negative masking), 893 steps × 128, then merged into the base weights. Released here as full bf16 weights.
- **Instructions:** queries take a task instruction, documents take none:
  `Instruct: {instruction}\nQuery: {query}`.

## Usage

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("petegordon/olmo3-embed-7b", model_kwargs={"torch_dtype": torch.bfloat16})
prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
q = model.encode(["how do solar panels work"], prompt=prompt)
d = model.encode(["Photovoltaic cells convert sunlight directly into electricity ..."])
print(model.similarity(q, d))
```

The tokenizer appends EOS itself (its post-processor), so no extra handling is needed. In MTEB, the per-task
instructions are each task's default MTEB prompt.

## Results: BEIR (MTEB), nDCG@10

| Avg (15) | ArguAna | ClimateFEVER | CQADupStack | DBPedia | FEVER | FiQA2018 | HotpotQA | MSMARCO | NFCorpus | NQ | Quora | SCIDOCS | SciFact | TREC-COVID | Touche2020 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **54.72** | 57.82 | 37.00 | 45.76 | 48.28 | 89.09 | 53.39 | 75.41 | 38.57 | 38.98 | 62.12 | 89.66 | 23.65 | 79.47 | 63.10 | 18.52 |

MSMARCO is the dev split; the rest are test. Scored with mteb 2.22–2.24 in bf16 on the LoRA run before merging. This merged
model, loaded through mteb, gives the same embeddings (cosine ≥ 0.9998 over all NFCorpus queries and 3,000 documents)
and SciFact 79.44 / NFCorpus 38.78 (bf16 batching noise).

## Training data

About 115k (query, positive, negatives) rows, all human-written public data. Details and licenses are in the
[data card](https://github.com/petegordon/olmo3-embed/blob/main/docs/data-card.md).

- Train splits of MS MARCO, HotpotQA, NQ, FEVER and FiQA-2018 (via `rlhn/rlhn-680K`), Quora duplicates, and the
  NFCorpus and SciFact train qrels. These are in-domain for those MTEB tasks.
- StackExchange title/body pairs (80 sites; the 12 CQADupStack forums excluded), S2ORC title–citation pairs, and
  PubMedQA questions with their abstracts.
- Hard negatives were mined and filtered with Qwen/Qwen3-Embedding-0.6B as a scorer.

**Decontamination:** every training query was checked against 72k test queries from 39 MTEB tasks (exact and
near-duplicate), and every passage against the test corpora of the zero-shot tasks (TREC-COVID, NFCorpus, SciFact,
SCIDOCS, ArguAna, Touche2020, CQADupStack): rows with any overlap were dropped. TREC-COVID was kept strictly
zero-shot: no queries, no qrels, and no document overlap.

## Limitations

English only, 512-token inputs. Trained for retrieval; other embedding tasks are untested. Inherits the base
model's biases.

## License

Apache-2.0, like the base model. Training data licenses are listed in the data card.
