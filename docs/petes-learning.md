# Pete's Learning: terms and concepts, explained as they come up

The plans say what we intend, the results docs say what we scored, and `docs/worklog.md` is the diary. **This file is
the glossary:** each time a term or concept comes up that I want to actually understand, the explanation goes here,
tied back to this project. Newest entry at the bottom.

## Contents

1. [Olmo-core 3 vs OLMo 3](#1-olmo-core-3-vs-olmo-3-2026-10-10) (Oct 10)
2. [Dense vs MoE, and causal vs bidirectional attention](#2-dense-vs-moe-and-causal-vs-bidirectional-attention-2026-10-10) (Oct 10)
3. [Is building an embedding model "post-training"?](#3-is-building-an-embedding-model-post-training-2026-10-10) (Oct 10)
4. [What would the weakly supervised stage take?](#4-what-would-the-weakly-supervised-stage-take-2026-10-10) (Oct 10)
5. [bge-en-icl and LENS: the open recipes closest to ours](#5-bge-en-icl-and-lens-the-open-recipes-closest-to-ours-2026-10-10) (Oct 10)

---

## 1. Olmo-core 3 vs OLMo 3 (2026-10-10)

**Question:** is the [Olmo-core 3 blog post](https://huggingface.co/blog/allenai/olmocore3) (Oct 1, 2026) about the
same OLMo 3 we're training on?

**No. Same name, different thing.**

- **OLMo-core** is Ai2's PyTorch *training library*. **Olmo-core 3** is version 3 of that library. Its headline is a
  redesigned mixture-of-experts (MoE) training system that scales into the trillion-parameter range. It ships no
  new model weights.
- **OLMo 3** is a *model family* (late 2025). Ours is `allenai/Olmo-3-1025-7B`, the dense 7B base model. The post
  says so itself: "Olmo 3, by contrast, used a dense architecture."
- The post says the **next** Olmo will be an MoE trained with the new stack. That model isn't out yet.

**What it means for us:** nothing for now. We train with Hugging Face `transformers` + LoRA, not OLMo-core. If the
MoE Olmo comes out, it's a possible future base, but harder for us. An MoE must keep *all* its parameters in GPU memory
even though each token uses only a few (the post's example: 47B total, ~3.2B active per token). LoRA and last-token
pooling are also much less proven on MoEs.

---

## 2. Dense vs MoE, and causal vs bidirectional attention (2026-10-10)

**Question:** are the leading MTEB/BEIR models dense or MoE? And what is the "causal" difference between OLMo 3 and
them?

### "Dense" has two meanings

1. **Dense architecture vs MoE:** in a dense model, every parameter is used for every token. An MoE routes each token
   to a few "expert" sub-networks. This is the meaning in the Olmo-core post.
2. **Dense retrieval vs sparse retrieval:** a dense retriever turns each text into one vector of numbers. Sparse
   retrieval matches words (BM25, SPLADE). Multi-vector models (ColBERT) keep one vector per token. This project is
   building a dense retriever.

The leading MTEB models are dense in **both** senses.

### Who leads, and what they're built on (as of Oct 2026)

| Model | Base | Architecture | Attention when embedding |
|---|---|---|---|
| Qwen3-Embedding-8B | Qwen3 8B decoder LLM | dense | **causal**, last-token pooling |
| NV-Embed-v2 | Mistral 7B decoder LLM | dense | **bidirectional**, latent-attention pooling |
| llama-embed-nemotron-8b | Llama 3.1 8B decoder LLM | dense | bidirectional |
| Gemini Embedding 001 | Gemini | undisclosed (closed) | undisclosed |
| Voyage 4 | undisclosed | MoE, per one blog | undisclosed |

The best **open** models are dense decoder LLMs of ~7–8B, the same kind of model as `Olmo-3-1025-7B`. MoE shows up
mainly in closed commercial APIs and in small models like Nomic Embed v2. (Rankings move often, and the blogs I found
disagree on who's #1. The pattern holds across them.)

### What "causal" means

A causal model reads text so that **each token can only see the tokens before it**. LLMs are trained that way because
they predict the next word, and seeing the future would be cheating. BERT-style encoders are **bidirectional**: every
token sees every other token.

```
causal (GPT, Llama, OLMo)        bidirectional (BERT-style encoders)
        The cat sat .                    The cat sat .
The     ✓   ·   ·   ·            The     ✓   ✓   ✓   ✓
cat     ✓   ✓   ·   ·            cat     ✓   ✓   ✓   ✓
sat     ✓   ✓   ✓   ·            sat     ✓   ✓   ✓   ✓
.       ✓   ✓   ✓   ✓            .       ✓   ✓   ✓   ✓
```

To get one embedding for a text, we combine the model's per-token outputs (hidden states). "Pooling" is how:

- **Mean pooling** averages all the token states. In a causal model, early tokens like "The" never saw the rest of
  the sentence, so they water down the average. That's a big part of why untrained OLMo 3 scores only 17 on NanoBEIR.
- **Last-token (EOS) pooling** takes only the final token's state. It's the only token that has seen the whole text,
  so the causal limit doesn't hurt. Qwen3-Embedding does this.
- **Removing the causal mask** (going bidirectional) makes every token see everything. The model was never trained
  that way, so it needs adapting. NV-Embed and LLM2Vec do that during training; LLM2Vec adds an extra step (MNTP,
  masked next-token prediction).

### Our Phase 2 already tested this

From `docs/phase2-results.md` (OLMo-2-1B proxy):

- Bidirectional + mean pooling: **+5.6** over causal mean pooling.
- Bidirectional + EOS pooling: **−10.4**. Removing the mask without adapting the model scrambles the final-token
  state.
- Causal + EOS pooling: best recipe, **57.51**.

So our 7B recipe (causal, EOS last-token pooling) takes the same route as Qwen3-Embedding-8B. Both routes can reach the
top; causal attention isn't what keeps us below the leaders.

### What actually separates us from the leaders

Mostly **training**, not architecture:

- **Data scale:** Qwen3-Embedding pre-trained on ~150M synthetic query–document pairs, then fine-tuned on high-quality
  labeled data.
- **Stages and tricks:** hard-negative mining, task instructions, and merging several checkpoints into one model.

Our Track B and synthetic-data work is aimed at that gap.

---

## 3. Is building an embedding model "post-training"? (2026-10-10)

**Short answer:** loosely yes, but it's not what the field means by the term. The precise name for what we do is
**contrastive fine-tuning** (of a base LLM into an embedding model).

### What "post-training" usually means

An LLM is built in two big steps:

1. **Pre-training:** next-token prediction on trillions of tokens of raw text. The result is a **base model** that
   continues text. Our `Olmo-3-1025-7B` is this.
2. **Post-training:** everything that turns the base model into an assistant. For OLMo 3 that's SFT (supervised
   fine-tuning on example conversations), DPO (preference tuning: "this answer is better than that one") and RLVR
   (reinforcement learning on answers that can be checked). The results are `Olmo-3-7B-Instruct` and `-Think`.

When Ai2 or anyone else says "post-training", they mean step 2. The model still *generates text*; it just behaves
better.

### What we do instead

We start from the **base** model (not Instruct, see `PLAN.md`) and train it on a different job entirely:

- **Objective:** contrastive loss (InfoNCE). Given a query, pull its relevant document's vector closer and push the
  irrelevant ones (in-batch and hard negatives) away. No text is generated and the next-token head is unused.
- **Output:** a vector per text, compared by cosine similarity.
- **Method:** LoRA on all linear layers; the original weights stay frozen.

It does happen *after* pre-training, so it's post-training in the literal sense. But it's really a **separate branch**
from the base model, alongside Ai2's chat post-training rather than after it:

```
                  ┌─ post-training (SFT → DPO → RLVR) ─→ Olmo-3-7B-Instruct / Think   (generates text)
Olmo-3-1025-7B ───┤
  (pre-trained)   └─ contrastive fine-tuning (us) ────→ olmo3-embed                  (outputs vectors)
```

### A naming trap in embedding papers

Embedding papers reuse "pre-training" for their own **large first stage of contrastive training** on weak, noisy pairs.
Qwen3-Embedding calls its ~150M-pair stage "weakly supervised pre-training" and its second stage "supervised
fine-tuning", even though both happen after the LLM's real pre-training. When reading a paper, check which model's
training it's talking about.

In those terms, we currently do a **single-stage supervised contrastive fine-tune** (MS MARCO / Track B + task-variety
data, hard negatives). A large weakly supervised first stage is the main thing the leaders have that we don't.

---

## 4. What would the weakly supervised stage take? (2026-10-10)

**Question:** what would it take to add the big first stage the leaders have? Is Qwen's data available? Is it just GPU
time, or data work too?

**Short answer:** all three are hard for us: the data, the cleaning, and the GPU time. And for a 7B LLM it may not be
worth much.

### The data

- **Qwen3's ~150M pairs are not released.** They were generated by Qwen3-32B, which wrote queries for documents while
  playing different user roles ([paper](https://arxiv.org/abs/2506.05176)). Only the models are public.
- **Open alternative:** [`nomic-ai/nomic-embed-unsupervised-data`](https://huggingface.co/datasets/nomic-ai/nomic-embed-unsupervised-data),
  the weak pairs behind Nomic Embed's first stage. About 470M raw pairs from ~29 sources (Reddit, StackExchange, PAQ,
  Amazon reviews, news, S2ORC, ...), filtered down to ~235M by checking each pair with an existing embedder
  ("consistency filtering"). Older option: the `sentence-transformers` embedding-training-data collection.
- **Make our own:** OLMo-3.1-32B ran at ~3.7 prompts/s on Colab in Phase 3, about 13k/hour. 1M synthetic pairs
  would be ~75 GPU-hours of generation before any training.

### The cleaning (the big hidden cost)

Weak-pair sources overlap BEIR's test corpora almost everywhere: PAQ and Wikipedia vs NQ, HotpotQA, FEVER, DBPedia and
ClimateFEVER; S2ORC vs SCIDOCS, SciFact and TREC-COVID; StackExchange vs CQADupStack; Reddit/Quora-style duplicates vs
Quora. Our decontamination pipeline (`doc_decontam`, index v2) would have to run over tens or hundreds of millions of
rows instead of ~400k, and licenses are mixed across sources. That's the most work, and it's what keeps Track A
"zero-shot-clean".

### The GPU time

From 7B run #1: ~128 rows/step at ~44 s/step on an A100 80GB ≈ 10k rows/hour, where each row is 9 texts (1 query,
1 positive, 7 hard negatives). A weak pair is 2 texts, so roughly 4–5× cheaper: **~45k pairs/hour, ~6.8 Colab
units/hour.** Rough estimates for the 7B:

| Weak pairs | A100 hours | Colab units |
|---|---|---|
| 150M (Qwen-scale) | ~3,300 | ~22,000 |
| 10M | ~220 | ~1,500 |
| 1M | ~22 | ~150 |

We have a couple of hundred units left in total. Qwen-scale isn't reachable on Colab; 1M is the most we could try.

### Is it even worth it for a 7B?

Evidence is mixed. The stage clearly helps small encoders (BERT-sized models). For 7B LLMs, E5-Mistral ("Improving Text
Embeddings with Large Language Models", Wang et al. 2023) found contrastive pre-training added little, because
the LLM's own pre-training already does much of that job. NV-Embed and SFR-Embedding also skipped it and still reached
the top.

### What a cheap test would look like

1. On the 1B proxy: a weak stage of ~2M decontaminated pairs from a few low-overlap Nomic sources, then our B3
   fine-tune on top.
2. Compare against plain B3 on the held-out tasks (the same gate as earlier phases).
3. Only if it wins clearly, try ~1M pairs on the 7B.

---

## 5. bge-en-icl and LENS: the open recipes closest to ours (2026-10-10)

**Question:** bge-en-icl (#5 on the HF BEIR board) says its training data and code are open; LENS-d8000 (#6) says its
data is open. What do they tell us?

| | bge-en-icl (BAAI) | LENS-d8000 (Lei et al., ACL 2025) | Us (7B run #2) |
|---|---|---|---|
| BEIR-15 | 62.16 | 61.86 | ~55 (estimate, rank ~33) |
| Base model | Mistral-7B (dense decoder) | Mistral-7B (dense decoder) | OLMo-3-7B (dense decoder) |
| Attention / pooling | causal, EOS last token | bidirectional, max-pool over vocabulary logits | causal, EOS last token |
| Output | one dense vector | 8,000-dim "lexicon" vector | one dense vector |
| Training | one supervised contrastive stage (InfoNCE, hard negatives) | same, on bge-en-icl's public data | one supervised contrastive stage |
| Weak 150M-pair stage | no | no | no |
| Data | `cfli/bge-full-data` (public, no license declared) | bge-en-icl's public data | Track B (partly from the same bge data) + variety |
| Code | open (FlagEmbedding) | open (github.com/Yibin-Lei/LENS) | ours |

**The lesson:** two of the top six reached ~62 from a 7B LLM **without** a Qwen-style weak stage, with one supervised
contrastive stage on public data. That's our recipe's shape. The ~7-point gap is in the details (data mix and size,
hard-negative quality, task instructions, few-shot examples), not a missing stage.

**What "ICL" means:** in-context learning. bge-en-icl can take a few example query–document pairs in front of the
query, the way you'd give an LLM examples in a prompt. It's trained with these examples so it learns to use them.

**Checked (Oct 10): the leaderboard's 62.16 is the few-shot score.**

- The MTEB results repo stores bge-en-icl's BEIR scores under revision `external`. That means BAAI reported them
  itself (via its model card); MTEB didn't run the model. The file metadata doesn't say which mode was used.
- The paper's Table 1 (full training data) reports MTEB Retrieval, the same 15 BEIR tasks, as **62.16 few-shot** and
  **61.67 zero-shot**. Our snapshot's 15-task average for bge-en-icl is exactly 62.16, so the leaderboard number is the
  few-shot one.
- BAAI's eval pipeline (with the per-task example pairs) is still unreleased: it's the one unchecked item in their
  open-source plan.
- Zero-shot (61.67), it would rank **#8** in our Oct 4 snapshot, just below inf-retriever-v1 (61.71) and above
  Qwen3-Embedding-4B (61.58). The few-shot gain on retrieval is small (+0.5), so it's still a strong zero-shot model.
- For our comparisons, use 61.67: our model gets no examples at eval time.

**What LENS does differently:** instead of a hidden-state vector, it reads the LLM's *next-token scores over the
vocabulary* (how strongly the text "means" each word), groups the ~32k vocabulary tokens into 8,000 clusters of
similar words, and keeps the max per cluster. Each dimension roughly stands for a group of words. It sits between dense
retrieval and sparse word matching (entry 2).

**The experiment this suggests:** train OLMo-3-7B with bge-en-icl's open code and public data, recipe unchanged. That
compares OLMo to Mistral under identical training, and shows how much of our gap is recipe vs base model. Caveats: the
bge data declares no license, contains BEIR train splits (Track B only), and needs our decontamination pass.

