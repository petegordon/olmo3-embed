# Phase 4: first OLMo-3-7B run (plan)

**Status: run #1 DONE 2026-10-07: NanoBEIR 65.93; gate A ✗ (Qwen3-4B 69.97); gate B (full BEIR) not yet run. Results:
`docs/phase4-results.md`. (Approved 2026-10-06: run #1 at 64k rows on Colab Pro.) Started 2026-10-06
on an A100 80GB.**

## Measured cost (step 0, 2026-10-06) — replaces the estimates below
Timing cell, A100-SXM4-80GB (High-RAM, 6.77 units/h), 4 steps per memory setting:

| Setting | s/step | Peak memory |
|---|---:|---:|
| **No gradient checkpointing, chunk 8** (picked) | **38.0** | 32.4 GB |
| Gradient checkpointing, chunk 16 | 45.6 | 16.0 GB |

- **In the run:** about 44 s/step on average, since MS MARCO and HotpotQA batches have longer passages than the
  timing sample.
- **Run #1 (496 steps):** about 6.2 h ≈ **42 units**, half the estimate below. My estimate scaled the proxy's s/step
  by FLOPs and missed the A100 being much better used at 7B.
- **A full-data epoch** (~2,900 steps): about 35 h ≈ **240 units**. That fits the remaining balance without top-ups.
  At ~5 h per Colab session, that's about 7 sessions.
- **LoRA:** 39.98M trainable parameters (0.58% of 6.93B).

## Goal and gate (PLAN.md §6)
Train `allenai/Olmo-3-1025-7B` into an embedder with the Phase 2–3 recipe, then check it against the Phase 4 gate:
1. **NanoBEIR ≥ Qwen3-Embedding-4B's NanoBEIR.** We haven't measured that yet; §4 adds it.
2. **Full BEIR (15 tasks) ≥ T1 = 56.6.** That beats LLM2Vec-Llama-3-8B, the reference open recipe.

**Starting point:** the 1B proxy on Track B v1 (B3, 64k rows) reached **62.28 NanoBEIR avg-13** and **49.35
held-out**, about 1.5 points behind bge-base (63.76 / 50.98). On our harness, NanoBEIR has run about 10 points above
full BEIR for small models (bge-base: 63.8 Nano vs ~53 BEIR). So T1 (56.6 BEIR) means roughly **66–67 NanoBEIR** for
the 7B.

## Recipe (unchanged from Phase 3; `configs/trackb-base.yaml` + 7B overrides)
| | Value | Why |
|---|---|---|
| Base | `allenai/Olmo-3-1025-7B` (base, not Instruct) | PLAN §3. Phase 1: 17.35 NanoBEIR untrained (mean pooling) |
| Attention / pooling | causal, EOS last-token | Phase 2: +12.9 for EOS; bidirectional −10.4 with EOS |
| Instructions | none | Phase 2 and Phase 3 B2: −4.6 even with matched prompts. A 7B ablation is optional (§5) |
| Data | Track B v1, rescored (373,762 rows) | Phase 3: rescore +1.65; more data +4.5 |
| Batching | one task per batch, T=0.5, k=3 from the 10 hardest, false-negative masking, τ=0.02, batch 128 | Phase 3 B1/B3 |
| LoRA | r=16, α=32, all linear layers | Same as the proxy, so the comparison stays clean. r=32 is a Phase 5 lever |
| Lengths | query 64, doc 256 tokens (eval 512) | Same as the proxy. OLMo-3's 4096 sliding window never applies at these lengths |
| Checkpointing | `save_every: 10` (~20 min of 7B steps), on Drive | Colab Pro sessions end when the tab closes. Losing ≤ 10 steps per disconnect is fine |

**Implementation notes:**
- Olmo-3-7B is 32 layers × 4096, full MHA (32 KV heads), with mixed sliding/full attention layers. Its tokenizer adds
  no BOS, and our `tokenize()` appends EOS for last-token pooling. Phase 1 already ran the 7B through
  `OlmoEmbedder`, so loading and eval are proven.
- **Untested at 7B: LoRA + GradCache training.** Step 0 covers it.

## Cost model (pre-measurement estimates; see "Measured cost" above)
- **Per token,** the 7B costs about 6.4× the proxy (≈6.9B vs ≈1.07B non-embedding params).
- **Proxy speed on Track B:** 13.4 s/step on an A100 40GB without gradient checkpointing.

| Setting | Est. s/step | Units/step | 500 steps (B3-sized) | 780 steps (100k rows) | ~2,900 steps (all rows, 1 epoch) |
|---|---:|---:|---:|---:|---:|
| A100 40GB (5.3 u/h), gradient checkpointing, chunk 16 | ~115 | ~0.17 | ~16 h, **~85 u** | ~25 h, ~130 u | ~93 h, ~490 u |
| A100 80GB High-RAM (6.77 u/h), no checkpointing, chunk 8 | ~85 | ~0.16 | ~12 h, **~80 u** | ~18 h, ~125 u | ~68 h, ~460 u |

**Plus evaluation:**
- NanoBEIR for the 7B: ~0.5 h on an A100, ~3 u.
- Full 15-task BEIR (~30M+ passages to encode): ~10 h on an L4 ≈ 17 u (PLAN §5), or ~4 h on an A100 ≈ 22 u.

**Balance:** about 310 units. A full-data epoch doesn't fit without top-ups (~$15–20 of extra units), and it means
**70–90 hours of open-tab time on Colab Pro**. That's the main practical constraint, more than money.

## Steps
**0. Smoke + timing (~30 min, ~3 units).** New `notebooks/phase4_7b_colab.ipynb`, A100.
- Load the 7B with LoRA, run 4 GradCache steps in each memory setting (40GB with checkpointing; 80GB without), and
  record s/step and peak memory.
- Confirm EOS pooling and the loss behave like the proxy's first steps.
- Output: the real cost table above, so we pick the cheapest GPU per step.

**1. Run #1: the 7B on B3's exact data budget (64k rows, ~500 steps).**
- **Why this size:**
  - It isolates *model scale* against the proxy's B3 (62.28), the cleanest test of whether the 7B pays.
  - It costs ~80–85 units (fits) and ~12–16 h of runtime.
  - It runs in 3–4 Colab sessions of a few hours each: Run all resumes from the Drive checkpoint.
- **Output:** NanoBEIR avg-13 / in-8 / held-out vs B3 and the references, pushed to `colab-results` after training.

**2. Gate check A, NanoBEIR vs Qwen3-Embedding-4B.**
- Evaluate `Qwen/Qwen3-Embedding-4B` on NanoBEIR with our harness (~0.5 h on an A100, ~3 u).
- Run #1 passes if it matches or beats that score.

**3. Gate check B, full BEIR.**
- If Nano looks promising (≥ ~66 avg-13), run the full 15-task BEIR eval: an L4 notebook for cost, or an A100 for
  speed. Uses MTEB's own BEIR tasks, so the score is directly comparable to the leaderboard.
- Pass at ≥ 56.6 (T1).

**4. Decide run #2 (Phase 5 territory).**
- **If run #1 clears T1:** scale data (100k → all rows, with top-ups) and/or LoRA r=32, aiming at T2 (59.5).
- **If it misses:** check whether the gap is data or capacity (held-out vs in-domain), and consider Track A synthetic
  data (ArguAna-like counter-arguments) before spending on more steps.

## Open decisions for Pete
1. **Run #1 size: 64k rows / ~500 steps** (recommended: clean 1B-vs-7B comparison, ~80–85 units), or 100k rows
   (~125–130 units), or all rows (~460–490 units, needs top-ups and ~3–4 days of open tabs).
2. **Where it runs:**
   - **Colab Pro in sessions** (recommended for run #1). The tab stays open per session, and Run all resumes from the
     Drive checkpoint.
   - **Colab Pro+ for a month** ($49.99, background execution: close the tab and it keeps running). Worth it if we go
     to full data next.
   - Modal / RunPod (fully headless; plan in `docs/colab-hf-storage-plan.md`).
3. **Instruction ablation at 7B** (a second ~500-step run with per-task instructions, another ~80 units).
   *Recommend deferring* until run #1 shows where the 7B stands; both proxy tests said no.
4. **Small-task upsampling with all rows (only matters for full data).** At T=0.5 over 374k rows, SciFact's ~700 rows
   would repeat ~10×. Cap any task at ~3 epochs, or raise T to 0.7. Not an issue at 64k rows: there SciFact repeats
   about 1.7×.
