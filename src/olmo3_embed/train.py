"""Contrastive fine-tuning of an OLMo base model into an embedder (LoRA + InfoNCE + GradCache).

    uv run python -m olmo3_embed.train --config configs/proxy-base.yaml [--set key=value ...]
    uv run python -m olmo3_embed.eval --run runs/<name> --suite nano

Each step:
  - B queries, each with 1 positive and k hard negatives (k = num_negatives), so B*(1+k) documents.
  - Loss: InfoNCE over all documents in the batch (the query's positive vs its own hard negatives and every other
    query's positive and negatives, i.e. in-batch negatives), temperature tau.
  - GradCache (Gao et al. 2021): pass 1 embeds everything without grad in chunks, computes the loss and the gradient
    w.r.t. the embeddings; pass 2 re-embeds chunk by chunk with grad and backprops those cached gradients. The
    effective batch is limited by time, not VRAM; peak memory is one chunk.

Outputs in runs/<name>/:
  config.yaml          the resolved config
  embed_config.json    what OlmoEmbedder.from_run needs (model, pooling, attention, instructions)
  adapter/             final LoRA adapter
  ckpt/                latest resumable checkpoint (adapter + optimizer + scheduler + step), every save_every steps
  log.jsonl            one line per log_every steps: step, loss, lr, sec/step, max GPU memory
Resume: rerun the same command; it continues from ckpt/ if present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields

import pandas as pd
import torch
import yaml

from olmo3_embed.model import INSTRUCTION_TEMPLATE, embed_batch, load_base, tokenize


@dataclass
class Config:
    run_name: str = "proxy-base"
    model: str = "allenai/OLMo-2-0425-1B"
    data: str = "data/msmarco_cocondenser_50k.parquet"  # or several, comma-separated (Track B + variety sources)
    n_examples: int = 16_000  # rows used from data (Phase 2 ablations: small, same for every arm); 0 = all
    # model / representation
    pooling: str = "mean"  # mean | last
    bidirectional: bool = False
    instructions: bool = True
    instruction_mode: str = "fixed"  # fixed: `instruction` for every query | per_task: the MTEB task prompt eval uses
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    # multi-task data (Phase 3 Track B parquet: query, positives[], negatives[] hardest-first, task)
    task_batching: bool = False  # every batch from one task, tasks sampled with temperature (plan §5)
    task_temperature: float = 0.5  # rows per task ∝ n_task ** T (1 = proportional, 0 = uniform)
    neg_pool: int = 0  # sample the k negatives from the first neg_pool candidates (0 = all)
    mask_false_negatives: bool = False  # mask in-batch docs equal to one of the query's positives
    max_len_query: int = 64
    max_len_doc: int = 256
    # contrastive objective
    num_negatives: int = 3  # hard negatives per query, sampled from the prepared rank window
    temperature: float = 0.02
    batch_size: int = 128  # queries per optimizer step (effective, via GradCache)
    chunk_size: int = 32  # texts per forward chunk (memory knob only; does not change the math)
    # LoRA / optimization
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    seed: int = 0
    # bookkeeping
    out_dir: str = "runs"
    log_every: int = 5
    save_every: int = 50
    max_steps: int = 0  # >0 caps steps (smoke tests)


def load_config(path: str | None, overrides: list[str]) -> Config:
    cfg = Config()
    raw = yaml.safe_load(open(path, encoding="utf-8")) if path else {}
    for kv in overrides:
        k, v = kv.split("=", 1)
        raw[k] = yaml.safe_load(v)
    names = {f.name: f.type for f in fields(Config)}
    for k, v in (raw or {}).items():
        if k not in names:
            raise KeyError(f"unknown config key {k!r}")
        setattr(cfg, k, v)
    return cfg


def build_batches(cfg: Config) -> list[list[dict]]:
    """Deterministic for a seed: fixed rows, fixed shuffle, fixed negative sampling. Resume reuses the same list.

    Each row: query, task, positive (the one used), positives (all, for false-negative masking), negatives (k)."""
    paths = [p.strip() for p in cfg.data.split(",") if p.strip()]  # several sources: Track B + task-variety data
    if len(paths) == 1:
        df = pd.read_parquet(paths[0])
    else:
        cols = ["query", "task", "positives", "negatives"]
        df = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    if "positive" in df.columns and not cfg.task_batching:  # Phase 2 parquet: exact original sampling order
        df = df.iloc[: cfg.n_examples] if cfg.n_examples else df
        rng = random.Random(cfg.seed)
        rows = []
        for q, pos, negs in zip(df["query"], df["positive"], df["negatives"]):
            negs = list(negs)
            if len(negs) < cfg.num_negatives:
                continue
            rows.append({"query": q, "task": "MSMARCO", "positive": pos, "positives": [pos],
                         "negatives": rng.sample(negs, cfg.num_negatives)})
        rng.shuffle(rows)
        n = len(rows) // cfg.batch_size
        return [rows[i * cfg.batch_size : (i + 1) * cfg.batch_size] for i in range(n)]

    if "positive" in df.columns:  # Phase 2 parquet under task batching: one task
        df = df.assign(positives=df["positive"].map(lambda p: [p]), task="MSMARCO")
    rng = random.Random(cfg.seed)
    by_task: dict[str, list[dict]] = {}
    for q, task, P, N in zip(df["query"], df["task"], df["positives"], df["negatives"]):
        P = [p for p in P if p]
        pool = list(N)[: cfg.neg_pool] if cfg.neg_pool else list(N)
        if not P or len(pool) < cfg.num_negatives:
            continue
        by_task.setdefault(task, []).append({"query": q, "task": task, "positives": P, "pool": pool})
    tasks = sorted(by_task)
    sizes = {t: len(by_task[t]) for t in tasks}
    total = min(cfg.n_examples or sum(sizes.values()), sum(sizes.values()))
    weights = {t: sizes[t] ** cfg.task_temperature for t in tasks}
    z = sum(weights.values())
    batches = []
    for t in tasks:
        want = round(total * weights[t] / z)
        src = by_task[t]
        picked = []
        while len(picked) < want:  # without replacement first; repeat (reshuffled) only if a small task is upsampled
            perm = src[:]
            rng.shuffle(perm)
            picked += perm[: want - len(picked)]
        rows = [{"query": r["query"], "task": t, "positive": rng.choice(r["positives"]), "positives": r["positives"],
                 "negatives": rng.sample(r["pool"], cfg.num_negatives)} for r in picked]
        n = len(rows) // cfg.batch_size
        batches += [rows[i * cfg.batch_size : (i + 1) * cfg.batch_size] for i in range(n)]
    if not cfg.task_batching:  # mixed batches: same rows, re-cut across tasks
        rows = [r for b in batches for r in b]
        rng.shuffle(rows)
        n = len(rows) // cfg.batch_size
        batches = [rows[i * cfg.batch_size : (i + 1) * cfg.batch_size] for i in range(n)]
    rng.shuffle(batches)
    return batches


def false_negative_mask(batch: list[dict], k: int) -> torch.Tensor:
    """(B, B*(1+k)) bool: True where an in-batch doc equals one of query i's positives, except its own target.
    Those docs are relevant to query i, so they must not count as negatives (shared corpora: Quora, HotpotQA, FEVER)."""
    docs = [t for r in batch for t in [r["positive"], *r["negatives"]]]
    mask = torch.zeros(len(batch), len(docs), dtype=torch.bool)
    for i, r in enumerate(batch):
        P = set(r["positives"])
        for j, d in enumerate(docs):
            if d in P and j != i * (1 + k):
                mask[i, j] = True
    return mask


def query_text(r: dict, cfg: Config) -> str:
    if not cfg.instructions:
        return r["query"]
    instr = task_instruction(r["task"]) if cfg.instruction_mode == "per_task" else cfg.instruction
    return INSTRUCTION_TEMPLATE.format(instruction=instr, text=r["query"])


_INSTR: dict[str, str] = {}


def task_instruction(task: str) -> str:
    """The query prompt MTEB's metadata gives the task: exactly what OlmoEmbedder uses at eval time."""
    if task not in _INSTR:
        import mteb
        p = mteb.get_task(task).metadata.prompt
        _INSTR[task] = p["query"] if isinstance(p, dict) else p
    return _INSTR[task]


def chunked_embed(model, tokenizer, texts, cfg: Config, max_len: int, grad_out: torch.Tensor | None = None):
    """No grad_out: embed all texts without grad, return (N, d). With grad_out (N, d): re-embed chunk by chunk with
    grad and backprop the cached embedding gradients (GradCache pass 2)."""
    device = next(model.parameters()).device
    outs = []
    for s in range(0, len(texts), cfg.chunk_size):
        enc = tokenize(tokenizer, texts[s : s + cfg.chunk_size], cfg.pooling, max_len).to(device)
        if grad_out is None:
            with torch.no_grad():
                outs.append(embed_batch(model, enc, cfg.pooling, cfg.bidirectional))
        else:
            emb = embed_batch(model, enc, cfg.pooling, cfg.bidirectional)
            emb.backward(grad_out[s : s + cfg.chunk_size])
    return torch.cat(outs) if outs else None


def infonce(q: torch.Tensor, d: torch.Tensor, k: int, tau: float,
            mask: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
    """q: (B, h); d: (B*(1+k), h) laid out [pos_0, negs_0..., pos_1, negs_1..., ...]. The target for query i is d[i*(1+k)].
    mask (B, B*(1+k)) bool: logits excluded from the softmax (in-batch false negatives)."""
    scores = q @ d.T / tau
    if mask is not None:
        scores = scores.masked_fill(mask.to(scores.device), float("-inf"))
    target = torch.arange(q.size(0), device=q.device) * (1 + k)
    loss = torch.nn.functional.cross_entropy(scores, target)
    acc = (scores.argmax(1) == target).float().mean().item()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value (YAML-parsed)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(run_dir, exist_ok=True)
    yaml.safe_dump(asdict(cfg), open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8"), sort_keys=False)

    torch.manual_seed(cfg.seed)
    from peft import LoraConfig, PeftModel, get_peft_model

    tokenizer, base = load_base(cfg.model, cfg.dtype)
    base.to("cuda")
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules="all-linear", bias="none",
    )
    resuming = os.path.isdir(ckpt_dir)
    model = PeftModel.from_pretrained(base, ckpt_dir, is_trainable=True) if resuming else get_peft_model(base, lora)
    model.print_trainable_parameters()

    batches = build_batches(cfg)
    total = len(batches) if cfg.max_steps <= 0 else min(cfg.max_steps, len(batches))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_ratio * total))

    def lr_lambda(s: int) -> float:  # linear warmup, then linear decay to 0
        return (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step = 0
    if resuming:
        st = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), weights_only=False)
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed from step {step}/{total}", flush=True)

    k = cfg.num_negatives
    log = open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")
    model.train()
    t0 = time.time()
    while step < total:
        batch = batches[step]
        q_texts = [query_text(r, cfg) for r in batch]
        d_texts = [t for r in batch for t in [r["positive"], *r["negatives"]]]
        mask = false_negative_mask(batch, k) if cfg.mask_false_negatives else None
        torch.cuda.reset_peak_memory_stats()

        # GradCache pass 1: embeddings without grad -> loss -> d loss / d embeddings
        q = chunked_embed(model, tokenizer, q_texts, cfg, cfg.max_len_query).requires_grad_()
        d = chunked_embed(model, tokenizer, d_texts, cfg, cfg.max_len_doc).requires_grad_()
        loss, acc = infonce(q, d, k, cfg.temperature, mask)
        loss.backward()
        # pass 2: recompute with grad, backprop the cached embedding grads into the LoRA weights
        chunked_embed(model, tokenizer, q_texts, cfg, cfg.max_len_query, grad_out=q.grad)
        chunked_embed(model, tokenizer, d_texts, cfg, cfg.max_len_doc, grad_out=d.grad)

        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == total:
            dt = (time.time() - t0) / cfg.log_every
            t0 = time.time()
            rec = {
                "step": step, "of": total, "task": batch[0]["task"], "loss": round(loss.item(), 4), "acc": round(acc, 3),
                "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 3), "sec_per_step": round(dt, 2),
                "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(rec, flush=True)
            if not math.isfinite(rec["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
        if step % cfg.save_every == 0 and step < total:
            model.save_pretrained(ckpt_dir)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step},
                       os.path.join(ckpt_dir, "trainer_state.pt"))

    model.save_pretrained(os.path.join(run_dir, "adapter"))
    json.dump(
        {"model": cfg.model, "run_name": cfg.run_name, "pooling": cfg.pooling, "bidirectional": cfg.bidirectional,
         "instructions": cfg.instructions, "instruction_mode": cfg.instruction_mode, "dtype": cfg.dtype, "steps": total},
        open(os.path.join(run_dir, "embed_config.json"), "w", encoding="utf-8"), indent=2,
    )
    print(f"done: {run_dir}/adapter ({total} steps)", flush=True)


if __name__ == "__main__":
    main()
