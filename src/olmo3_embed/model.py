"""OLMo embedding model: the shared forward pass for training (train.py) and MTEB evaluation (OlmoEmbedder).

Pooling:
  mean  average of last hidden states over non-pad tokens
  last  hidden state of an appended EOS token. OLMo tokenizers do NOT add EOS themselves, so we append it after
        truncation (otherwise "last token" is just the final word)

Attention:
  causal         the pretrained default
  bidirectional  every token attends to every non-pad token, via a 4D additive mask (also overrides OLMo-3's
                 sliding-window layers). Verified 2026-10-04: editing the last token changes the first token's
                 hidden state only in this mode.

Instructions (queries only, same format as Qwen3/E5): "Instruct: {task instruction}\nQuery: {text}".
Documents never get an instruction.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from transformers import AutoModel, AutoTokenizer

INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"


# ---------------------------------------------------------------- shared forward pass (train + eval)

def token_ids(tokenizer, texts: list[str], pooling: str, max_length: int) -> list[list[int]]:
    eos = tokenizer.eos_token_id
    if pooling == "mean":
        ids = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
        # Empty docs exist (NanoFiQA2018 has 27) and OLMo adds no BOS, so they'd be 0 tokens; a batch of only
        # empties (length-sorted batching puts them together) then crashes attention. Encode them as [EOS].
        return [x or [eos] for x in ids]
    # last-token: truncate leaving room for EOS, append it (padding goes on the right)
    ids = tokenizer(texts, truncation=True, max_length=max_length - 1)["input_ids"]
    return [x + [eos] for x in ids]


def tokenize(tokenizer, texts: list[str], pooling: str, max_length: int) -> dict[str, torch.Tensor]:
    ids = token_ids(tokenizer, texts, pooling, max_length)
    return tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")


class DocEmbeddingCache:
    """Document embeddings keyed by a hash of the text, stored float16 in a preallocated memmap on local disk.

    FEVER and ClimateFEVER share one 5.4M-passage Wikipedia corpus, so evaluating them in one process with this cache
    embeds it once (~20 A100-hours saved). Lives for one process only; documents past `capacity` just aren't cached.
    """

    def __init__(self, path: str, dim: int, capacity: int):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.emb = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(capacity, dim))
        self.rows: dict[bytes, int] = {}
        self.hits = self.misses = 0

    @staticmethod
    def key(text: str) -> bytes:
        import hashlib

        return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).digest()

    def get(self, keys: list[bytes]) -> list[int | None]:
        return [self.rows.get(k) for k in keys]

    def put(self, keys: list[bytes], emb: np.ndarray) -> None:
        for k, e in zip(keys, emb):
            if k in self.rows or len(self.rows) >= len(self.emb):
                continue
            self.rows[k] = len(self.rows)
            self.emb[self.rows[k]] = e


def bidirectional_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(b, k) padding mask -> (b, 1, q, k) additive mask where every query attends to every non-pad key."""
    keep = attention_mask[:, None, None, :].bool().expand(-1, 1, attention_mask.size(1), -1)
    return torch.zeros(keep.shape, dtype=dtype, device=attention_mask.device).masked_fill(
        ~keep, torch.finfo(dtype).min
    )


def embed_batch(model, enc: dict[str, torch.Tensor], pooling: str, bidirectional: bool) -> torch.Tensor:
    """Forward + pool + L2-normalize. Differentiable; callers wrap in no_grad for inference."""
    mask = enc["attention_mask"]
    attn = bidirectional_mask(mask, model.dtype) if bidirectional else mask
    hidden = model(input_ids=enc["input_ids"], attention_mask=attn).last_hidden_state.float()
    if pooling == "mean":
        emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    else:
        emb = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
    return torch.nn.functional.normalize(emb, dim=-1)


def load_base(model_name: str, dtype: str = "bfloat16", revision: str | None = None, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.padding_side = "right"
    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
        kwargs["device_map"] = {"": 0}
    return tokenizer, AutoModel.from_pretrained(model_name, revision=revision, **kwargs)


# ---------------------------------------------------------------- MTEB encoder

class OlmoEmbedder(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "last",
        instructions: bool = True,
        bidirectional: bool = False,
        adapter: str | None = None,
        run_name: str | None = None,
        dtype: str = "bfloat16",
        max_length: int = 512,
        revision: str | None = None,
        device: str | None = None,
        load_in_4bit: bool = False,
        batch_size: int = 32,
        token_budget: int | None = None,
        doc_cache: DocEmbeddingCache | None = None,
    ):
        if pooling not in ("mean", "last"):
            raise ValueError(pooling)
        self.pooling, self.instructions, self.bidirectional = pooling, instructions, bidirectional
        self.max_length = max_length
        self.batch_size = batch_size  # adaptive: halved on CUDA OOM
        # token_budget: batch by padded tokens (rows x longest row) instead of a fixed row count, so short documents
        # go in big batches. None keeps the fixed batch_size (the setting all results before 2026-10-13 used).
        self.token_budget = token_budget
        self.doc_cache = doc_cache
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer, model = load_base(model_name, dtype, revision, load_in_4bit)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        if not load_in_4bit:
            self.model.to(self.device)
        sha = getattr(self.model.config, "_commit_hash", None) or revision or "main"
        if run_name:  # trained run: results/olmo3-embed__<run_name>/
            name = f"olmo3-embed/{run_name}"
        else:  # untrained variant (Phase 1 naming, kept stable)
            tag = f"{pooling}{'-bidir' if bidirectional else ''}{'-instr' if instructions else ''}"
            name = f"olmo3-embed/{model_name.split('/')[-1]}-{tag}{'-4bit' if load_in_4bit else ''}"
        # Results folder = results/<name>/<revision>/; one folder per variant so runs never collide.
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=name,
            revision=sha,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=sum(p.numel() for p in self.model.parameters()),
            memory_usage_mb=None,
            max_tokens=max_length,
            embed_dim=self.model.config.hidden_size,
            license="apache-2.0",
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            reference=f"https://huggingface.co/{model_name}",
            similarity_fn_name=ScoringFunction.COSINE,
            use_instructions=instructions,
            training_datasets=None,
        )

    @classmethod
    def from_run(cls, run_dir: str, **overrides) -> "OlmoEmbedder":
        """Load a trained run: runs/<name>/embed_config.json + runs/<name>/adapter/."""
        cfg = json.load(open(os.path.join(run_dir, "embed_config.json"), encoding="utf-8"))
        kwargs = dict(
            pooling=cfg["pooling"], instructions=cfg["instructions"], bidirectional=cfg["bidirectional"],
            adapter=os.path.join(run_dir, "adapter"), run_name=cfg["run_name"], dtype=cfg.get("dtype", "bfloat16"),
        )
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(cfg["model"], **kwargs)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        enc = tokenize(self.tokenizer, texts, self.pooling, self.max_length).to(self.device)
        return embed_batch(self.model, enc, self.pooling, self.bidirectional)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs) -> np.ndarray:
        instruction = None
        if self.instructions and prompt_type == PromptType.query:
            instruction = self.get_instruction(task_metadata, prompt_type) or None
        texts = [str(t) for batch in inputs for t in batch["text"]]
        if instruction:
            texts = [INSTRUCTION_TEMPLATE.format(instruction=instruction, text=t) for t in texts]
        if self.doc_cache is not None and prompt_type == PromptType.document:
            keys = [DocEmbeddingCache.key(t) for t in texts]
            rows = self.doc_cache.get(keys)
            miss = [i for i, r in enumerate(rows) if r is None]
            out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
            hit = [i for i, r in enumerate(rows) if r is not None]
            if hit:
                out[hit] = self.doc_cache.emb[[rows[i] for i in hit]].astype(np.float32)
            if miss:
                emb = self._encode_texts([texts[i] for i in miss])
                out[miss] = emb
                self.doc_cache.put([keys[i] for i in miss], emb)
            self.doc_cache.hits += len(hit)
            self.doc_cache.misses += len(miss)
            return out
        return self._encode_texts(texts)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.token_budget:
            return self._encode_budget(texts)
        # Encode longest-first in length-sorted batches: minimal padding, and any OOM surfaces on the first batch,
        # where we halve the batch size and retry (it stays halved for the rest of the run).
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        s = 0
        while s < len(order):
            idx = order[s : s + self.batch_size]
            try:
                out[idx] = self.embed([texts[i] for i in idx]).cpu().numpy()
            except torch.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                torch.cuda.empty_cache()
                self.batch_size //= 2
                print(f"[OlmoEmbedder] CUDA OOM -> batch_size {self.batch_size}", flush=True)
                continue
            s += len(idx)
        return out

    def _encode_budget(self, texts: list[str]) -> np.ndarray:
        """Tokenize once, sort longest-first, and fill each batch up to token_budget padded tokens."""
        ids = token_ids(self.tokenizer, texts, self.pooling, self.max_length)
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        s = 0
        while s < len(order):
            longest = len(ids[order[s]])  # sorted: the first row is the longest in the batch
            idx = order[s : s + max(1, self.token_budget // longest)]
            try:
                enc = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt")
                with torch.no_grad():
                    emb = embed_batch(self.model, enc.to(self.device), self.pooling, self.bidirectional)
                out[idx] = emb.cpu().numpy()
            except torch.OutOfMemoryError:
                if self.token_budget <= self.max_length:
                    raise
                torch.cuda.empty_cache()
                self.token_budget //= 2
                print(f"[OlmoEmbedder] CUDA OOM -> token_budget {self.token_budget}", flush=True)
                continue
            s += len(idx)
        return out

    def similarity(self, embeddings1, embeddings2) -> torch.Tensor:
        """Cosine similarity on the GPU in float32 (same math as mteb's cos_sim; the CPU matmul of ~7k queries
        against each 50k-document chunk is otherwise minutes per chunk on full BEIR)."""
        a = torch.as_tensor(embeddings1, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(embeddings2, dtype=torch.float32, device=self.device)
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return (a @ b.T).cpu()
