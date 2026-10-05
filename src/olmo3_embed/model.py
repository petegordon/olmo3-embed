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

def tokenize(tokenizer, texts: list[str], pooling: str, max_length: int) -> dict[str, torch.Tensor]:
    eos = tokenizer.eos_token_id
    if pooling == "mean":
        ids = tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]
        # Empty docs exist (NanoFiQA2018 has 27) and OLMo adds no BOS, so they'd be 0 tokens; a batch of only
        # empties (length-sorted batching puts them together) then crashes attention. Encode them as [EOS].
        ids = [x or [eos] for x in ids]
    else:
        # last-token: truncate leaving room for EOS, append it, then pad on the right
        ids = tokenizer(texts, truncation=True, max_length=max_length - 1)["input_ids"]
        ids = [x + [eos] for x in ids]
    return tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")


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
    ):
        if pooling not in ("mean", "last"):
            raise ValueError(pooling)
        self.pooling, self.instructions, self.bidirectional = pooling, instructions, bidirectional
        self.max_length = max_length
        self.batch_size = batch_size  # adaptive: halved on CUDA OOM
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
