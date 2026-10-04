"""OLMo embedding encoder for MTEB (and, later, the training forward pass).

Pooling:
  mean  average of last hidden states over non-pad tokens
  last  hidden state of an appended EOS token. OLMo tokenizers do NOT add EOS themselves, so we append it after
        truncation (otherwise "last token" is just the final word)

Instructions (queries only, same format as Qwen3/E5): "Instruct: {task instruction}\nQuery: {text}".
Documents never get an instruction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from transformers import AutoModel, AutoTokenizer

INSTRUCTION_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"


class OlmoEmbedder(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "last",
        instructions: bool = True,
        dtype: str = "bfloat16",
        max_length: int = 512,
        revision: str | None = None,
        device: str | None = None,
        load_in_4bit: bool = False,
    ):
        if pooling not in ("mean", "last"):
            raise ValueError(pooling)
        self.pooling, self.instructions, self.max_length = pooling, instructions, max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "right"
        kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )
            kwargs["device_map"] = {"": 0}
        self.model = AutoModel.from_pretrained(model_name, revision=revision, **kwargs).eval()
        if not load_in_4bit:
            self.model.to(self.device)
        sha = getattr(self.model.config, "_commit_hash", None) or revision or "main"
        tag = f"{pooling}{'-instr' if instructions else ''}{'-4bit' if load_in_4bit else ''}"
        # Results folder = results/<name>/<revision>/; one folder per variant so runs never collide.
        self.mteb_model_meta = ModelMeta(
            loader=None,
            name=f"olmo3-embed/{model_name.split('/')[-1]}-{tag}",
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

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.pooling == "mean":
            enc = self.tokenizer(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
            return enc.to(self.device)
        # last-token: truncate leaving room for EOS, append it, then pad on the right
        ids = self.tokenizer(texts, truncation=True, max_length=self.max_length - 1)["input_ids"]
        ids = [x + [self.tokenizer.eos_token_id] for x in ids]
        enc = self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        return enc.to(self.device)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        enc = self._tokenize(texts)
        hidden = self.model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"]
        if self.pooling == "mean":
            emb = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            emb = hidden[torch.arange(hidden.size(0), device=hidden.device), mask.sum(1) - 1]
        return torch.nn.functional.normalize(emb, dim=-1)

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs) -> np.ndarray:
        instruction = None
        if self.instructions and prompt_type == PromptType.query:
            instruction = self.get_instruction(task_metadata, prompt_type) or None
        texts = [str(t) for batch in inputs for t in batch["text"]]
        if instruction:
            texts = [INSTRUCTION_TEMPLATE.format(instruction=instruction, text=t) for t in texts]
        # Encode longest-first in length-sorted batches: minimal padding, and any OOM surfaces on the first batch.
        bs = inputs.batch_size or 32
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        out = np.empty((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            out[idx] = self.embed([texts[i] for i in idx]).cpu().numpy()
        return out
