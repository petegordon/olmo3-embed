"""GradCache must give the same LoRA gradients as plain full-batch backprop (tiny random OLMo-2, fp32, CPU).

    uv run python -m pytest tests/ -q
"""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from olmo3_embed.model import embed_batch, tokenize
from olmo3_embed.train import Config, chunked_embed, infonce


ARCHS = {
    "olmo2": "allenai/OLMo-2-0425-1B",  # the Phase 2-3 proxy
    "olmo3": "allenai/Olmo-3-1025-7B",  # the Phase 4 target: mixed sliding-window / full attention layers
}


def tiny_model(arch: str = "olmo2"):
    cfg = AutoConfig.from_pretrained(ARCHS[arch])
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 2, 64, 128
    cfg.num_attention_heads = cfg.num_key_value_heads = 4
    if getattr(cfg, "layer_types", None):  # OLMo-3: keep one of each layer type, with a window short enough to bite
        cfg.layer_types, cfg.sliding_window = ["sliding_attention", "full_attention"], 4
    torch.manual_seed(0)
    base = AutoModel.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False))
    return model


@pytest.mark.parametrize("arch", sorted(ARCHS))
@pytest.mark.parametrize("pooling,bidirectional", [("mean", False), ("last", False), ("mean", True), ("last", True)])
def test_gradcache_matches_full_batch(pooling, bidirectional, arch):
    tok = AutoTokenizer.from_pretrained(ARCHS[arch])
    tok.padding_side = "right"
    k = 2
    queries = ["what is a cat", "stock prices today", "how tall is everest"]
    docs = [d for i in range(3) for d in (f"answer {i} about things", f"wrong {i} a", f"also wrong {i} b c d")]
    cfg = Config(pooling=pooling, bidirectional=bidirectional, chunk_size=2, max_len_query=16, max_len_doc=16)

    # reference: one graph, full batch
    ref = tiny_model(arch)
    q = embed_batch(ref, tokenize(tok, queries, pooling, 16), pooling, bidirectional)
    d = embed_batch(ref, tokenize(tok, docs, pooling, 16), pooling, bidirectional)
    loss_ref, _ = infonce(q, d, k, 0.05)
    loss_ref.backward()
    g_ref = {n: p.grad.clone() for n, p in ref.named_parameters() if p.requires_grad}

    # GradCache: same init (seeded), chunked two-pass. chunked_embed moves to the model's device (CPU here).
    gc = tiny_model(arch)
    qe = chunked_embed(gc, tok, queries, cfg, 16).requires_grad_()
    de = chunked_embed(gc, tok, docs, cfg, 16).requires_grad_()
    loss_gc, _ = infonce(qe, de, k, 0.05)
    loss_gc.backward()
    chunked_embed(gc, tok, queries, cfg, 16, grad_out=qe.grad)
    chunked_embed(gc, tok, docs, cfg, 16, grad_out=de.grad)

    assert torch.allclose(loss_ref, loss_gc, atol=1e-5)
    # Compare whole-tensor relative error: chunking changes padding lengths, so fp32 kernels differ by ~1e-5 absolute,
    # which an element-wise atol flags on near-zero entries. A real GradCache bug gives O(1) relative error.
    for n, p in gc.named_parameters():
        if p.requires_grad:
            rel = ((p.grad - g_ref[n]).norm() / g_ref[n].norm().clamp_min(1e-12)).item()
            assert rel < 1e-4, f"{n}: relative grad error {rel:.2e}"
