"""mteb ModelMeta for the published model (the code proposed to embeddings-benchmark/mteb).

Also used by scripts/export_hf.py --verify to load the exported folder exactly the way the leaderboard will.
"""
from __future__ import annotations

import torch
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta

HF_REPO = "petegordon/olmo3-embed-7b"

# Every training source that is (the train split of) an MTEB task. Nothing from the remaining BEIR tasks (ArguAna,
# ClimateFEVER, CQADupStack, DBPedia, SCIDOCS, TREC-COVID, Touche2020) was trained on; see docs/data-card.md.
training_datasets = {
    "MSMARCO", "HotpotQA", "NQ", "FEVER", "FiQA2018", "NFCorpus", "SciFact", "QuoraRetrieval",
    "NanoMSMARCORetrieval", "NanoHotpotQARetrieval", "NanoNQRetrieval", "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval", "NanoNFCorpusRetrieval", "NanoSciFactRetrieval", "NanoQuoraRetrieval",
}

olmo3_embed_7b = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template="Instruct: {instruction}\nQuery: ",
        apply_instruction_to_passages=False,
        max_seq_length=512,
        model_kwargs={"torch_dtype": torch.bfloat16},
    ),
    name=HF_REPO,
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="ad1008e0c5857f40e472077b7ff43450f89b66d9",
    release_date="2026-10-14",
    n_parameters=6_887_272_448,  # Olmo3Model without the LM head
    n_embedding_parameters=410_738_688,
    memory_usage_mb=13137,
    embed_dim=4096,
    max_tokens=512,
    license="apache-2.0",
    reference=f"https://huggingface.co/{HF_REPO}",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/petegordon/olmo3-embed",
    public_training_data="https://github.com/petegordon/olmo3-embed/blob/main/docs/data-card.md",
    training_datasets=training_datasets,
    adapted_from="allenai/Olmo-3-1025-7B",
)
