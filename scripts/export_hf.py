"""Export a trained run as a standalone Sentence Transformers model (LoRA merged into OLMo-3) and verify it.

    python scripts/export_hf.py --run runs/p7-run4-7b --out /content/hf/olmo3-embed-7b [--verify SciFact NFCorpus]

- Weights: base + adapter merged (peft merge_and_unload), saved in bf16 safetensors.
- Tokenizer: OLMo tokenizers add no EOS; ours embeds the hidden state of an appended EOS (model.py, last-token
  pooling). The export bakes EOS into the tokenizer's post-processor, so a plain `tokenizer(text, truncation=True,
  max_length=512)` gives exactly our ids (511 text tokens + EOS).
- Modules: Transformer (max_seq_length 512) -> Pooling(lasttoken) -> Normalize.
- --verify: (1) cosine between our OlmoEmbedder and the export on sample queries/documents; (2) mteb scores of the
  export loaded through hf/model_meta.py (the leaderboard's loader) vs. our results for the same tasks.
"""
import argparse
import json
import os
import shutil
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--card", default="hf/README.md")
ap.add_argument("--verify", nargs="*", default=[], help="mteb tasks to re-score through the exported model")
ap.add_argument("--verify-out", default="results-hf-verify")
args = ap.parse_args()
cfg = json.load(open(os.path.join(args.run, "embed_config.json")))
assert cfg["pooling"] == "last" and not cfg["bidirectional"], "export supports causal last-token runs only"

from peft import PeftModel
from sentence_transformers import SentenceTransformer, models
from tokenizers import processors
from transformers import AutoModel, AutoTokenizer

if not os.path.exists(os.path.join(args.out, "modules.json")):
    tmp = args.out + "-hf"
    base = AutoModel.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, os.path.join(args.run, "adapter")).merge_and_unload()
    merged.save_pretrained(tmp, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    eos = tok.eos_token
    tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"$A {eos}", pair=f"$A {eos} $B:1 {eos}:1", special_tokens=[(eos, tok.eos_token_id)])
    tok.padding_side = "right"
    tok.save_pretrained(tmp)
    del base, merged
    st = SentenceTransformer(modules=[
        models.Transformer(tmp, max_seq_length=512, model_args={"torch_dtype": torch.bfloat16}),
        models.Pooling(4096, pooling_mode="lasttoken"),
        models.Normalize(),
    ])
    st.save(args.out, safe_serialization=True)
    shutil.rmtree(tmp)
    if os.path.exists(args.card):
        shutil.copy(args.card, os.path.join(args.out, "README.md"))
    print("exported", args.out, sorted(os.listdir(args.out)), flush=True)

if args.verify:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hf"))
    import mteb
    from model_meta import olmo3_embed_7b
    from mteb.types import PromptType

    from olmo3_embed.model import OlmoEmbedder

    meta = olmo3_embed_7b.model_copy(update={"name": olmo3_embed_7b.name + "-verify"})
    model = meta.loader(args.out, revision=None, **meta.loader_kwargs)
    model.mteb_model_meta = meta
    # (1) embedding agreement with our own encoder on real query/document text
    task = mteb.get_task("SciFact")
    ours = OlmoEmbedder.from_run(args.run, max_length=512, batch_size=16)
    q = ["0-dimensional biomaterials show inductive properties.", "ALDH1 expression is associated with poorer prognosis for breast cancer primary tumors."]
    d = ["Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development. " * 3, "Short doc."]
    from torch.utils.data import DataLoader
    from datasets import Dataset
    def enc(m, texts, pt):
        dl = DataLoader(Dataset.from_dict({"text": texts}), batch_size=8)
        return torch.as_tensor(m.encode(dl, task_metadata=task.metadata, hf_split="test", hf_subset="default", prompt_type=pt)).float()
    for pt, texts in ((PromptType.query, q), (PromptType.document, d)):
        a, b = enc(ours, texts, pt), enc(model, texts, pt)
        cos = torch.nn.functional.cosine_similarity(a, b).tolist()
        print(f"cosine ours vs export ({pt.value}): {[round(c, 4) for c in cos]}", flush=True)
    del ours
    torch.cuda.empty_cache()
    # (2) mteb scores through the leaderboard loader
    mteb.evaluate(model, mteb.get_tasks(tasks=args.verify, languages=["eng"]), cache=mteb.ResultCache(args.verify_out),
                  encode_kwargs={"batch_size": 32}, overwrite_strategy="only-missing")
    for t in args.verify:
        import glob
        f = glob.glob(f"{args.verify_out}/results/*-verify/*/{t}.json")[0]
        print(f"verify {t}: {100 * json.load(open(f))['scores']['test'][0]['ndcg_at_10']:.2f}", flush=True)
