"""Merge trained runs by averaging their LoRA weight updates exactly (a "model soup" of adapters).

Each run's update is delta_i = scale_i * B_i @ A_i. Averaging A and B separately would not average the updates, so the
merged adapter stacks them instead: A = cat(w_i * scale_i * A_i) along the rank, B = cat(B_i) along the rank, with
scale 1 (lora_alpha = r = sum of ranks). Then B @ A = sum_i w_i * delta_i. CPU only, no base model needed.

    python -m olmo3_embed.merge --runs runs/a runs/b --out runs/a+b [--weights 0.5 0.5] [--run-name NAME]
"""
import argparse, json, os, shutil

import torch
from safetensors.torch import load_file, save_file


def merge(run_dirs: list[str], weights: list[float], out_dir: str, run_name: str) -> None:
    cfgs = [json.load(open(os.path.join(d, "adapter", "adapter_config.json"))) for d in run_dirs]
    embs = [json.load(open(os.path.join(d, "embed_config.json"))) for d in run_dirs]
    for k in ("model", "pooling", "bidirectional", "instructions", "instruction_mode"):
        if len({json.dumps(e.get(k)) for e in embs}) != 1:
            raise ValueError(f"runs differ in {k}: can't merge")
    for c in cfgs:
        if c.get("use_dora") or c.get("use_rslora") or c.get("rank_pattern") or c.get("alpha_pattern"):
            raise ValueError("only plain LoRA adapters are supported")
    sds = [load_file(os.path.join(d, "adapter", "adapter_model.safetensors")) for d in run_dirs]
    keys = sorted(sds[0])
    if any(sorted(sd) != keys for sd in sds):
        raise ValueError("adapters target different modules")
    merged = {}
    for k in keys:
        if ".lora_A." in k:
            merged[k] = torch.cat([sd[k].float() * w * c["lora_alpha"] / c["r"] for sd, w, c in zip(sds, weights, cfgs)], 0)
        elif ".lora_B." in k:
            merged[k] = torch.cat([sd[k].float() for sd in sds], 1)
        else:
            raise ValueError(f"unexpected tensor {k}")
    dtype = sds[0][keys[0]].dtype
    merged = {k: v.to(dtype).contiguous() for k, v in merged.items()}
    os.makedirs(os.path.join(out_dir, "adapter"), exist_ok=True)
    save_file(merged, os.path.join(out_dir, "adapter", "adapter_model.safetensors"))
    rank = sum(c["r"] for c in cfgs)
    json.dump({**cfgs[0], "r": rank, "lora_alpha": rank}, open(os.path.join(out_dir, "adapter", "adapter_config.json"), "w"), indent=2)
    json.dump({**embs[0], "run_name": run_name, "steps": None,
               "merged_from": [{"run": e["run_name"], "weight": w} for e, w in zip(embs, weights)]},
              open(os.path.join(out_dir, "embed_config.json"), "w"), indent=2)
    print(f"merged {len(run_dirs)} adapters (rank {rank}, {len(keys)} tensors) -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, help="default: equal weights summing to 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", help="default: basename of --out")
    a = ap.parse_args()
    w = a.weights or [1 / len(a.runs)] * len(a.runs)
    if len(w) != len(a.runs):
        ap.error("--weights must match --runs")
    merge(a.runs, w, a.out, a.run_name or os.path.basename(os.path.normpath(a.out)))


if __name__ == "__main__":
    main()
