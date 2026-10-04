"""Diff our local MTEB results against the published leaderboard results.

    uv run python scripts/compare_published.py [--tol 0.5]

Ours:      results/**/<model>/<revision>/<Task>.json   (written by olmo3_embed.eval)
Published: ~/.cache/mteb/remote/results/<model>/*/<Task>.json   (embeddings-benchmark/results clone)
Scores are nDCG@10 x 100. Tasks with no published counterpart (e.g. NanoBEIR) are listed as ours-only.
"""
import argparse
import glob
import json
import os

PUBLISHED = os.path.expanduser("~/.cache/mteb/remote/results")


def ndcg(path):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for split in ("test", "dev", "train"):  # NanoBEIR stores its eval under "train"
        for e in d.get("scores", {}).get(split, []):
            if e.get("hf_subset") in ("default", "en"):
                return 100 * e.get("ndcg_at_10", e.get("main_score"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="results")
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args()

    ours = {}
    for p in glob.glob(os.path.join(args.ours, "**", "*.json"), recursive=True):
        task = os.path.basename(p)[:-5]
        model = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if task == "model_meta" or (v := ndcg(p)) is None:
            continue
        ours[(model, task)] = v

    fails = 0
    for model in sorted({m for m, _ in ours}):
        print(f"\n{model}")
        print(f"  {'task':<28}{'ours':>8}{'published':>11}{'delta':>8}")
        nano = []
        for (m, task), v in sorted(ours.items()):
            if m != model:
                continue
            pubs = [x for p in glob.glob(os.path.join(PUBLISHED, model, "*", task + ".json")) if (x := ndcg(p)) is not None]
            if task.startswith("Nano"):
                nano.append(v)
            if not pubs:
                print(f"  {task:<28}{v:8.2f}{'-':>11}")
                continue
            pub = min(pubs, key=lambda x: abs(x - v))  # several revisions may be published; take the closest
            delta = v - pub
            flag = "" if abs(delta) <= args.tol else "  <-- off"
            fails += bool(flag)
            print(f"  {task:<28}{v:8.2f}{pub:11.2f}{delta:+8.2f}{flag}")
        if nano:
            print(f"  {'NanoBEIR avg (' + str(len(nano)) + ' tasks)':<28}{sum(nano) / len(nano):8.2f}")
    print(f"\n{'PASS' if not fails else f'{fails} task(s) outside ±{args.tol}'}")


if __name__ == "__main__":
    main()
