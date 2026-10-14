"""Score a corpus-sharded retrieval run (see src/olmo3_embed/shards.py) with mteb, as one full-corpus result.

    python scripts/merge_shards.py --task MSMARCO --meta results/results/olmo3-embed__p7-run4-7b/main/model_meta.json \\
        shard0.json.gz shard1.json.gz shard2.json.gz

The result lands where a single run would put it (results/results/<model>/<revision>/<Task>.json).
"""
import argparse
import json

import mteb
from mteb.models.model_meta import ModelMeta

from olmo3_embed.shards import MergedSearch

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--meta", required=True, help="model_meta.json of the model the shards came from")
ap.add_argument("--out", default="results")
ap.add_argument("shards", nargs="+")
args = ap.parse_args()
meta = json.load(open(args.meta))
meta["loader"] = None  # saved as a name string; not needed to score saved rankings
model = MergedSearch(ModelMeta.model_validate(meta), args.shards)
mteb.evaluate(model, mteb.get_tasks(tasks=[args.task], languages=["eng"]), cache=mteb.ResultCache(args.out),
              overwrite_strategy="only-missing")
