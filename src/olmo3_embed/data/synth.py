"""Synthetic passage -> query data for the held-out task shapes (docs/task-variety-plan.md, sources B).

  sample    python -m olmo3_embed.data.synth sample --shape argument --n 8000 --out data/synth/passages_argument.jsonl
  generate  python -m olmo3_embed.data.synth generate --passages data/synth/passages_argument.jsonl --backend claude \
                [--model M] [--part 0 --parts 4] --out data/synth/argument_claude.jsonl
  collect   python -m olmo3_embed.data.synth collect data/synth/*_*.jsonl --out data/variety_synth/ingested.parquet

Real passages from vetted, non-test sources (license on each source's card); an LLM writes the query in the target
task's shape. Hard negatives are NOT generated: rescore.py mines them from each task's own pool of positives.
--part/--parts give each generator a disjoint slice of the same passage sample, so generators can be compared on
the 1B proxy without seeing the same passages. Every output row records generator, model and prompt hash.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re

from datasets import load_dataset

SHAPES = {
    # shape: (task name, passage source, instruction to the LLM)
    "argument": ("SynthArgument", "biglam/hansard_speech",
                 "Write the general debate question this passage takes a side on: a short, neutral, controversial "
                 "yes/no question (4-12 words) at the level of a public policy or social issue, like a user would type "
                 "into a debate search engine (e.g. 'Should zoos be banned?', 'Is nuclear power a good energy "
                 "source?'). Never mention specific bills, people, places, dates or parliamentary procedure. If a "
                 "passage takes no clear side on any such issue, still write the closest general question."),
    "counterargument": ("SynthCounterArgument", "biglam/hansard_speech",
                        "Write the argument this passage directly rebuts: 2-4 sentences stating the opposing position "
                        "the speaker is answering, written as that position's own advocate would argue it. Don't "
                        "mention the speaker or the passage."),
    "claim": ("SynthClimateClaim", "wikimedia/wikipedia",
              "Write one short factual claim (8-25 words) about climate, environment or earth science, in the style of "
              "a claim seen in news or on social media, that this passage SUPPORTS or REFUTES. Across the batch, make "
              "roughly half of the claims ones the passage refutes. Don't copy sentences from the passage."),
    "entity": ("SynthEntitySearch", "wikimedia/wikipedia",
               "Write a keyword-style entity search query (2-8 lowercase words, not a question) whose best answer is "
               "the entity this passage describes, using its type and distinguishing attributes, e.g. 'british "
               "politician abolition of slavery' or 'szechwan dish chili peppercorn'. Never include the entity's own "
               "name."),
    "citation": ("SynthCitation", "common-pile/arxiv_abstracts",
                 "Write the title of a different, plausible later scientific paper that would cite this abstract's "
                 "work. Only the title (6-20 words), in the style of real paper titles."),
}
CLIMATE = re.compile(r"\b(climate|global warming|greenhouse|carbon dioxide|emissions?|sea[- ]level|glacier|ice sheet|"
                     r"ocean acidification|deforestation|drought|el ni[nñ]o|methane|ozone|permafrost|heatwave|wildfire)\b",
                     re.I)
CLIMATE_TOPIC = re.compile(r"\b(climate change|global warming|greenhouse gas(es)?|climate science|climate model|"
                           r"carbon emissions?|carbon dioxide emissions?|sea[- ]level rise|ocean acidification|"
                           r"ice sheets?|paleoclimat\w*|climatolog\w*|kyoto protocol|paris agreement|ipcc|"
                           r"renewable energy|fossil fuels?|carbon capture)\b", re.I)
TEMPLATE = """You are creating training data for a search engine.

{instruction}

Reply with JSON only, no prose: {{"items": [{{"id": "<id>", "query": "<text>"}}, ...]}} with one item per passage.

PASSAGES:
{passages}"""
PROMPT_CHARS = 1800  # passage text shown to the LLM (the full passage is kept as the training positive)


def clean(t: str) -> str:
    return " ".join((t or "").split())


def sample(shape: str, n: int, seed: int):
    _, src, _ = SHAPES[shape]
    if src == "biglam/hansard_speech":
        ds = load_dataset(src, split="train", streaming=True).shuffle(seed=seed, buffer_size=50_000)
        for r in ds:
            t = clean(r["speech"])
            # substantive debate speeches, not question-time one-liners or points of order
            if r.get("speech_class") == "Speech" and 900 <= len(t) <= 2500 and "point of order" not in t[:200].lower():
                yield {"passage": t, "meta": {"date": r["date"], "heading": r["major_heading"], "hansard_id": r["id"]}}
    elif src == "wikimedia/wikipedia":
        yield from sample_wikipedia(shape, seed)
    elif src == "common-pile/arxiv_abstracts":
        ds = load_dataset(src, split="train", streaming=True).shuffle(seed=seed, buffer_size=50_000)
        for r in ds:
            t = clean(r["text"])
            if 400 <= len(t) <= 1800:
                yield {"passage": t, "meta": {"arxiv_id": r["id"]}}


def sample_wikipedia(shape: str, seed: int):
    """From the local English Wikipedia parquet ($OLMO3_RAW/wikipedia, one-time ~11.6 GB download): files in seeded
    random order, rows shuffled per batch. Vectorized filters, since climate articles are a tiny fraction (streaming +
    per-row regex was far too slow). entity: lead paragraph. claim: climate-topic articles (strong phrase in title or
    lead), first climate-mentioning paragraph."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(os.environ.get("OLMO3_RAW", "data/raw"), "wikipedia", "20231101.en", "*.parquet")))
    if not files:
        raise SystemExit("no local Wikipedia parquet: download wikimedia/wikipedia 20231101.en to $OLMO3_RAW/wikipedia")
    rng = random.Random(seed)
    rng.shuffle(files)
    for fpath in files:
        for b in pq.ParquetFile(fpath).iter_batches(columns=["id", "title", "text"], batch_size=50_000):
            df = b.to_pandas()
            lead = df["text"].str.split("\n").map(lambda ps: next((" ".join(p.split()) for p in ps if len(p.strip()) >= 200), ""))
            df = df.assign(lead=lead)[lead.str.len().between(200, 1500)]
            if shape == "claim":
                df = df[(df["title"] + " " + df["lead"]).str.contains(CLIMATE_TOPIC)]
            for r in df.sample(frac=1, random_state=rng.randrange(2**31)).itertuples(index=False):
                if shape == "entity":
                    yield {"passage": f"{r.title} {r.lead}", "meta": {"wiki_id": r.id, "title": r.title}}
                else:  # up to 3 climate paragraphs per article: only ~7.4k English articles pass the topic filter
                    hits = [clean(p) for p in r.text.split("\n") if 200 <= len(clean(p)) <= 1500 and CLIMATE.search(p)]
                    for k, p in enumerate(hits[:3]):
                        yield {"passage": f"{r.title} {p}", "meta": {"wiki_id": r.id, "title": r.title, "para": k}}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(shape: str, items: list[dict]) -> str:
    ps = "\n\n".join(f"[id: {it['id']}]\n{it['passage'][:PROMPT_CHARS]}" for it in items)
    return TEMPLATE.format(instruction=SHAPES[shape][2], passages=ps)


PROMPT_SHA = hashlib.sha1((TEMPLATE + json.dumps({k: v[2] for k, v in SHAPES.items()}, sort_keys=True)).encode()
                          ).hexdigest()[:12]


def parse(text: str, ids: list[str]) -> dict[str, str]:
    dec, obj = json.JSONDecoder(), None
    for m in re.finditer(r'\{\s*"items"', text):  # the last decodable {"items": ...} wins (CLIs may echo the prompt)
        try:
            obj = dec.raw_decode(text, m.start())[0]
        except json.JSONDecodeError:
            continue
    out = {}
    for it in (obj or {}).get("items", []):
        q = clean(str(it.get("query", ""))) if isinstance(it, dict) else ""
        if isinstance(it, dict) and str(it.get("id")) in ids and 3 <= len(q) <= 600 and q != "<text>":
            out[str(it["id"])] = q
    return out


def cmd_sample(a):
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in sample(a.shape, a.n, a.seed):
            if n >= a.n:
                break
            f.write(json.dumps({"id": f"{a.shape}-{n}", "shape": a.shape, **r}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} {a.shape} passages -> {a.out}")


def cmd_generate(a):
    from olmo3_embed.data.judge.backends import BACKENDS
    items = read_jsonl(a.passages)[a.part :: a.parts]  # disjoint slice per generator
    done = {r["id"] for r in read_jsonl(a.out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} passages in part {a.part}/{a.parts}, {len(done)} done, {len(todo)} to go ({a.backend})", flush=True)
    if not todo:
        return
    backend = BACKENDS[a.backend](**({"model": a.model} if a.model else {}))
    model = getattr(backend, "model", None) or a.backend
    groups = [todo[i : i + a.batch] for i in range(0, len(todo), a.batch)]
    with open(a.out, "a", encoding="utf-8") as f:
        for s in range(0, len(groups), a.chunk):
            chunk = groups[s : s + a.chunk]
            for g, raw in zip(chunk, backend([build_prompt(g[0]["shape"], g) for g in chunk])):
                got = parse(raw, [it["id"] for it in g])
                for it in g:
                    if it["id"] in got:
                        f.write(json.dumps({"id": it["id"], "shape": it["shape"], "passage": it["passage"],
                                            "query": got[it["id"]], "meta": it.get("meta"), "generator": a.backend,
                                            "model": model, "prompt": PROMPT_SHA}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(s + a.chunk, len(groups))}/{len(groups)} prompts", flush=True)


def cmd_collect(a):
    import pandas as pd
    rows = []
    for p in sorted(set(x for pat in a.inputs for x in glob.glob(pat))):
        for r in read_jsonl(p):
            task, src, _ = SHAPES[r["shape"]]
            rows.append({"qid": f"{task}:{r['generator']}:{r['id']}", "task": task, "source": src,
                         "query": r["query"], "positives": [r["passage"]], "negatives": [],
                         "generator": r["generator"], "model": r["model"]})
    df = pd.DataFrame(rows).drop_duplicates("qid")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out, index=False, row_group_size=20_000)
    print(df.groupby(["task", "generator"]).size().to_string(), f"\ntotal {len(df)} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--shape", choices=sorted(SHAPES), required=True)
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--passages", required=True)
    g.add_argument("--backend", required=True)
    g.add_argument("--model")
    g.add_argument("--part", type=int, default=0)
    g.add_argument("--parts", type=int, default=1)
    g.add_argument("--batch", type=int, default=8, help="passages per prompt (8 x 1800 chars fits agy's argv limit)")
    g.add_argument("--chunk", type=int, default=8, help="prompts per backend call (vllm batches them)")
    g.add_argument("--out", required=True)
    c = sub.add_parser("collect")
    c.add_argument("inputs", nargs="+")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sample": cmd_sample, "generate": cmd_generate, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
