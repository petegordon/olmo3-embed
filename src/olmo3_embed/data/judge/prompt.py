"""One prompt and verdict format shared by every judge backend, so verdicts are comparable across models.

The judge sees the query, one known positive (as an anchor for what "relevant" means in this task) and up to ~25
candidates, and returns {"verdicts": [{"id": ..., "relevant": true|false}, ...]}.
"""
import hashlib
import json
import re

MAX_CHARS = 1200  # per document; candidates are passages/abstracts, the head carries the relevance signal

DEFINITIONS = {
    "QuoraRetrieval": "Relevant means the candidate question asks the same thing as the query: a duplicate with the "
                      "same intent, whose answer would also answer the query. Related, broader or narrower questions "
                      "are NOT relevant.",
    "SciFact": "Relevant means the candidate abstract provides evidence that supports or refutes the scientific claim.",
    "NFCorpus": "Relevant means the candidate document directly addresses the health or nutrition question, the way "
                "the known positive does.",
    "default": "Relevant means the candidate would be a correct search result for the query, as good as the known "
               "positive.",
}

TEMPLATE = """You are checking retrieval training data for false negatives.

Task: {task}. {definition}

QUERY:
{query}

KNOWN POSITIVE (relevant, for reference):
{positive}

For each CANDIDATE below, decide whether it is relevant to the QUERY by the definition above. Judge each candidate on
its own; most candidates are NOT relevant. Reply with JSON only, no prose:
{{"verdicts": [{{"id": "<id>", "relevant": true or false}}, ...]}}

CANDIDATES:
{candidates}"""


def clip(t: str) -> str:
    t = " ".join(t.split())
    return t if len(t) <= MAX_CHARS else t[:MAX_CHARS] + " …"


def build(task: str, query: str, positive: str, items: list[dict]) -> str:
    cands = "\n\n".join(f'[id: {it["id"]}]\n{clip(it["candidate"])}' for it in items)
    return TEMPLATE.format(task=task, definition=DEFINITIONS.get(task, DEFINITIONS["default"]), query=clip(query),
                           positive=clip(positive), candidates=cands)


PROMPT_SHA = hashlib.sha1((TEMPLATE + json.dumps(DEFINITIONS, sort_keys=True)).encode()).hexdigest()[:12]


def parse(text: str, ids: list[str]) -> dict[str, bool]:
    """Verdicts for the expected ids. Tolerates code fences and prose around the JSON; unknown ids are ignored,
    missing ids are simply absent (the caller retries them)."""
    obj, dec = None, json.JSONDecoder()
    for m in re.finditer(r'\{\s*"verdicts"', text):  # the last decodable {"verdicts": ...} object wins
        try:
            obj = dec.raw_decode(text, m.start())[0]
        except json.JSONDecodeError:
            continue
    if obj is None:
        return {}
    out = {}
    for v in obj.get("verdicts", []) if isinstance(obj, dict) else []:
        if isinstance(v, dict) and str(v.get("id")) in ids and isinstance(v.get("relevant"), bool):
            out[str(v["id"])] = v["relevant"]
    return out
