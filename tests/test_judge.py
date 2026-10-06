"""Judge plumbing: prompt/verdict parsing, resumable judging, cascade filtering, agreement report (no real LLM).

    uv run python -m pytest tests/ -q
"""
import json
from argparse import Namespace

from olmo3_embed.data.judge import backends, run
from olmo3_embed.data.judge.prompt import build, parse


def test_parse_tolerates_fences_and_ignores_unknown_ids():
    raw = 'Sure:\n```json\n{"verdicts": [{"id": "a", "relevant": true}, {"id": "zz", "relevant": true},' \
          ' {"id": "b", "relevant": "yes"}]}\n```'
    assert parse(raw, ["a", "b"]) == {"a": True}  # "yes" isn't a bool -> b left for retry
    assert parse("no json here", ["a"]) == {}
    # an echoed prompt (whose template has braces and a placeholder verdicts object) before the real answer
    echoed = build("QuoraRetrieval", "q", "p", [{"id": "a", "candidate": "c"}]) + \
        '\ncodex\n{"verdicts": [{"id": "a", "relevant": false}]}\ntokens used 1,890'
    assert parse(echoed, ["a"]) == {"a": False}


def test_build_contains_ids_and_clips():
    p = build("QuoraRetrieval", "q?", "pos", [{"id": "x#0", "candidate": "c" * 5000}])
    assert "[id: x#0]" in p and "duplicate" in p and len(p) < 3000


class Fake:
    model = "fake-1"
    calls = 0

    def __init__(self, **kw):
        pass

    def __call__(self, prompts):
        out = []
        for p in prompts:
            Fake.calls += 1
            ids = [line.split("[id: ")[1].rstrip("]") for line in p.splitlines() if line.startswith("[id: ")]
            # "relevant" iff the candidate id ends in #0; drop the last id of the very first prompt (retry path)
            vs = [{"id": i, "relevant": i.endswith("#0")} for i in ids]
            out.append(json.dumps({"verdicts": vs[:-1] if Fake.calls == 1 else vs}))
        return out


def write_pairs(path, n_q=3, per=4):
    with open(path, "w", encoding="utf-8") as f:
        for q in range(n_q):
            for r in range(per):
                f.write(json.dumps({"id": f"Q:{q}#{r}", "qid": f"Q:{q}", "task": "QuoraRetrieval", "query": f"q{q}",
                                    "positive": "p", "candidate": f"c{q}{r}", "cand_score": 0.8, "pos_score": 0.9})
                        + "\n")


def test_judge_resumes_and_cascade(tmp_path, monkeypatch):
    monkeypatch.setitem(backends.BACKENDS, "fake", Fake)
    monkeypatch.setitem(run.BACKENDS, "fake", Fake)
    pairs, out = tmp_path / "pairs.jsonl", tmp_path / "v.jsonl"
    write_pairs(pairs)
    a = Namespace(pairs=str(pairs), backend="fake", model=None, out=str(out), only_flagged=None, batch=25, chunk=2,
                  calibration=0.0)
    run.judge(a)
    first = run.read_jsonl(out)
    assert len(first) == 11  # one id unparsed on the first call
    run.judge(a)  # resume: only the missing id is re-judged
    vs = run.read_jsonl(out)
    assert len(vs) == 12 and len({v["id"] for v in vs}) == 12
    assert {v["id"] for v in vs if v["relevant"]} == {"Q:0#0", "Q:1#0", "Q:2#0"}
    assert all(v["model"] == "fake-1" and v["prompt"] for v in vs)
    # cascade: a second judge sees only what the first flagged
    out2 = tmp_path / "v2.jsonl"
    run.judge(Namespace(**{**vars(a), "out": str(out2), "only_flagged": str(out)}))
    assert {v["id"] for v in run.read_jsonl(out2)} == {"Q:0#0", "Q:1#0", "Q:2#0"}


def test_kappa():
    assert run.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(run.kappa([True, True, False, False], [True, False, True, False])) < 1e-9
