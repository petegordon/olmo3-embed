"""Phase 3 training-data plumbing: Phase 2 batches unchanged, task-homogeneous temperature batching, in-batch
false-negative masking, per-task instructions (docs/phase3-plan.md §5).

    uv run python -m pytest tests/ -q
"""
import collections
import random

import pandas as pd
import torch

from olmo3_embed.train import Config, build_batches, false_negative_mask, infonce, query_text


def phase2_reference(cfg: Config) -> list[list[dict]]:
    """The Phase 2 build_batches, verbatim: the new code must reproduce it exactly (resume + comparability)."""
    df = pd.read_parquet(cfg.data).iloc[: cfg.n_examples]
    rng = random.Random(cfg.seed)
    rows = []
    for q, pos, negs in zip(df["query"], df["positive"], df["negatives"]):
        negs = list(negs)
        if len(negs) < cfg.num_negatives:
            continue
        rows.append({"query": q, "positive": pos, "negatives": rng.sample(negs, cfg.num_negatives)})
    rng.shuffle(rows)
    n = len(rows) // cfg.batch_size
    return [rows[i * cfg.batch_size : (i + 1) * cfg.batch_size] for i in range(n)]


def test_phase2_batches_unchanged(tmp_path):
    p = tmp_path / "p2.parquet"
    pd.DataFrame({"query": [f"q{i}" for i in range(100)], "positive": [f"p{i}" for i in range(100)],
                  "negatives": [[f"n{i}_{j}" for j in range(i % 7)] for i in range(100)]}).to_parquet(p)
    cfg = Config(data=str(p), n_examples=90, batch_size=8, num_negatives=3)
    new, ref = build_batches(cfg), phase2_reference(cfg)
    strip = lambda bs: [[(r["query"], r["positive"], r["negatives"]) for r in b] for b in bs]
    assert strip(new) == strip(ref)


def trackb(tmp_path, sizes: dict[str, int], n_neg: int = 10):
    rows = [{"query": f"{t}-q{i}", "task": t, "positives": [f"{t}-p{i}a", f"{t}-p{i}b"],
             "negatives": [f"{t}-n{i}_{j}" for j in range(n_neg)]} for t, n in sizes.items() for i in range(n)]
    p = tmp_path / "tb.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return str(p)


def test_task_batching_homogeneous_deterministic_and_tempered(tmp_path):
    sizes = {"MSMARCO": 4000, "NQ": 1000, "SciFact": 100}
    cfg = Config(data=trackb(tmp_path, sizes), n_examples=2000, batch_size=10, num_negatives=3, neg_pool=5,
                 task_batching=True, task_temperature=0.5)
    a, b = build_batches(cfg), build_batches(cfg)
    assert a == b  # deterministic -> resume reuses the same batches
    for batch in a:
        assert len({r["task"] for r in batch}) == 1
        for r in batch:
            assert r["positive"] in r["positives"]
            i = r["query"].split("-q")[1]
            assert all(int(n.split("_")[1]) < 5 for n in r["negatives"]) and all(f"-n{i}_" in n for n in r["negatives"])
    rows = collections.Counter(b[0]["task"] for b in a for _ in b)
    w = {t: n ** 0.5 for t, n in sizes.items()}
    for t in sizes:  # rows per task ∝ n ** T (within one batch of rounding)
        assert abs(rows[t] - 2000 * w[t] / sum(w.values())) <= cfg.batch_size


def test_small_task_upsampled_without_duplicates_until_exhausted(tmp_path):
    cfg = Config(data=trackb(tmp_path, {"A": 1000, "B": 20}), n_examples=1000, batch_size=5, num_negatives=2,
                 task_batching=True, task_temperature=0.0)  # uniform: B gets 500 rows from 20 -> repeats
    qs = collections.Counter(r["query"] for b in build_batches(cfg) for r in b if r["task"] == "B")
    assert sum(qs.values()) == 500 and set(qs.values()) == {25}  # whole reshuffled passes: every row used equally


def test_mixed_batches_when_task_batching_off(tmp_path):
    cfg = Config(data=trackb(tmp_path, {"A": 300, "B": 300}), n_examples=0, batch_size=20, num_negatives=2)
    assert any(len({r["task"] for r in b}) > 1 for b in build_batches(cfg))


def test_false_negative_mask_and_masked_infonce():
    k = 1
    batch = [  # row 1's negative is row 0's second positive; row 2's positive is row 0's positive (shared corpus)
        {"task": "Q", "positive": "p0", "positives": ["p0", "p0b"], "negatives": ["x0"]},
        {"task": "Q", "positive": "p1", "positives": ["p1"], "negatives": ["p0b"]},
        {"task": "Q", "positive": "p0", "positives": ["p0"], "negatives": ["x2"]},
    ]
    m = false_negative_mask(batch, k)  # docs: p0 x0 p1 p0b p0 x2
    assert m[0].tolist() == [False, False, False, True, True, False]  # p0b and the other p0 masked, own target not
    assert m[1].tolist() == [False] * 6
    assert m[2].tolist() == [True, False, False, False, False, False]
    torch.manual_seed(0)
    q, d = torch.randn(3, 8), torch.randn(6, 8)
    loss, _ = infonce(q, d, k, 0.05, m)
    s = q @ d.T / 0.05
    ref = torch.stack([-(s[i, i * 2] - torch.logsumexp(s[i][~m[i]], 0)) for i in range(3)]).mean()
    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.isfinite(loss)


def test_query_text_modes():
    r = {"query": "who wrote hamlet", "task": "NQ"}
    assert query_text(r, Config(instructions=False)) == "who wrote hamlet"
    assert "web search query" in query_text(r, Config())  # fixed (Phase 2 default)
    per = query_text(r, Config(instruction_mode="per_task"))
    assert per.startswith("Instruct: Given a question, retrieve Wikipedia passages") and per.endswith("who wrote hamlet")


def test_upstream_negs_variant(tmp_path):
    import subprocess, sys
    r = pd.DataFrame({"qid": ["MSMARCO:1", "SciFact:2"], "task": ["MSMARCO", "SciFact"], "query": ["q1", "q2"],
                      "positives": [["p1"], ["p2"]], "negatives": [["rescored1"], ["mined2a", "mined2b"]]})
    d = pd.DataFrame({"qid": ["MSMARCO:1", "SciFact:2"], "negatives": [[f"up{i}" for i in range(40)], []]})
    r.to_parquet(tmp_path / "r.parquet"), d.to_parquet(tmp_path / "d.parquet")
    subprocess.run([sys.executable, "-m", "olmo3_embed.data.upstream_negs", "--rescored", str(tmp_path / "r.parquet"),
                    "--decontam", str(tmp_path / "d.parquet"), "--out", str(tmp_path / "o.parquet")], check=True)
    o = pd.read_parquet(tmp_path / "o.parquet").set_index("qid")
    assert list(o.at["MSMARCO:1", "negatives"]) == [f"up{i}" for i in range(30)]  # upstream, capped, source order
    assert list(o.at["SciFact:2", "negatives"]) == ["mined2a", "mined2b"]  # no upstream: mined list kept
