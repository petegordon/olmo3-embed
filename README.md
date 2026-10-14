# olmo3-embed

A fully open text-embedding model built on [OLMo 3](https://huggingface.co/allenai) and aimed at the
[MTEB BEIR benchmark](https://mteb-leaderboard.hf.space/benchmark/BEIR).

- Training runs on Google Colab (A100/H100). The dev loop and ablations run on a local RTX 5070 Ti (12 GB).
- See **[PLAN.md](PLAN.md)** for the recipe, targets, phases and open decisions.
- **[docs/worklog.md](docs/worklog.md)** is the running diary; **[docs/petes-learning.md](docs/petes-learning.md)** explains terms and concepts as they come up.
- Leaderboard snapshot: `docs/beir_snapshot_2026-10-04.csv`, regenerated with `uv run scripts/beir_snapshot.py`.

## Status (2026-10-14)

Best model: 7B run #4 (`configs/7b-run4.yaml`), a LoRA adapter on OLMo-3-7B trained on human-written public data only,
decontaminated against the BEIR test sets (see [docs/data-card.md](docs/data-card.md)).

- **54.72** average nDCG@10 on all 15 BEIR tasks: rank 37 of 140 full-BEIR models in a 2026-10-04 snapshot of the
  [MTEB BEIR leaderboard](https://mteb-leaderboard.hf.space/benchmark/BEIR). Per-task scores are in
  [docs/worklog.md](docs/worklog.md); result files are on the `colab-results` branch.
- The adapter is release [`model-p7-run4-7b`](https://github.com/petegordon/olmo3-embed/releases/tag/model-p7-run4-7b).
  The Hugging Face upload and leaderboard submission are still to come.
