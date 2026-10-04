# olmo3-embed

A fully open text-embedding model built on [OLMo 3](https://huggingface.co/allenai) and aimed at the
[MTEB BEIR benchmark](https://mteb-leaderboard.hf.space/benchmark/BEIR).

- Training runs on Google Colab (A100/H100). The dev loop and ablations run on a local RTX 5070 Ti (12 GB).
- See **[PLAN.md](PLAN.md)** for the recipe, targets, phases and open decisions.
- Leaderboard snapshot: `docs/beir_snapshot_2026-10-04.csv`, regenerated with `uv run scripts/beir_snapshot.py`.

Status: Phase 0 (scaffold).
