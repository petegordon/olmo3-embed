# Colab ops: Hugging Face storage, Colab MCP and a job launcher (plan)

**Status: PLANNED 2026-10-05, not implemented.** Blocked on a write-scoped HF token (the current `HF_TOKEN` is
read-only). Until then, Phase 3 keeps the current Drive-based notebooks.

## Problems this solves (from Phases 1–2)
1. **Drive OAuth on every new runtime.** `drive.mount` asks for consent each time, and once it timed out
   (`mount failed`) while the consent was still in progress.
2. **Notebook edits mean reloading from GitHub.** That brings "Leave site?" dialogs, the "not authored by Google"
   warning, duplicate tabs, and stray runtimes.
3. **Driving Colab by screenshots.** Outputs live in cross-origin iframes, so progress is read from images.

## Design

### 1. Storage: private Hugging Face repos instead of Google Drive
| What | Repo (private) | Written by | Read by |
|---|---|---|---|
| Checkpoints + final adapters | `petegordon/olmo3-embed-runs` (model repo), one folder per `run_name` | `train.py` every `save_every` steps and at the end | `train.py` resume; `eval.py` |
| Training data (parquet) | `petegordon/olmo3-embed-data` (dataset repo) | the data scripts (laptop or Colab), once per data version | `train.py` at start (cached in `/content/hf_cache`) |
| Results | stays on the `colab-results` git branch | `scripts/push_results.sh` | Claude, via `git fetch` |

- **Auth:** the `HF_TOKEN` Colab secret. Secrets don't need an OAuth prompt, so no per-runtime consent.
- **Token:** needs a **fine-grained token with write access to just these two repos**, not a classic write token.
- **Sizes:**
  - 1B LoRA adapter ≈ 50 MB, plus optimizer state ≈ 150 MB per checkpoint.
  - 7B ≈ 200–600 MB per checkpoint.
  - Keep only the latest checkpoint and the final adapter per run (`delete_patterns` on upload) to bound repo size.
- **Upload timing:** `upload_folder` runs on a background thread, so training doesn't wait on the network. A
  failed upload is logged and retried at the next save. Training only fails if the *final* upload fails.
- **Resume:**
  - `train.py --resume auto` checks local `runs/<name>/ckpt` first, then `snapshot_download` from the runs repo.
  - Same semantics as today: a lost runtime costs at most `save_every` steps.
- **Code changes:**
  - A small `olmo3_embed/storage.py` with `push_ckpt`, `pull_ckpt` and `pull_data`, behind a `storage: local | hf`
    config key. Local laptop runs keep `storage: local` and are unchanged.
  - Update `push_results.sh` to read adapters from HF when needed.

### 2. Control: Google's official Colab MCP server
[`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp) lets Claude Code run cells in an open Colab
session and read outputs as text.
- **Install** (project `.mcp.json`; the session needs a restart to load it):
  ```json
  {"mcpServers": {"colab-mcp": {"command": "uvx", "args": ["git+https://github.com/googlecolab/colab-mcp"], "timeout": 30000}}}
  ```
- **Limitation:** it attaches to a Colab tab that is already open. It doesn't create runtimes on its own. Per
  session, Pete (or Claude via Chrome) opens the launcher notebook and connects an A100 or H100 once. After that,
  everything goes through the MCP server.
- **To verify at setup:** the server needs the client to support `notifications/tools/list_changed`. If this
  Claude Code build doesn't, fall back to Chrome automation; storage (§1) and the launcher (§3) still remove most
  of the friction.

### 3. One stable launcher notebook
`notebooks/colab_launcher.ipynb` replaces the per-phase notebooks for new work, and it never changes per experiment:
1. **Setup cell:**
   - GPU check
   - read secrets (`GH_TOKEN`, `HF_TOKEN`)
   - clone or pull the repo
   - install packages, and uninstall Colab's torchao
   - no Drive mount
2. **`run_job(name)` cell:** runs `python -m olmo3_embed.jobs jobs/<name>.yaml`.
   - A job YAML lists steps: train, eval, push results, and later judge/vLLM.
   - It's idempotent: finished steps are skipped, and interrupted steps resume.
3. **New experiments mean a new `jobs/*.yaml` committed and pushed**, then `run_job` via the MCP server. That's no
   notebook edit, no reload, and no GitHub warning.
4. **vLLM jobs** (the Phase 3 OLMo judge) run in a separate runtime, because vLLM's torch pin can clash with the
   training stack. The job YAML sets `runtime: vllm`, and the setup cell installs the matching stack.

## Rollout
1. **Pete:** create a fine-grained HF token with write access to `petegordon/olmo3-embed-runs` and
   `petegordon/olmo3-embed-data`, then replace the Colab `HF_TOKEN` secret with it. Claude creates the repos (private)
   with that token.
2. **Claude:** write `storage.py` and its tests, with a laptop round-trip (push, wipe locally, resume from HF, and
   check the next step's loss matches, like the Phase 2 resume test).
3. **Claude:** write `jobs.py`, the launcher notebook and the first job (re-run a Phase 2 arm to check the result
   matches a8 within noise).
4. **Claude:** add `.mcp.json`, and Pete restarts the session. Verify the MCP server works in this Claude Code build.
5. Move the next Colab work (Phase 3 B-runs) to the launcher. Leave the Phase 1/2 notebooks as historical records.

## When to go fully headless instead
Colab Pro still needs a browser tab during runs. Before Phase 4's multi-hour 7B runs, compare:

| Option | Background execution? | A100 cost | Notes |
|---|---|---|---|
| Colab Pro (current) | No | ≈ $0.53/h, from 361 units on hand | |
| Colab Pro+ | Yes | $49.99/mo | Still needs a tab to *start* a run |
| Colab Enterprise | Fully API-driven | Google Cloud rates | Consumer compute units don't apply |
| Modal / RunPod / Lambda | Fully headless | ≈ $1.5–4/h for A100/H100 | Launched from Claude Code by CLI |

The storage layer (§1) and job YAMLs (§3) carry over unchanged to any of them.
