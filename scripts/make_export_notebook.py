"""Generates notebooks/phase9_export_hf_colab.ipynb: merge run #4, verify it through mteb's loader, upload to HF."""
import json
md = """# Export run #4 to Hugging Face (Phase 9)

One cell, no Drive (like Phase 8): bootstrap from the `model-p7-run4-7b` release, merge the LoRA into OLMo-3-7B and
save a Sentence Transformers model (`scripts/export_hf.py`), then verify it the way the leaderboard will load it
(`hf/model_meta.py`): embedding cosine vs. our encoder, and SciFact + NFCorpus scores vs. the run's own (79.47 /
38.98). Only if both scores match within 0.15 does it upload to `petegordon/olmo3-embed-7b` (token from
`/root/.olmo3_HF_TOKEN`, uploaded by `scripts/colab_hf_secret.sh`). Ends with `runtime.unassign()`."""
code = r'''import os, subprocess, time, glob, json
RUN = 'p7-run4-7b'; REPO = 'petegordon/olmo3-embed-7b'; OUT = '/content/hf/olmo3-embed-7b'
REF = {'SciFact': 79.47, 'NFCorpus': 38.98}
LOG = '/content/export.log'
def sh(cmd):
    with open(LOG, 'a') as f:
        f.write(f'\n$ {cmd}\n'); f.flush()
        return subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=f, stderr=subprocess.STDOUT).returncode
def log(msg):
    with open(LOG, 'a') as f: f.write(time.strftime('%F %T ') + msg + '\n')
    print(msg, flush=True)
try:
    log('export ' + RUN + ' -> ' + REPO)
    assert sh('curl -sSfL https://raw.githubusercontent.com/petegordon/olmo3-embed/main/scripts/colab_bootstrap.sh -o /content/bootstrap.sh'
              ' && bash /content/bootstrap.sh model-p7-run4-7b ' + RUN) == 0, 'bootstrap failed'
    os.chdir('/content/olmo3-embed')
    rc = sh(f'python scripts/export_hf.py --run runs/{RUN} --out {OUT} --verify SciFact NFCorpus 2>&1 | grep -vE "it/s]|Warning"')
    for line in open(LOG):
        if line.startswith(('exported', 'cosine', 'verify')): log(line.strip())
    got = {}
    for t in REF:
        f = glob.glob(f'results-hf-verify/results/*-verify/*/{t}.json')
        if f: got[t] = round(100 * json.load(open(f[0]))['scores']['test'][0]['ndcg_at_10'], 2)
    ok = rc == 0 and all(t in got and abs(got[t] - REF[t]) <= 0.15 for t in REF)
    log(f'verify {got} vs {REF}: ' + ('PASS' if ok else 'FAIL'))
    if ok and os.path.exists('/root/.olmo3_HF_TOKEN'):
        from huggingface_hub import HfApi
        api = HfApi(token=open('/root/.olmo3_HF_TOKEN').read().strip())
        api.create_repo(REPO, exist_ok=True)
        info = api.upload_folder(repo_id=REPO, folder_path=OUT, commit_message='olmo3-embed-7b (run #4, p7-run4-7b): merged LoRA, Sentence Transformers format')
        log(f'UPLOADED {REPO} commit {info.oid}')
    log('EXPORT_DONE' if ok else 'EXPORT_FAILED')
finally:
    if not os.path.exists('/root/.olmo3_KEEP'):
        from google.colab import runtime
        runtime.unassign()
'''
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open('notebooks/phase9_export_hf_colab.ipynb', 'w'), indent=1)
