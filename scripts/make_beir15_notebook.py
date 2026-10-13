"""Generates notebooks/phase8_beir15_colab.ipynb.  python scripts/make_beir15_notebook.py JOBS_JSON TOKEN_BUDGET"""
import json
md = """# Full BEIR-15 for 7B run #4 (Phase 8)

Evaluates the published run #4 adapter (release `model-p7-run4-7b`) on the 7 BEIR tasks not yet scored: MSMARCO, FEVER,
ClimateFEVER, HotpotQA, DBPedia, NQ, Touche2020. The other 8 were scored in Phase 7 (`p7-run4-7b` on `colab-results`).

**One cell, no Drive.** The whole job (setup, evals, results push, runtime shutdown) is a single execution, so the runtime
needs nothing from the client once it starts: close the laptop, and Colab's background execution carries on (Ultra).
The adapter comes from the GitHub release and results go to the `colab-results` branch with the GitHub token.

Jobs run on separate runtimes in parallel; set `JOB` (or upload `/root/.olmo3_JOB` from the Colab CLI). FEVER and
ClimateFEVER share a 5.4M-passage corpus, so they run in one job with the document cache: the corpus is embedded once.
The cell ends with `runtime.unassign()`, so the runtime stops itself even when no client is connected."""
code = r'''import os, subprocess, time, glob, json
JOB = open('/root/.olmo3_JOB').read().strip() if os.path.exists('/root/.olmo3_JOB') else 'A'  # browser: set A/B/C/D
JOBS = __JOBS__
TOKEN_BUDGET = __BUDGET__
RUN = 'p7-run4-7b'
LOG = '/content/beir15.log'
def sh(cmd):  # stream to the cell and to LOG (download LOG with `colab download` to check progress)
    with open(LOG, 'a') as f:
        f.write(f'\n$ {cmd}\n'); f.flush()
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=f, stderr=subprocess.STDOUT)
    return p.returncode
def log(msg):
    with open(LOG, 'a') as f: f.write(time.strftime('%F %T ') + msg + '\n')
    print(msg, flush=True)
try:
    tok = '/root/.olmo3_GH_TOKEN'
    os.environ['GH_TOKEN'] = open(tok).read().strip() if os.path.exists(tok) else __import__('google.colab.userdata', fromlist=['get']).get('GH_TOKEN')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    gpu = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip()
    log(f'JOB {JOB}: {JOBS[JOB]} on {gpu}')
    assert sh('curl -sSfL https://raw.githubusercontent.com/petegordon/olmo3-embed/main/scripts/colab_bootstrap.sh -o /content/bootstrap.sh'
              ' && bash /content/bootstrap.sh model-p7-run4-7b ' + RUN) == 0, 'bootstrap failed'
    os.chdir('/content/olmo3-embed')
    sh('git remote set-url origin https://x-access-token:$GH_TOKEN@github.com/petegordon/olmo3-embed.git')
    CHECK = 'SciFact NFCorpus'  # token-budget batching re-check vs the Phase 7 scores (79.47 / 38.98), own folder
    for task in JOBS[JOB]:  # one process per task group so a failure in one doesn't lose the others
        tasks = task.split()
        cache = ' --doc-cache /content/doc_cache.npy' if task == 'FEVER ClimateFEVER' else ''
        if task == CHECK: cache = ' --name-suffix -tbcheck'
        log(f'start {task}')
        rc = sh(f'python -m olmo3_embed.eval --run runs/{RUN} --tasks {task} --max-seq-length 512 --batch-size 64 '
                f'--token-budget {TOKEN_BUDGET}{cache} 2>&1 | grep -vE "it/s]|Warning"')
        if task == CHECK:
            for f in sorted(glob.glob(f'results/results/olmo3-embed__{RUN}-tbcheck/*/*.json')):
                log(f'check {os.path.basename(f)[:-5]}: ' + str(round(100 * json.load(open(f))['scores']['test'][0]['ndcg_at_10'], 2)))
        have = {os.path.basename(f)[:-5] for f in glob.glob(f'results/results/olmo3-embed__{RUN}/*/*.json')}
        if task != CHECK: log(f'end {task}: ' + ('ok' if all(t in have for t in tasks) else f'MISSING {[t for t in tasks if t not in have]}'))
        for attempt in range(6):  # parallel jobs push to the same branch: retry on a rejected push
            if sh(f'bash scripts/push_results.sh "Full BEIR run #4 {RUN}: {task} (job {JOB}, Colab)"') == 0: break
            time.sleep(30 + 30 * attempt)
    have = {os.path.basename(f)[:-5] for f in glob.glob(f'results/results/olmo3-embed__{RUN}/*/*.json')}
    want = [t for g in JOBS[JOB] if g != CHECK for t in g.split()]
    log(('BEIR15_JOB_DONE ' if all(t in have for t in want) else 'BEIR15_JOB_INCOMPLETE ') + JOB)
finally:
    if not os.path.exists('/root/.olmo3_KEEP'):  # create this file to keep the runtime for debugging
        from google.colab import runtime
        runtime.unassign()
'''
def build(jobs, budget):
    c = code.replace('__JOBS__', json.dumps(jobs)).replace('__BUDGET__', str(budget))
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(True)},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(True)}],
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open('notebooks/phase8_beir15_colab.ipynb', 'w'), indent=1)
import sys
build(json.loads(sys.argv[1]), int(sys.argv[2]))
