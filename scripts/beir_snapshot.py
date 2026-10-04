"""Snapshot the MTEB BEIR leaderboard (15-task nDCG@10 avg) from the embeddings-benchmark/results repo.

Run once with `uv run --with mteb python -c "import mteb; mteb.load_results(download_latest=True)"` to clone results
into ~/.cache/mteb, then `uv run scripts/beir_snapshot.py`.
"""
import datetime
OUT=f"docs/beir_snapshot_{datetime.date.today()}.csv"
R=os.path.expanduser('~/.cache/mteb/remote/results')
T=['TRECCOVID','NFCorpus','NQ','HotpotQA','FiQA2018','ArguAna','Touche2020','CQADupstackRetrieval','QuoraRetrieval','DBPedia','SCIDOCS','FEVER','ClimateFEVER','SciFact','MSMARCO']
CQ=['Android','English','Gaming','Gis','Mathematica','Physics','Programmers','Stats','Tex','Unix','Webmasters','Wordpress']
def score(p, split):
    try: d=json.load(open(p,encoding='utf-8'))
    except Exception: return None
    s=d.get('scores',{}).get(split)
    if not s: return None
    for e in s:
        if e.get('hf_subset') in ('default','en'): return e.get('ndcg_at_10', e.get('main_score'))
    return s[0].get('ndcg_at_10', s[0].get('main_score'))
rows=[]
for m in os.listdir(R):
    best=None
    for rev in glob.glob(os.path.join(R,m,'*')):
        sc={}
        for t in T:
            sp='dev' if t=='MSMARCO' else 'test'
            if t=='CQADupstackRetrieval':
                v=score(os.path.join(rev,t+'.json'),sp)
                if v is None:
                    vs=[score(os.path.join(rev,f'CQADupstack{c}Retrieval.json'),sp) for c in CQ]
                    v=st.mean(vs) if all(x is not None for x in vs) else None
            else: v=score(os.path.join(rev,t+'.json'),sp)
            if v is not None: sc[t]=v
        if best is None or len(sc)>len(best): best=sc
    if best and len(best)>=14:
        rows.append((st.mean(best.values())*100,len(best),m,best))
rows.sort(reverse=True)
import csv; w=csv.writer(open(OUT,"w",newline="")); w.writerow(["model","beir_avg","n_tasks"]+T)
for a,n,m,b in rows: w.writerow([m.replace("__","/"),round(a,2),n]+[round(b[t]*100,2) if t in b else "" for t in T])
for a,n,m,b in rows[:30]:
    print(f"{a:6.2f} n={n} {m}")
print('--- per-task for top 5')
for a,n,m,b in rows[:5]:
    print(m, {k:round(v*100,1) for k,v in b.items()})
print('--- reference models')
for a,n,m,b in rows:
    if any(k in m.lower() for k in ['e5-mistral','bge-base-en-v1.5','qwen3-embedding','gte-qwen2-7b','nv-embed','olmo','llm2vec','promptriever','repllama']):
        print(f"{a:6.2f} n={n} {m}")
