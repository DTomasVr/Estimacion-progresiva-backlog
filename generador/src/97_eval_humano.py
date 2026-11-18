"""sF1 of the CURRENT generator against the 328 stories written by people.

It is the figure that goes to the thesis: the production pipeline as it is
(src/19_pipeline.py with its defaults), the 6 projects in data/reales/test_real.json
(Bragilovski et al., CC BY 4.0) and three seeds. Each row is one project with one seed.

The sF1 is our own: greedy max matching over the embeddings of the project's SBERT, as in
BERTScore but over SETS of stories (src/11_eval_gen.py, puntuar). It only scores stories,
not tasks.

The interval is a bootstrap over the 18 rows (project x seed), so it measures the
variation between projects and seeds, not that of a larger population: with 6 projects
it is wide by construction.

  python src/97_eval_humano.py

out: results/49_real_vigente.csv  (+ summary on screen)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PBE_BASE", BASE)
sys.stdout.reconfigure(encoding="utf-8")
SEEDS = (11, 22, 33)


def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = load(f"{BASE}/src/11_eval_gen.py", "ev")
pl = load(f"{BASE}/src/19_pipeline.py", "pl")
T = json.load(open(f"{BASE}/data/reales/test_real.json", encoding="utf-8"))
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(ev.SBERT, device=pl.DISPOSITIVO)
P = pl.Pipeline(traducir=False)
rows = []
for s in SEEDS:
    pl.SEMILLA = s
    for c in T:
        t0 = time.time()
        r = P(c["descripcion"])
        H = [h["historia"] for h in r["historias"]]
        sc = ev.puntuar(sb, [H], [c["historias"]])
        rows.append(dict(seed=s, project=c["proyecto"], n_ref=len(c["historias"]),
                         n_gen=len(H), n_tasks=r["n_tareas"],
                         anchors=len(getattr(P, "ultimas_anclas", []) or []),
                         sF1=round(sc["sF1"], 4), precision=round(sc["sPrec"], 4),
                         coverage=round(sc["sCob"], 4), seconds=round(time.time() - t0, 1)))
        print(f"  seed {s} {c['proyecto']:<12} sF1 {rows[-1]['sF1']:.4f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{BASE}/results/49_real_vigente.csv", index=False)
v = R["sF1"].to_numpy()
rng = np.random.default_rng(0)
bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
print("\nby project (mean of 3 seeds):")
print(R.groupby("project")[["sF1", "precision", "coverage"]].mean().round(4).to_string())
print(f"\nby seed: {R.groupby('seed')['sF1'].mean().round(4).to_dict()}")
print(f"sF1 {v.mean():.4f}  CI95 [{np.quantile(bs, .025):.4f}, {np.quantile(bs, .975):.4f}]"
      f"  precision {R['precision'].mean():.4f}  coverage {R['coverage'].mean():.4f}"
      f"  (n = {len(v)} rows)")
print(f"written results/49_real_vigente.csv")
