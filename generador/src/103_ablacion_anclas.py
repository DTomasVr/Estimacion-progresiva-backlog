"""Ablation of the anchors on the CURRENT model, in the same session (PBE_SEED does not
reproduce across sessions): same protocol as src/97_eval_humano.py (328 Bragilovski
stories, 6 projects, seeds 11/22/33), with and without anchors.

CI95 of the paired difference in two ways: bootstrap over the 18 pairs and bootstrap
by project (6, averaging the seeds first).

  python src/103_ablacion_anclas.py
out: results/103_ablacion_anclas.csv
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
for anchor in (True, False):
    pl.ANCLAR = anchor
    for s in SEEDS:
        pl.SEMILLA = s
        for c in T:
            t0 = time.time()
            r = P(c["descripcion"])
            H = [h["historia"] for h in r["historias"]]
            sc = ev.puntuar(sb, [H], [c["historias"]])
            rows.append(dict(anchors_on=int(anchor), seed=s, project=c["proyecto"],
                             n_gen=len(H), anchors=len(P.ultimas_anclas or []),
                             sF1=round(sc["sF1"], 4), precision=round(sc["sPrec"], 4),
                             coverage=round(sc["sCob"], 4),
                             seconds=round(time.time() - t0, 1)))
            print(f"  anchors={int(anchor)} seed {s} {c['proyecto']:<12} "
                  f"sF1 {rows[-1]['sF1']:.4f}", flush=True)

R = pd.DataFrame(rows)
R.to_csv(f"{BASE}/results/103_ablacion_anclas.csv", index=False)
on = R[R.anchors_on == 1].set_index(["project", "seed"]).sF1
off = R[R.anchors_on == 0].set_index(["project", "seed"]).sF1
d = (on - off.reindex(on.index)).to_numpy()
rng = np.random.default_rng(0)
b18 = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
dp = (on - off.reindex(on.index)).groupby(level=0).mean().to_numpy()
b6 = dp[rng.integers(0, len(dp), (10000, len(dp)))].mean(1)
print(f"\nwith anchors {on.mean():.4f} · without anchors {off.mean():.4f} · difference {d.mean():+.4f}")
print(f"CI95 over 18 pairs    [{np.quantile(b18, .025):+.4f}, {np.quantile(b18, .975):+.4f}]")
print(f"CI95 over 6 projects  [{np.quantile(b6, .025):+.4f}, {np.quantile(b6, .975):+.4f}]"
      f"  · projects where it helps: {(dp > 0).sum()}/6 · {np.round(dp, 4).tolist()}")
print("precision on/off", round(R[R.anchors_on == 1].precision.mean(), 4),
      round(R[R.anchors_on == 0].precision.mean(), 4),
      "· coverage on/off", round(R[R.anchors_on == 1].coverage.mean(), 4),
      round(R[R.anchors_on == 0].coverage.mean(), 4))
