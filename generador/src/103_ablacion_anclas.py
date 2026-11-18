"""Ablacion de las anclas sobre el modelo VIGENTE, en la misma sesion (PBE_SEED no
reproduce entre sesiones): mismo protocolo que src/97_eval_humano.py (328 historias de
Bragilovski, 6 proyectos, semillas 11/22/33), con y sin anclas.

IC95 de la diferencia emparejada de dos formas: remuestreo de los 18 pares y bootstrap
por proyecto (6, promediando antes las semillas).

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
SEMILLAS = (11, 22, 33)


def cargar(ruta, alias):
    spec = importlib.util.spec_from_file_location(alias, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ev = cargar(f"{BASE}/src/11_eval_gen.py", "ev")
pl = cargar(f"{BASE}/src/19_pipeline.py", "pl")
T = json.load(open(f"{BASE}/data/reales/test_real.json", encoding="utf-8"))
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(ev.SBERT, device=pl.DISPOSITIVO)
P = pl.Pipeline(traducir=False)
filas = []
for anclar in (True, False):
    pl.ANCLAR = anclar
    for s in SEMILLAS:
        pl.SEMILLA = s
        for c in T:
            t0 = time.time()
            r = P(c["descripcion"])
            H = [h["historia"] for h in r["historias"]]
            sc = ev.puntuar(sb, [H], [c["historias"]])
            filas.append(dict(anclas_on=int(anclar), semilla=s, proyecto=c["proyecto"],
                              n_gen=len(H), anclas=len(P.ultimas_anclas or []),
                              sF1=round(sc["sF1"], 4), precision=round(sc["sPrec"], 4),
                              cobertura=round(sc["sCob"], 4),
                              segundos=round(time.time() - t0, 1)))
            print(f"  anclas={int(anclar)} semilla {s} {c['proyecto']:<12} "
                  f"sF1 {filas[-1]['sF1']:.4f}", flush=True)

R = pd.DataFrame(filas)
R.to_csv(f"{BASE}/results/103_ablacion_anclas.csv", index=False)
on = R[R.anclas_on == 1].set_index(["proyecto", "semilla"]).sF1
off = R[R.anclas_on == 0].set_index(["proyecto", "semilla"]).sF1
d = (on - off.reindex(on.index)).to_numpy()
rng = np.random.default_rng(0)
b18 = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
dp = (on - off.reindex(on.index)).groupby(level=0).mean().to_numpy()
b6 = dp[rng.integers(0, len(dp), (10000, len(dp)))].mean(1)
print(f"\ncon anclas {on.mean():.4f} · sin anclas {off.mean():.4f} · diferencia {d.mean():+.4f}")
print(f"IC95 por 18 pares    [{np.quantile(b18, .025):+.4f}, {np.quantile(b18, .975):+.4f}]")
print(f"IC95 por 6 proyectos [{np.quantile(b6, .025):+.4f}, {np.quantile(b6, .975):+.4f}]"
      f"  · proyectos donde ayuda: {(dp > 0).sum()}/6 · {np.round(dp, 4).tolist()}")
print("precision on/off", round(R[R.anclas_on == 1].precision.mean(), 4),
      round(R[R.anclas_on == 0].precision.mean(), 4),
      "· cobertura on/off", round(R[R.anclas_on == 1].cobertura.mean(), 4),
      round(R[R.anclas_on == 0].cobertura.mean(), 4))
