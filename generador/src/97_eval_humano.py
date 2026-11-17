"""sF1 del generador VIGENTE contra las 328 historias escritas por personas.

Es la cifra que va a la memoria: la tuberia de produccion tal cual (src/19_pipeline.py con
sus valores por omision), los 6 proyectos de data/reales/test_real.json (Bragilovski et al.,
CC BY 4.0) y tres semillas. Cada fila es un proyecto con una semilla.

El sF1 es propio: emparejamiento voraz por maximos sobre los embeddings del SBERT del
proyecto, como en BERTScore pero sobre CONJUNTOS de historias (src/11_eval_gen.py,
puntuar). Solo puntua historias, no tareas.

El intervalo es por remuestreo de las 18 filas (proyecto x semilla), asi que mide la
variacion entre proyectos y semillas, no la de una poblacion mayor: con 6 proyectos es
ancho por construccion.

  python src/97_eval_humano.py

out: results/49_real_vigente.csv  (+ resumen por pantalla)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGIX_BASE", BASE)
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
for s in SEMILLAS:
    pl.SEMILLA = s
    for c in T:
        t0 = time.time()
        r = P(c["descripcion"])
        H = [h["historia"] for h in r["historias"]]
        sc = ev.puntuar(sb, [H], [c["historias"]])
        filas.append(dict(semilla=s, proyecto=c["proyecto"], n_ref=len(c["historias"]),
                          n_gen=len(H), n_tareas=r["n_tareas"],
                          anclas=len(getattr(P, "ultimas_anclas", []) or []),
                          sF1=round(sc["sF1"], 4), precision=round(sc["sPrec"], 4),
                          cobertura=round(sc["sCob"], 4), segundos=round(time.time() - t0, 1)))
        print(f"  semilla {s} {c['proyecto']:<12} sF1 {filas[-1]['sF1']:.4f}", flush=True)

R = pd.DataFrame(filas)
R.to_csv(f"{BASE}/results/49_real_vigente.csv", index=False)
v = R["sF1"].to_numpy()
rng = np.random.default_rng(0)
bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
print("\npor proyecto (media de 3 semillas):")
print(R.groupby("proyecto")[["sF1", "precision", "cobertura"]].mean().round(4).to_string())
print(f"\npor semilla: {R.groupby('semilla')['sF1'].mean().round(4).to_dict()}")
print(f"sF1 {v.mean():.4f}  IC95 [{np.quantile(bs, .025):.4f}, {np.quantile(bs, .975):.4f}]"
      f"  precision {R['precision'].mean():.4f}  cobertura {R['cobertura'].mean():.4f}"
      f"  (n = {len(v)} filas)")
print(f"escrito results/49_real_vigente.csv")
