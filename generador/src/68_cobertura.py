"""COBERTURA DEL ALCANCE, sin referencia.

El hueco que faltaba. Todo lo que mediamos compara contra un backlog de
referencia, asi que no sirve sobre una descripcion nueva de PBE, que es el caso
real. Y para la cadena de estimacion importa de forma directa: si un rasgo que la
descripcion menciona no queda cubierto por ninguna tarea, ese trabajo NO esta en
el backlog, y su esfuerzo falta entero.

Como se puede medir sin referencia: las descripciones de la taxonomia v3 se
construyen a partir de rasgos conocidos ("shopping cart and checkout", "discount
coupons", "wishlists"), y ahora quedan registrados en descripciones_v3.json. Asi
que se puede preguntar, rasgo a rasgo, si el backlog lo cubre.

  cobertura(rasgo) = max similitud del rasgo con cualquier historia (o tarea)
  cubierto         = esa similitud pasa de 0,5, el mismo umbral del F1 suave

Se reporta por historias y por TAREAS aparte: un rasgo puede quedar cubierto por
una tarea aunque ninguna historia lo nombre, y para el esfuerzo lo que cuenta son
las tareas.

  PBE_SUBMODULAR=1 python src/68_cobertura.py [N]

out: results/68_cobertura.csv
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMBRAL = 0.5
N = int(next((a for a in sys.argv[1:] if a.isdigit()), 30))

os.environ.setdefault("PBE_DEVICE", "cuda")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
for v, at in [("PBE_ARRIBA", "M_ARRIBA"), ("PBE_ABAJO", "M_ABAJO")]:
    if os.environ.get(v):
        setattr(pl, at, os.environ[v])

# descripciones RESERVADAS: las del split de test de v3, nunca entrenadas
D = {x["id"]: x for x in json.load(open(f"{BASE}/data/synth/descripciones_v3.json",
                                        encoding="utf-8"))}
t = pd.read_parquet(f"{BASE}/data/processed/s2s_v3a_test.parquet", columns=["parent_uid"])
ids = [i for i in dict.fromkeys(str(x).split("#")[0] for x in t.parent_uid.dropna())
       if i in D][:N]
print(f"{len(ids)} descripciones reservadas | submodular="
      f"{os.environ.get('PBE_SUBMODULAR', 'no')} tau={os.environ.get('PBE_TAU', '0')}",
      flush=True)

P = pl.Pipeline(traducir=False)
filas = []
for n, i in enumerate(ids):
    caso = D[i]
    r = P(caso["descripcion"], k_historias=12)
    hs = [h["historia"] for h in r["historias"]]
    ts = [x for h in r["historias"] for x in h["tareas"]]
    if not hs:
        continue
    Er = P.sb.encode(caso["rasgos"], normalize_embeddings=True, show_progress_bar=False)
    Eh = P.sb.encode(hs, normalize_embeddings=True, show_progress_bar=False)
    Et = (P.sb.encode(ts, normalize_embeddings=True, show_progress_bar=False)
          if ts else np.zeros((1, Eh.shape[1]), dtype=Eh.dtype))
    for k, rasgo in enumerate(caso["rasgos"]):
        filas.append(dict(id=i, sector=caso["sector"], rasgo=rasgo,
                          sim_hist=float((Er[k] @ Eh.T).max()),
                          sim_tarea=float((Er[k] @ Et.T).max()),
                          n_hist=len(hs), n_tareas=len(ts)))
    if (n + 1) % 10 == 0:
        print(f"  {n+1}/{len(ids)}", flush=True)

F = pd.DataFrame(filas)
suf = "_sub" if os.environ.get("PBE_SUBMODULAR") == "1" else ""
F.to_csv(f"{BASE}/results/68_cobertura{suf}.csv", index=False)

print("\n" + "=" * 66)
print(f"  cobertura suave por HISTORIAS : {F.sim_hist.mean():.4f}")
print(f"  rasgos cubiertos (>{UMBRAL})   : {100*(F.sim_hist >= UMBRAL).mean():.1f} %")
print(f"  cobertura suave por TAREAS    : {F.sim_tarea.mean():.4f}")
print(f"  rasgos cubiertos (>{UMBRAL})   : {100*(F.sim_tarea >= UMBRAL).mean():.1f} %")
print(f"  al menos uno de los dos       : "
      f"{100*((F[['sim_hist','sim_tarea']].max(1)) >= UMBRAL).mean():.1f} %")
print(f"  backlogs completos (3 de 3)   : "
      f"{100*F.groupby('id').apply(lambda g: (g[['sim_hist','sim_tarea']].max(1) >= UMBRAL).all()).mean():.1f} %")
print(f"  historias {F.n_hist.mean():.1f} | tareas {F.n_tareas.mean():.1f} por backlog")
peor = F.nsmallest(6, "sim_tarea")[["sector", "rasgo", "sim_tarea"]]
print("\nrasgos peor cubiertos:")
for _, x in peor.iterrows():
    print(f"  {x.sim_tarea:.3f}  [{x.sector}] {x.rasgo}")
print(f"-> results/68_cobertura{suf}.csv")
