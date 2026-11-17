"""Comparacion pareada podado contra sin podar, a partir de las generaciones guardadas.

Cada brazo se evalua en su PROPIO proceso porque necesita su propio parquet de test
(los identificadores del vocabulario podado no significan nada para el modelo
completo). Los dos parquets salen de s2s_pruned_test.parquet en el mismo orden, asi
que las filas se emparejan por indice y se comprueba con parent_uid.

  python src/79_pareado_poda.py <modelo_podado> <modelo_sin_podar>

out: results/79_pareado_poda.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from sentence_transformers import SentenceTransformer  # noqa: E402

A, B = sys.argv[1], sys.argv[2]          # nombres de carpeta, no rutas
SEED = 20260922


def cargar(nombre):
    p = f"{BASE}/results/11_gen_test_{nombre}.parquet"
    if not os.path.exists(p):
        raise SystemExit(f"falta {p}")
    return pd.read_parquet(p)


da, db = cargar(A), cargar(B)
if len(da) != len(db) or (da.parent_uid.values != db.parent_uid.values).any():
    raise SystemExit("las dos corridas no cubren las mismas filas en el mismo orden")
print(f"{len(da)} filas emparejadas", flush=True)

sb = SentenceTransformer(ev.SBERT, device=os.environ.get("PBE_DEVICE", "cuda"))
refs = [str(r).split(" ; ") for r in da.ref]
sa = ev.puntuar(sb, [str(g).split(" ; ") for g in da.gen], refs)
sbb = ev.puntuar(sb, [str(g).split(" ; ") for g in db.gen], refs)

filas = []
for nom, s in [(A, sa), (B, sbb)]:
    for mitad in ["todas", "tasks", "stories"]:
        m = np.ones(len(da), bool) if mitad == "todas" else (da.mitad == mitad).values
        filas.append(dict(modelo=nom, mitad=mitad, n=int(m.sum()),
                          **{c: round(float(s["mat"][m, i].mean()), 4)
                             for i, c in enumerate(s["cols"])}))
R = pd.DataFrame(filas)
print(R.to_string(index=False))

rng = np.random.default_rng(SEED)
comp = []
for mitad in ["todas", "tasks", "stories"]:
    m = np.ones(len(da), bool) if mitad == "todas" else (da.mitad == mitad).values
    dd = (sbb["por_ejemplo"] - sa["por_ejemplo"])[m]
    bs = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
    comp.append(dict(mitad=mitad, n=len(dd), delta_sF1=round(float(dd.mean()), 4),
                     ic95_lo=round(float(np.quantile(bs, .025)), 4),
                     ic95_hi=round(float(np.quantile(bs, .975)), 4)))
C = pd.DataFrame(comp)
print(f"\nsin podar - podado ({B} - {A}):")
print(C.to_string(index=False))
R.to_csv(f"{BASE}/results/79_pareado_poda.csv", index=False)
C.to_csv(f"{BASE}/results/79_pareado_poda_ic.csv", index=False)
print("-> results/79_pareado_poda.csv")
