"""Anade al artefacto v6 la banda del total calibrada sobre lo ABIERTO y por k.

E107 midio que la banda del total de hoy no se estrecha nunca: multiplica el
total entero -incluidas las horas ya cerradas, que se saben- por el residuo en
frio. Con 20 tareas cerradas sigue en x10,61 y cubre el 98 % prometiendo el 80.

La correccion pone la incertidumbre solo en lo que queda abierto, y la calibra
al numero de tareas cerradas y al tramo de tamano:

    banda = cerradas + abiertas x e^off x exp([cuantil_lo, cuantil_hi])

Esto guarda en el paquete los residuos de lo abierto para k en KS y cada tramo,
reunidos sobre los sorteos. Son datos: no se reentrena nada.
"""
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
V6 = Path("models/estimador_pbe_v6.joblib")
MIN_TAREAS, LAMBDA, SORTEOS = 8, 12, 20
KS = (0, 3, 5, 10, 20)
CORTES = [20, 60, 200]


def main():
    t0 = time.time()
    resp = V6.with_suffix(".joblib.antes_de_banda_abierta")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    tr = np.digitize([len(idx[p]) for p in nom], CORTES)

    tabla = {}
    for k in KS:
        res = np.zeros((SORTEOS, len(nom)))
        for s in range(SORTEOS):
            rng = np.random.default_rng(8000 + s)
            for j, p in enumerate(nom):
                i = idx[p]
                kk = min(k, len(i) - 1)
                if kk:
                    sel = rng.choice(i, kk, replace=False)
                    rest = np.setdiff1d(i, sel)
                    off = np.log(y[sel].sum() / pred[sel].sum()) * kk / (kk + LAMBDA)
                    res[s, j] = np.log(y[rest].sum() /
                                       (pred[rest] * np.exp(off)).sum())
                else:
                    res[s, j] = np.log(y[i].sum() / pred[i].sum())
        tabla[k] = {int(g): res[:, tr == g].ravel() for g in np.unique(tr)}
        tabla[k][-1] = res.ravel()          # sin tramo, por si un tramo no llega
        print(f"  k={k}: " + " · ".join(f"{g}:{len(v)}" for g, v in
                                        sorted(tabla[k].items())), flush=True)

    p = dict(joblib.load(V6))
    p["residuos_abierto"] = tabla
    p["ks_banda"] = list(KS)
    joblib.dump(p, V6, compress=3)
    print(f"escrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
