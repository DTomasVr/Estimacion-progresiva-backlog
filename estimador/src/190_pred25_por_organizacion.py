"""E77 - PRED(25) del total del proyecto, desglosado por organizacion.

Mismo protocolo que la metrica publicada: cada proyecto predicho como si fuera nuevo,
y las filas con tareas cerradas promedian la METRICA sobre 60 sorteos de cuales se
cerraron primero.

Aviso que hay que leer junto a la tabla: PRED(25) sobre 3 o 9 proyectos es muy
ruidoso. Se marca el n de cada organizacion y se agrupa en "resto" todo lo que tiene
un solo proyecto, que si no son 45 filas de una observacion.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/pred25_organizacion")
S = "=" * 96
LAMBDA, SORTEOS, KS = 12, 60, (3, 5, 10)
MIN_ORG = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]

    proy = sw.project.to_numpy()
    real = sw.effort_hours.to_numpy(float)
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = {p: real[idx[p]].sum() for p in nom}

    org0 = {p: (p.split(":")[0] if ":" in p else p) for p in nom}
    cuenta = pd.Series(list(org0.values())).value_counts()
    org = {p: (org0[p] if cuenta[org0[p]] >= MIN_ORG else "resto (1-2 proyectos)")
           for p in nom}
    grupos = {}
    for p in nom:
        grupos.setdefault(org[p], []).append(p)

    def total_k(p, k, rng):
        i = idx[p]
        if len(i) <= k:
            return pred[i].sum()
        sel = rng.choice(i, k, replace=False)
        resto = np.setdiff1d(i, sel)
        off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
        return real[sel].sum() + (pred[resto] * np.exp(off)).sum()

    def pred25(ps, est):
        r = np.array([abs(est[p] - real_tot[p]) / real_tot[p] for p in ps])
        return float(np.mean(r <= .25))

    frio = {p: pred[idx[p]].sum() for p in nom}
    filas = []
    for g, ps in sorted(grupos.items(), key=lambda x: -len(x[1])):
        f = dict(organizacion=g, proyectos=len(ps),
                 tareas=int(sum(len(idx[p]) for p in ps)),
                 horas_medianas=float(np.median([real_tot[p] for p in ps])),
                 en_frio=pred25(ps, frio))
        for k in KS:
            vs = []
            for s in range(SORTEOS):
                rng = np.random.default_rng(1000 * k + s)
                vs.append(pred25(ps, {p: total_k(p, k, rng) for p in ps}))
            f[f"k{k}"] = float(np.mean(vs))
        filas.append(f)
    # fila global
    f = dict(organizacion="TODOS", proyectos=len(nom),
             tareas=int(len(sw)),
             horas_medianas=float(np.median(list(real_tot.values()))),
             en_frio=pred25(nom, frio))
    for k in KS:
        vs = []
        for s in range(SORTEOS):
            rng = np.random.default_rng(1000 * k + s)
            vs.append(pred25(nom, {p: total_k(p, k, rng) for p in nom}))
        f[f"k{k}"] = float(np.mean(vs))
    filas.append(f)

    t = pd.DataFrame(filas)
    print(f"{S}\nPRED(25) DEL TOTAL DEL PROYECTO, POR ORGANIZACION\n{S}")
    print(f"{'organizacion':<26}{'proy':>5}{'tareas':>8}{'h med':>8}"
          f"{'en frio':>9}{'k=3':>8}{'k=5':>8}{'k=10':>8}")
    for _, r in t.iterrows():
        sep = "-" * 80 if r.organizacion == "TODOS" else None
        if sep:
            print(sep)
        print(f"{r.organizacion:<26}{int(r.proyectos):>5}{int(r.tareas):>8}"
              f"{r.horas_medianas:>8.0f}{r.en_frio:>8.1%}{r.k3:>8.1%}"
              f"{r.k5:>8.1%}{r.k10:>8.1%}")
    print(f"\n  con menos de ~10 proyectos un punto de PRED(25) vale mas de 10 puntos "
          f"porcentuales:\n  esas filas son indicativas, no comparables entre si.")
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
