"""E101 - ¿El ancla por persona estropea el total del proyecto?

E100 midio el ancla por desarrollador sobre la vista POR TAREA. Pero el total
del proyecto se calcula sumando la cabeza de la media, y si se aplican
desplazamientos por persona esa suma cambia. Antes de llevar nada al artefacto
hay que comprobar que no se rompe lo unico que funciona bien.

Hay motivo para sospechar que no aporta: los desplazamientos de las personas de
un mismo proyecto se promedian al sumar TODO el proyecto, asi que deberia
cancelarse casi todo. Pero hay dos casos en que podria aportar:

  reparto desigual  si una persona lleva el 60 % de las tareas, su
                    desplazamiento no se cancela
  backlog parcial   cuando solo queda abierta la parte de un subconjunto de
                    personas, que es la situacion real a mitad de proyecto

Se mide lo segundo tambien, que es el caso de uso de verdad.

  A  total en frio
  B  anclaje por PROYECTO (lo actual, lambda=12)
  C  anclaje por PERSONA (lambda_d=8)
  D  los dos
y lo mismo midiendo el error SOLO sobre las tareas que quedan abiertas, que es
lo que el jefe de proyecto necesita saber.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/persona_en_el_total")
S = "=" * 100
LAM_P, LAM_D, SORTEOS, MIN_TAREAS = 12, 8, 40, 8
KS = (3, 5, 10)


def met(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    sw = todo[~todo.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    dev = sw.desarrollador.fillna("?").astype(str).to_numpy()
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([y[idx[p]].sum() for p in nom])

    # cuanto se concentra el trabajo en pocas personas
    conc = []
    for p in nom:
        v = pd.Series(dev[idx[p]]).value_counts(normalize=True)
        conc.append(float(v.iloc[0]))
    conc = np.array(conc)
    con_dev = np.array([(dev[idx[p]] != "?").mean() > .5 for p in nom])
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · "
          f"{int(con_dev.sum())} con desarrollador en mas de la mitad · "
          f"la persona que mas carga lleva: mediana {np.median(conc):.0%} "
          f"de las tareas · {time.time() - t0:.0f} s", flush=True)

    def desplaza(sel_mask, grupos, lam):
        """offset por grupo, estimado SOLO con las tareas marcadas como cerradas."""
        off = np.zeros(len(y))
        gl = pd.DataFrame(dict(g=grupos, y=y, p=pred, c=sel_mask))
        for g_, d in gl[gl.c].groupby("g"):
            k = len(d)
            o = np.log(d.y.sum() / d.p.sum()) * k / (k + lam)
            off[grupos == g_] = o
        return off

    filas, filas_ab = [], []
    for k in KS:
        acc = {b: [] for b in "ABCD"}
        acc_ab = {b: [] for b in "ABCD"}
        for s in range(SORTEOS):
            rng = np.random.default_rng(5000 + s)
            cerr = np.zeros(len(y), bool)
            for p in nom:
                i = idx[p]
                if len(i) > k:
                    cerr[rng.choice(i, k, replace=False)] = True
            # claves de grupo: el proyecto, y la persona DENTRO del proyecto
            gp = proy
            gd = np.array([f"{a}|{b}" for a, b in zip(proy, dev)])
            op = desplaza(cerr, gp, LAM_P)
            od = desplaza(cerr, gd, LAM_D)
            for b, off in (("A", np.zeros(len(y))), ("B", op), ("C", od),
                           ("D", op + od)):
                aj = pred * np.exp(off)
                tot = np.array([y[idx[p]][cerr[idx[p]]].sum() +
                                aj[idx[p]][~cerr[idx[p]]].sum() for p in nom])
                acc[b].append(met(real_tot, tot))
                # solo lo que queda ABIERTO
                ab_r = np.array([y[idx[p]][~cerr[idx[p]]].sum() for p in nom])
                ab_p = np.array([aj[idx[p]][~cerr[idx[p]]].sum() for p in nom])
                v = ab_r > 0
                acc_ab[b].append(met(ab_r[v], ab_p[v]))
        etq = {"A": "A · en frio", "B": "B · ancla de PROYECTO",
               "C": "C · ancla de PERSONA", "D": "D · las dos"}
        for b in "ABCD":
            filas.append(dict(brazo=etq[b], k=k,
                              **{c: float(np.mean([x[c] for x in acc[b]]))
                                 for c in acc[b][0]}))
            filas_ab.append(dict(brazo=etq[b], k=k,
                                 **{c: float(np.mean([x[c] for x in acc_ab[b]]))
                                    for c in acc_ab[b][0]}))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    tab = pd.DataFrame(filas_ab)
    print(f"\n{S}\nEL TOTAL DEL PROYECTO ({len(nom)} proyectos)\n{S}")
    print(t.round(4).to_string(index=False))
    print(f"\n{S}\nSOLO LO QUE QUEDA ABIERTO (lo que de verdad hay que pronosticar)"
          f"\n{S}")
    print(tab.round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    for nombre, tt in (("total", t), ("solo abierto", tab)):
        b5 = float(tt[(tt.brazo.str.startswith("B")) & (tt.k == 5)].PRED25.iloc[0])
        d5 = float(tt[(tt.brazo.str.startswith("D")) & (tt.k == 5)].PRED25.iloc[0])
        print(f"  {nombre:14s} k=5: solo proyecto {b5:.1%} · con persona {d5:.1%} "
              f"({(d5 - b5) * 100:+.1f} pts)")
    d5 = float(tab[(tab.brazo.str.startswith("D")) & (tab.k == 5)].PRED25.iloc[0])
    b5 = float(tab[(tab.brazo.str.startswith("B")) & (tab.k == 5)].PRED25.iloc[0])
    if d5 - b5 > .005:
        print("\n  La persona APORTA sobre lo abierto: ahi los desplazamientos no")
        print("  se cancelan porque no todas las personas tienen el mismo resto.")
    elif d5 - b5 < -.005:
        print("\n  La persona ESTROPEA el total. En el artefacto debe quedar como")
        print("  ancla de la vista POR TAREA y no entrar en el total.")
    else:
        print("\n  Neutra para el total: los desplazamientos se cancelan al sumar")
        print("  el proyecto entero. Su sitio es la vista por tarea.")
    t.to_csv(OUT / "total.csv", index=False)
    tab.to_csv(OUT / "abierto.csv", index=False)
    json.dump({"proyectos": len(nom), "lam_p": LAM_P, "lam_d": LAM_D,
               "carga_max_mediana": float(np.median(conc))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
