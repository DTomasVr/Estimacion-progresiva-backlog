"""E107 - Una banda del total que se estrecha con el avance, como debe.

Revisando estimar_proyecto aparecen dos cosas que ensanchan la banda sin motivo:

  1  la banda MULTIPLICA EL TOTAL ENTERO:
         banda = total x exp(cuantil del residuo)
     y el total incluye las horas de las tareas ya cerradas, que se SABEN. Si
     el 80 % del proyecto esta hecho, esas horas no tienen incertidumbre y aun
     asi entran en la banda como si la tuvieran
  2  los residuos son SIEMPRE los de en frio. Con 10 tareas cerradas el
     anclaje ya ha bajado el error (MdMRE 0,355 -> 0,260), pero la banda sigue
     calibrada para el caso sin historial

Lo correcto es poner la incertidumbre solo donde la hay:

     total = cerradas + abiertas x e^off
     banda = cerradas + abiertas x e^off x exp([cuantil_lo, cuantil_hi])

con los cuantiles calibrados sobre el residuo de lo ABIERTO al mismo numero de
tareas cerradas k, dentro del tramo de tamano (la cobertura por tramo de E89).

  A  lo de hoy: total entero x residuo en frio
  B  solo lo abierto, residuo en frio
  C  solo lo abierto, residuo calibrado a ese k, por tramo
  D  como C pero sin tramo

Conforme dejando fuera el proyecto evaluado, con la metrica promediada entre
sorteos de que tareas se cerraron. Se mide la cobertura al 80 % global y por
tramo, y la anchura de la banda del TOTAL (alto/bajo).
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
pd.set_option("display.width", 220)
OUT = Path("results/banda_que_se_estrecha")
S = "=" * 110
MIN_TAREAS, LAMBDA, SORTEOS, NIVEL = 8, 12, 20, .8
KS = (0, 3, 5, 10, 20)
CORTES = [20, 60, 200]
ETIQ = ["8-20", "21-60", "61-200", "mas de 200"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    P = len(nom)
    n_t = np.array([len(idx[p]) for p in nom])
    tr = np.digitize(n_t, CORTES)
    real_tot = np.array([y[idx[p]].sum() for p in nom])
    a = (1 - NIVEL) / 2
    print(f"{P} proyectos · {len(sw):,} tareas · {time.time() - t0:.0f} s",
          flush=True)

    # por sorteo y k: lo cerrado, la prediccion de lo abierto y su residuo
    cer = {k: np.zeros((SORTEOS, P)) for k in KS}
    abi = {k: np.zeros((SORTEOS, P)) for k in KS}
    res = {k: np.zeros((SORTEOS, P)) for k in KS}
    for k in KS:
        for s in range(SORTEOS):
            rng = np.random.default_rng(8000 + s)
            for j, p in enumerate(nom):
                i = idx[p]
                kk = min(k, len(i) - 1)
                if kk:
                    sel = rng.choice(i, kk, replace=False)
                    rest = np.setdiff1d(i, sel)
                    off = np.log(y[sel].sum() / pred[sel].sum()) * kk / (kk + LAMBDA)
                    c_, a_ = y[sel].sum(), (pred[rest] * np.exp(off)).sum()
                    ra = y[rest].sum()
                else:
                    c_, a_, ra = 0., pred[i].sum(), y[i].sum()
                cer[k][s, j], abi[k][s, j] = c_, a_
                res[k][s, j] = np.log(ra / a_)
    print(f"  residuos listos · {time.time() - t0:.0f} s", flush=True)

    def pool(k, j, por_tramo):
        """Residuos de calibracion: los OTROS proyectos, todos los sorteos."""
        otros = np.arange(P) != j
        if por_tramo:
            g = otros & (tr == tr[j])
            if g.sum() >= 12:
                otros = g
        return res[k][:, otros].ravel()

    filas = []
    for k in KS:
        acc = {}
        for s in range(SORTEOS):
            for j in range(P):
                tot = cer[k][s, j] + abi[k][s, j]
                for etq in ("A", "B", "C", "D"):
                    if etq == "A":
                        q = pool(0, j, True)
                        lo = tot * np.exp(qconf(q, a, False))
                        hi = tot * np.exp(qconf(q, 1 - a, True))
                    else:
                        kk = 0 if etq == "B" else k
                        q = pool(kk, j, etq != "D")
                        lo = cer[k][s, j] + abi[k][s, j] * np.exp(qconf(q, a, False))
                        hi = cer[k][s, j] + abi[k][s, j] * np.exp(qconf(q, 1 - a, True))
                    acc.setdefault(etq, []).append(
                        (j, real_tot[j] >= lo and real_tot[j] <= hi, hi / lo))
        nombres = {"A": "A · hoy: total x residuo en frio",
                   "B": "B · solo lo abierto, residuo en frio",
                   "C": "C · solo lo abierto, residuo a ese k, por tramo",
                   "D": "D · como C sin tramo"}
        for etq, v in acc.items():
            jj = np.array([x[0] for x in v])
            cov = np.array([x[1] for x in v], float)
            w = np.array([x[2] for x in v])
            d = dict(k=k, banda=nombres[etq], cobertura=float(cov.mean()),
                     anchura_media=float(w.mean()), anchura_mediana=float(np.median(w)))
            for g, e in enumerate(ETIQ):
                s_ = tr[jj] == g
                d[f"cob_{e}"] = float(cov[s_].mean())
                d[f"anch_{e}"] = float(np.median(w[s_]))
            filas.append(d)
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nANCHURA MEDIANA DE LA BANDA DEL TOTAL AL 80 % (alto/bajo)\n{S}")
    print(t.pivot(index="k", columns="banda",
                  values="anchura_mediana").round(2).to_string())
    print(f"\n{S}\nCOBERTURA GLOBAL (objetivo 0,80)\n{S}")
    print(t.pivot(index="k", columns="banda", values="cobertura").round(3).to_string())
    print(f"\n{S}\nPOR TRAMO, LA BANDA C (cobertura · anchura mediana)\n{S}")
    c = t[t.banda.str.startswith("C")].set_index("k")
    for e in ETIQ:
        print(f"  {e:>11}  " + "  ".join(
            f"k={k}: {c.loc[k, f'cob_{e}']:.2f}·x{c.loc[k, f'anch_{e}']:.2f}"
            for k in KS))

    print(f"\n{S}\nLECTURA\n{S}")
    for k in KS:
        a_ = t[(t.k == k) & t.banda.str.startswith("A")].iloc[0]
        c_ = t[(t.k == k) & t.banda.str.startswith("C")].iloc[0]
        print(f"  k={k:2d}: hoy x{a_.anchura_mediana:5.2f} (cubre {a_.cobertura:.0%}) "
              f"-> corregida x{c_.anchura_mediana:5.2f} (cubre {c_.cobertura:.0%})")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": P, "sorteos": SORTEOS, "lambda": LAMBDA,
               "nivel": NIVEL, "cortes": CORTES},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
