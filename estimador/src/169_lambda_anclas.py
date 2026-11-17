"""E62 - La contraccion del anclaje esta tomada del nivel de tarea, y aqui puede sobrar.

El desplazamiento del proyecto se aplica contraido: con k tareas cerradas se usa
solo la fraccion k/(k+lambda) del desplazamiento observado, con lambda = 12. Ese
valor viene del anclaje por TAREA, donde protege de que dos tareas sueltas muevan
todas las predicciones.

Para el TOTAL puede estar mal calibrado, y E61 da un motivo para sospecharlo: los
proyectos que registran a 0,5 h por tarea necesitan una correccion de x5, y con
k=3 y lambda=12 se les aplica solo el 20 % de lo que hace falta.

Protocolo, con los tres cuidados de E53: el desplazamiento se estima con el
cociente de SUMAS, las tareas cerradas se SABEN y no se estiman, y se promedia la
METRICA entre sorteos, no la prediccion. Este ultimo punto es el que hizo falta
descubrir aqui: con la version que promediaba predicciones, lambda = 0 daba un
MdMRE de 0,08 y un +-25 % del 92 %, que era imposible y venia de que promediar 60
sorteos cancela la varianza de cual tarea toca.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/lambda_anclas")
S = "=" * 112
MIN_TAREAS, SORTEOS = 8, 40
LAMBDAS = (0, 1, 2, 3, 6, 12, 24)


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                MdMRE=float(np.median(r)),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    le = np.log(es_)
    g = d.groupby("p")
    real = g.real.sum().reindex(nom).to_numpy()
    hpt = g.real.median().reindex(nom).to_numpy()
    print(f"{len(nom)} proyectos · {SORTEOS} sorteos por casilla")

    M = {}
    for k in (1, 3, 5, 10, 20):
        rng = np.random.default_rng(3000 + k)
        sel_s = []
        for _ in range(SORTEOS):
            sel_s.append({p: (rng.choice(idx[p], k, replace=False)
                              if len(idx[p]) > k else None) for p in nom})
        for lam in LAMBDAS:
            A = np.zeros((SORTEOS, len(nom)))
            for s, sel in enumerate(sel_s):
                for j, p in enumerate(nom):
                    i = idx[p]
                    if sel[p] is None:
                        A[s, j] = es_[i].sum()
                        continue
                    ss = sel[p]
                    resto = np.setdiff1d(i, ss)
                    off = np.log(re_[ss].sum() / es_[ss].sum()) * k / (k + lam)
                    A[s, j] = re_[ss].sum() + np.exp(le[resto] + off).sum()
            M[(k, lam)] = A
        print(f"  k={k} listo", flush=True)

    filas = []
    for (k, lam), A in M.items():
        ms = [mt(real, A[s]) for s in range(SORTEOS)]
        filas.append(dict(k=k, lam=lam,
                          **{c: float(np.mean([m[c] for m in ms])) for c in ms[0]}))
    t = pd.DataFrame(filas)
    for c in ("MdMRE", "PRED25", "MAE_h"):
        print(f"\n{S}\n{c} DEL TOTAL SEGUN k Y lambda\n{S}")
        print(t.pivot(index="k", columns="lam", values=c).round(4).to_string())

    print(f"\n{S}\nlambda ELEGIDO DEJANDO FUERA CADA PROYECTO\n{S}")
    fs = []
    for k in (1, 3, 5, 10, 20):
        ms = []
        for s in range(SORTEOS):
            pred = np.zeros(len(real))
            for i in range(len(real)):
                o = np.ones(len(real), bool)
                o[i] = False
                mejor, ml = 1e9, LAMBDAS[0]
                for lam in LAMBDAS:
                    v = float(np.median(np.abs(M[(k, lam)][s][o] - real[o]) / real[o]))
                    if v < mejor:
                        mejor, ml = v, lam
                pred[i] = M[(k, ml)][s][i]
            ms.append(mt(real, pred))
        fs.append(dict(k=k, lam="elegido",
                       **{c: float(np.mean([m[c] for m in ms])) for c in ms[0]}))
        fs.append(dict(k=k, lam="12 (el de siempre)",
                       **{c: float(np.mean([mt(real, M[(k, 12)][s])[c]
                                            for s in range(SORTEOS)])) for c in ms[0]}))
    e = pd.DataFrame(fs)
    print(e.round(4).to_string(index=False))

    print(f"\n{S}\nMdMRE POR TIPO DE PROYECTO Y lambda (k=3)\n{S}")
    tr = pd.cut(hpt, [0, 1, 2, 4, 1e9], labels=["<1 h", "1-2 h", "2-4 h", ">4 h"])
    fs = []
    for etq in tr.categories:
        m = np.asarray(tr == etq)
        f = dict(tramo=etq, proyectos=int(m.sum()))
        for lam in LAMBDAS:
            f[f"lam{lam}"] = float(np.mean([
                np.median(np.abs(M[(3, lam)][s][m] - real[m]) / real[m])
                for s in range(SORTEOS)]))
        fs.append(f)
    c = pd.DataFrame(fs)
    print(c.round(3).to_string(index=False))
    t.to_csv(OUT / "barrido.csv", index=False)
    e.to_csv(OUT / "elegido.csv", index=False)
    c.to_csv(OUT / "por_tramo.csv", index=False)
    json.dump({"lambdas": list(LAMBDAS), "sorteos": SORTEOS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/barrido.csv")


if __name__ == "__main__":
    main()
