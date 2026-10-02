"""E86 - Que contraccion necesita el anclaje segun haya o no estimacion del usuario.

E85 encontro que, encima de la estimacion del usuario, anclar con 3 tareas cerradas
EMPEORA (70,5 % contra 73,2 % sin anclar). El motivo esta medido: tras la estimacion
solo queda un 1,9 % de la varianza del residuo ENTRE proyectos, asi que el anclaje
casi no tiene nada que corregir y con pocas tareas su ruido pesa mas que su senal.

La solucion no es apagar el anclaje: es contraerlo mas. El factor k/(k+lambda) con
lambda = 12 se calibro sobre el residuo en frio, donde habia un 11 % entre proyectos.
Si ahora hay un 1,9 %, lambda deberia ser mayor.

Se barre lambda en los dos escenarios y se elige el que hace falta en cada uno. El
resultado va al artefacto: lambda condicional a si hay estimacion.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/lambda_condicional")
S = "=" * 108
SORTEOS, PLIEGUES, COB = 60, 5, .8
KS = (1, 3, 5, 10, 20)
LAMBDAS = (6, 12, 24, 48, 96)


def pred25(real, pred):
    return float(np.mean(np.abs(np.asarray(pred, float) - real) / real <= .25))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, V, pred = sw[m].reset_index(drop=True), V[m], pred[m]
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    cob = pd.Series(hay).groupby(sw.project.to_numpy()).mean()
    nom = list(cob[cob >= COB].index)
    s_p = sw.project.isin(nom).to_numpy()
    sw, V, pred, est, hay = (sw[s_p].reset_index(drop=True), V[s_p], pred[s_p],
                             est[s_p], hay[s_p])
    proy, real = sw.project.to_numpy(), sw.effort_hours.to_numpy(float)
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([real[idx[p]].sum() for p in nom])
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · {time.time() - t0:.0f} s",
          flush=True)

    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = pred.copy()
    for a, b in GroupKFold(PLIEGUES).split(X, real, proy):
        a = a[hay[a]]
        mo = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                           l2_regularization=1., loss="poisson",
                                           random_state=0).fit(X[a], real[a])
        bb = b[hay[b]]
        corr[bb] = mo.predict(X[bb])
    corr = np.clip(corr, 1e-6, None)
    print(f"  corrector listo · {time.time() - t0:.0f} s", flush=True)

    def ancla(base, k, lam, rng):
        t_ = np.empty(len(nom))
        for j, p in enumerate(nom):
            i = idx[p]
            if len(i) <= k:
                t_[j] = base[i].sum()
                continue
            sel = rng.choice(i, k, replace=False)
            resto = np.setdiff1d(i, sel)
            off = np.log(real[sel].sum() / base[sel].sum()) * k / (k + lam)
            t_[j] = real[sel].sum() + (base[resto] * np.exp(off)).sum()
        return t_

    filas = []
    for etq, base in (("sin estimacion", pred), ("CON estimacion", corr)):
        sin_anclar = pred25(real_tot, np.array([base[idx[p]].sum() for p in nom]))
        filas.append(dict(escenario=etq, k=0, lam=0, PRED25=sin_anclar))
        for k in KS:
            for lam in LAMBDAS:
                v = np.mean([pred25(real_tot, ancla(base, k, lam,
                                                    np.random.default_rng(400 + s)))
                             for s in range(SORTEOS)])
                filas.append(dict(escenario=etq, k=k, lam=lam, PRED25=float(v)))
        print(f"  {etq} listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)

    for etq in ("sin estimacion", "CON estimacion"):
        d = t[(t.escenario == etq) & (t.k > 0)]
        base0 = float(t[(t.escenario == etq) & (t.k == 0)].PRED25.iloc[0])
        print(f"\n{S}\nPRED(25) · {etq.upper()} · sin anclar = {base0:.1%}\n{S}")
        print(d.pivot(index="lam", columns="k", values="PRED25").round(4).to_string())

    print(f"\n{S}\nQUE LAMBDA HACE FALTA EN CADA CASO\n{S}")
    recom = {}
    for etq in ("sin estimacion", "CON estimacion"):
        d = t[(t.escenario == etq) & (t.k > 0)]
        # el lambda que mejor va en promedio sobre los k pequenos, que es donde duele
        chico = d[d.k <= 5].groupby("lam").PRED25.mean()
        gl = d.groupby("lam").PRED25.mean()
        recom[etq] = dict(mejor_global=int(gl.idxmax()), mejor_k_pequeno=int(chico.idxmax()))
        print(f"  {etq:16s} mejor en promedio: lambda={gl.idxmax()} · "
              f"mejor con k<=5: lambda={chico.idxmax()}")
        base0 = float(t[(t.escenario == etq) & (t.k == 0)].PRED25.iloc[0])
        peor = d[d.PRED25 < base0]
        if len(peor):
            print(f"    con lambda=12 el anclaje EMPEORA en k = "
                  f"{sorted(peor[peor.lam == 12].k.tolist())}")
            l_rec = recom[etq]["mejor_k_pequeno"]
            p2 = peor[peor.lam == l_rec]
            print(f"    con lambda={l_rec} empeora en k = "
                  f"{sorted(p2.k.tolist()) if len(p2) else 'ninguno'}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "recomendacion": recom},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
