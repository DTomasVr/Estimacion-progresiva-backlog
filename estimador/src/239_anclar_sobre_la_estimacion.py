"""E119 - ¿Sobra el anclaje cuando hay estimacion del usuario?

E117 enseno algo incomodo: sobre lo que queda abierto, anadir el anclaje ENCIMA
de la estimacion del usuario la EMPEORA, y cada vez mas segun suben las tareas
cerradas (75,4 % -> 61,6 % con k=3; 79,5 % -> 48,0 % con k=20). Pero alli el
anclaje usaba lambda=12.

E86 ya habia diagnosticado esto a nivel de TOTAL y lo resolvio con lambda=96:
despues de la estimacion solo queda un 1,9 % de la varianza del residuo ENTRE
proyectos, asi que el anclaje casi no tiene nada que corregir y con lambda bajo
su ruido pesa mas que su senal. El artefacto ya lleva lambda=96 cuando hay
estimacion.

Falta comprobarlo donde importa y con datos limpios: el pronostico de lo
ABIERTO, con el control de contaminacion de E118 (evaluar y entrenar solo con
filas limpias).

  E   solo la estimacion, SIN anclar
  F12 estimacion + anclaje lambda=12    (lo que medimos en E117)
  F48 estimacion + anclaje lambda=48
  F96 estimacion + anclaje lambda=96    (lo desplegado)
  F192 estimacion + anclaje lambda=192

Si ninguno le gana a E, la conclusion es limpia: con estimacion del usuario el
anclaje SOBRA sobre lo abierto, y habria que apagarlo en vez de contraerlo.
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
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/anclar_sobre_estimacion")
S = "=" * 112
MIN_TAREAS, SORTEOS, PLIEGUES, COB = 8, 40, 5, .8
KS = (3, 5, 10, 20)
LAMS = (12, 48, 96, 192)
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def p25(r, p):
    return float(np.mean(np.abs(np.clip(p, 1e-6, None) - r) / r <= .25))


def mdm(r, p):
    return float(np.median(np.abs(np.clip(p, 1e-6, None) - r) / r))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, V, pred = sw[m].reset_index(drop=True), V[m], pred[m]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    limpio = hay & (np.abs(np.where(hay, est, 1) - y) / y > .01)
    cob = pd.Series(hay).groupby(proy).mean()
    nom = [p for p in pd.unique(proy) if cob.get(p, 0) >= COB]
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    print(f"{len(nom)} proyectos · control limpio de E118 · "
          f"{time.time() - t0:.0f} s", flush=True)

    # corrector entrenado SOLO con filas limpias (lo honesto, E118)
    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = pred.copy()
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        al = a[limpio[a]]
        bb = b[hay[b]]
        corr[bb] = HistGradientBoostingRegressor(
            **dict(PAR, loss="poisson")).fit(X[al], y[al]).predict(X[bb])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)
    corr = np.clip(corr, 1e-6, None)

    filas = []
    for k in KS:
        acc = {}
        for s in range(SORTEOS):
            rng = np.random.default_rng(2300 + s)
            dat = {}
            for p in nom:
                i = idx[p]
                if len(i) <= k:
                    continue
                sel = rng.choice(i, k, replace=False)
                rest = np.setdiff1d(i, sel)
                lim = rest[limpio[rest]]
                if len(lim) < 3:
                    continue
                dat.setdefault("R", []).append(y[lim].sum())
                base = corr[lim]
                dat.setdefault("E", []).append(base.sum())
                # el anclaje se estima sobre las CERRADAS, comparando su real
                # con lo que el corrector habria dicho de ellas
                pc = np.where(hay[sel], corr[sel], pred[sel])
                for lam in LAMS:
                    o = np.log(y[sel].sum() / pc.sum()) * k / (k + lam)
                    dat.setdefault(f"F{lam}", []).append(
                        (base * np.exp(o)).sum())
                # y el anclaje "clasico", estimado contra la prediccion en frio
                o12 = np.log(y[sel].sum() / pred[sel].sum()) * k / (k + 12)
                dat.setdefault("Ffrio12", []).append((base * np.exp(o12)).sum())
            R = np.array(dat["R"])
            for kk, v in dat.items():
                if kk == "R":
                    continue
                acc.setdefault(kk, []).append((p25(R, np.array(v)),
                                               mdm(R, np.array(v))))
        for kk, v in acc.items():
            filas.append(dict(k=k, brazo=kk,
                              PRED25=float(np.mean([a for a, _ in v])),
                              MdMRE=float(np.mean([b for _, b in v]))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    NOM = {"E": "E · solo estimacion, SIN anclar"}
    NOM.update({f"F{l}": f"F · + anclaje lambda={l}" for l in LAMS})
    NOM["Ffrio12"] = "F · anclaje contra el frio, lambda=12"
    g = t.pivot(index="k", columns="brazo", values="PRED25").rename(columns=NOM)
    cols = [NOM["E"]] + [NOM[f"F{l}"] for l in LAMS] + [NOM["Ffrio12"]]
    print(f"\n{S}\n±25 % DE LO ABIERTO · 41 proyectos, datos y corrector LIMPIOS"
          f"\n{S}")
    print(g[cols].round(4).to_string())
    gm = t.pivot(index="k", columns="brazo", values="MdMRE").rename(columns=NOM)
    print(f"\n{S}\nMdMRE (menos es mejor)\n{S}")
    print(gm[cols].round(4).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    e = g[NOM["E"]]
    for k in KS:
        mej = max(LAMS, key=lambda l: g.loc[k, NOM[f"F{l}"]])
        print(f"  k={k:2d}: sin anclar {e[k]:.1%} · mejor anclaje "
              f"lambda={mej} {g.loc[k, NOM[f'F{mej}']]:.1%} "
              f"({(g.loc[k, NOM[f'F{mej}']] - e[k]) * 100:+.1f} pts)")
    d96 = (g[NOM["F96"]] - e).mean() * 100
    dmej = max((g[NOM[f"F{l}"]] - e).mean() * 100 for l in LAMS)
    print(f"\n  media: lambda=96 (lo desplegado) {d96:+.1f} puntos · "
          f"el mejor lambda {dmej:+.1f} puntos")
    if dmej < .5:
        print("\n  NINGUN lambda le gana a no anclar. Con estimacion del usuario")
        print("  el anclaje SOBRA sobre lo abierto: habria que apagarlo, no")
        print("  contraerlo.")
    else:
        print("\n  Anclar sigue aportando si se contrae lo suficiente.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "delta_96": float(d96),
               "delta_mejor": float(dmej)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
