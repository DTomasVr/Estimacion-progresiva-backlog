"""E118 - El control de E93 aplicado a lo ABIERTO, que es donde mas puede doler.

E117 midio que la estimacion del usuario lleva el +-25 % de lo que queda
abierto de ~40 % con anclaje a 70-79 %. Es la unica palanca que mejora el
pronostico de lo que FALTA y no solo la aritmetica de lo ya hecho.

Pero falta el control de E93: en el 34,7 % de las tareas con estimacion, las
horas reales son la estimacion COPIADA, porque la tarea se cierra imputando lo
estimado. En esas filas el corrector acierta de regalo. Si abundan entre las
tareas abiertas, el 70-79 % esta inflado.

Hay DOS contaminaciones distintas y las dos hay que quitar:

  al EVALUAR    si la tarea abierta tiene el real copiado, acertarla no mide
                nada. Se evalua solo sobre las abiertas limpias
  al ENTRENAR   el corrector aprendio "devuelve la estimacion" de esas mismas
                filas. Se reentrena solo con filas limpias

  A  en frio
  B  anclaje (lambda=12)
  E1 estimacion · corrector entrenado con TODO, evaluado sobre TODO   (E117)
  E2 estimacion · corrector entrenado con TODO, evaluado sobre LIMPIAS
  E3 estimacion · corrector entrenado y evaluado LIMPIO               <- el honesto

Si E3 aguanta por encima del anclaje, la palanca es real. Si se desploma hasta
el, lo de E117 era el artefacto.
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
OUT = Path("results/abierto_sin_contaminar")
S = "=" * 112
MIN_TAREAS, SORTEOS, PLIEGUES, COB, LAMBDA = 8, 40, 5, .8, 12
KS = (3, 5, 10, 20)
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def p25(r, p):
    return float(np.mean(np.abs(np.clip(p, 1e-6, None) - r) / r <= .25))


def mdm(r, p):
    return float(np.median(np.abs(np.clip(p, 1e-6, None) - r) / r))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
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
    print(f"{len(nom)} proyectos con estimacion · "
          f"limpias {limpio[hay].mean():.1%} de las que la traen · "
          f"{time.time() - t0:.0f} s", flush=True)
    # cuantas quedan limpias por proyecto
    nl = np.array([limpio[idx[p]].sum() for p in nom])
    print(f"  tareas limpias por proyecto: mediana {int(np.median(nl))} · "
          f"p10 {int(np.quantile(nl, .1))} · p90 {int(np.quantile(nl, .9))}",
          flush=True)

    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr_t = pred.copy()      # entrenado con TODO
    corr_l = pred.copy()      # entrenado solo con LIMPIAS
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        bb = b[hay[b]]
        corr_t[bb] = HistGradientBoostingRegressor(
            **dict(PAR, loss="poisson")).fit(X[a[hay[a]]], y[a[hay[a]]]).predict(X[bb])
        al = a[limpio[a]]
        corr_l[bb] = HistGradientBoostingRegressor(
            **dict(PAR, loss="poisson")).fit(X[al], y[al]).predict(X[bb])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)
    corr_t = np.clip(corr_t, 1e-6, None)
    corr_l = np.clip(corr_l, 1e-6, None)

    filas = []
    for k in KS:
        acc = {}
        for s in range(SORTEOS):
            rng = np.random.default_rng(2100 + s)
            dat = {}
            for p in nom:
                i = idx[p]
                if len(i) <= k:
                    continue
                sel = rng.choice(i, k, replace=False)
                rest = np.setdiff1d(i, sel)
                off = np.log(y[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
                # -- sobre TODAS las abiertas
                dat.setdefault("R_t", []).append(y[rest].sum())
                dat.setdefault("A_t", []).append(pred[rest].sum())
                dat.setdefault("B_t", []).append((pred[rest] * np.exp(off)).sum())
                dat.setdefault("E1", []).append(
                    np.where(hay[rest], corr_t[rest], pred[rest]).sum())
                # -- sobre las abiertas LIMPIAS
                lim = rest[limpio[rest]]
                if len(lim) < 3:
                    for kk in ("R_l", "A_l", "B_l", "E2", "E3"):
                        dat.setdefault(kk, []).append(np.nan)
                    continue
                dat.setdefault("R_l", []).append(y[lim].sum())
                dat.setdefault("A_l", []).append(pred[lim].sum())
                dat.setdefault("B_l", []).append((pred[lim] * np.exp(off)).sum())
                dat.setdefault("E2", []).append(corr_t[lim].sum())
                dat.setdefault("E3", []).append(corr_l[lim].sum())
            Rt = np.array(dat["R_t"])
            Rl = np.array(dat["R_l"])
            ok = ~np.isnan(Rl)
            for etq, v, R, msk in (
                    ("A · en frio (todas)", "A_t", Rt, slice(None)),
                    ("B · anclaje (todas)", "B_t", Rt, slice(None)),
                    ("E1 · estimacion, todo contaminado", "E1", Rt, slice(None)),
                    ("A' · en frio, solo limpias", "A_l", Rl, ok),
                    ("B' · anclaje, solo limpias", "B_l", Rl, ok),
                    ("E2 · entrenado con todo, evaluado limpio", "E2", Rl, ok),
                    ("E3 · entrenado y evaluado LIMPIO", "E3", Rl, ok)):
                P = np.array(dat[v])
                acc.setdefault(etq, []).append(
                    (p25(R[msk], P[msk]), mdm(R[msk], P[msk])))
        for etq, v in acc.items():
            filas.append(dict(k=k, brazo=etq,
                              PRED25=float(np.mean([a for a, _ in v])),
                              MdMRE=float(np.mean([b for _, b in v]))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\n±25 % DE LO QUE QUEDA ABIERTO\n{S}")
    print(t.pivot(index="k", columns="brazo", values="PRED25").round(4).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    g = t.pivot(index="k", columns="brazo", values="PRED25")
    cb, c1, c3 = ("B' · anclaje, solo limpias",
                  "E1 · estimacion, todo contaminado",
                  "E3 · entrenado y evaluado LIMPIO")
    for k in KS:
        print(f"  k={k:2d}: anclaje limpio {g.loc[k, cb]:.1%} · "
              f"E1 contaminado {g.loc[k, c1]:.1%} · "
              f"E3 LIMPIO {g.loc[k, c3]:.1%}")
    e1, e3, b_ = g[c1].mean(), g[c3].mean(), g[cb].mean()
    print(f"\n  media: anclaje {b_:.1%} · contaminado {e1:.1%} · limpio {e3:.1%}")
    if e3 - b_ > .08:
        print("\n  La palanca AGUANTA el control: la estimacion sigue muy por")
        print("  encima del anclaje con los datos limpios.")
    elif e3 - b_ > .02:
        print("\n  Aguanta, pero la ventaja se encoge bastante al quitar las")
        print("  filas copiadas. Hay que reportar la cifra limpia.")
    else:
        print("\n  NO aguanta: casi todo lo que ganaba la estimacion sobre lo")
        print("  abierto eran las filas con el real copiado.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "limpias": float(limpio[hay].mean()),
               "anclaje": float(b_), "contaminado": float(e1), "limpio": float(e3)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
