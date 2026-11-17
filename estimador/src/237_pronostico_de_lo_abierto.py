"""E117 - El pronostico de LO QUE QUEDA ABIERTO, que es lo que esta estancado.

E101 destapo que el +-25 % del TOTAL sube con las tareas cerradas (40,2 % con
3, 47,7 % con 10) pero el +-25 % de SOLO LO ABIERTO se queda clavado en ~36 %.
O sea que el total mejora sobre todo por contabilidad -cada vez mas parte de el
son horas ya conocidas- y no porque predigamos mejor lo que falta.

Aqui se ataca ese 36 %. Hay una hipotesis mecanica que lo explicaria y que se
puede comprobar: al cerrar mas tareas se conoce mejor el desplazamiento del
proyecto, PERO quedan menos tareas abiertas sobre las que promediar el ruido
por tarea. Los dos efectos van en direcciones opuestas y podrian cancelarse.

Lo que se mide:

  1 ¿DEPENDE DE n_abiertas?   si el pronostico de lo abierto es mejor cuando
    quedan muchas tareas, la hipotesis se sostiene y el estancamiento no es un
    techo del modelo sino una cancelacion
  2 LA ESTIMACION DEL USUARIO ¿rompe el estancamiento o tambien choca con el?
    es la pregunta que decide si hay palanca
  3 LAMBDA PARA LO ABIERTO    el 12 se calibro para el TOTAL, no para esto
  4 DESPLAZAMIENTO CON PENDIENTE  hoy el anclaje aplica un factor UNICO a todas
    las abiertas. Si el modelo comprime la escala, el factor deberia depender
    del tamano de la tarea: nivel + pendiente en log(prediccion)

  A  en frio, sin anclar
  B  anclaje actual, lambda=12
  C  anclaje con el lambda que le va a lo abierto
  D  anclaje con nivel Y pendiente
  E  la estimacion del usuario en las abiertas
  F  estimacion + anclaje
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
OUT = Path("results/pronostico_abierto")
S = "=" * 112
MIN_TAREAS, SORTEOS, PLIEGUES, COB = 8, 40, 5, .8
KS = (3, 5, 10, 20)
LAMBDAS = (3, 6, 12, 24, 48)
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
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    cob = pd.Series(hay).groupby(proy).mean()
    con_est = [p for p in nom if cob.get(p, 0) >= COB]
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · "
          f"{len(con_est)} con estimacion en >={COB:.0%} · "
          f"{time.time() - t0:.0f} s", flush=True)

    # corrector de la estimacion, fuera de pliegue (igual que E84)
    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = pred.copy()
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        aa = a[hay[a]]
        bb = b[hay[b]]
        corr[bb] = HistGradientBoostingRegressor(
            **dict(PAR, loss="poisson")).fit(X[aa], y[aa]).predict(X[bb])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)
    corr = np.clip(corr, 1e-6, None)

    filas, diag = [], []
    for k in KS:
        acc = {}
        for s in range(SORTEOS):
            rng = np.random.default_rng(1900 + s)
            for lista, etq_l in ((nom, "114"), (con_est, "43")):
                r_ab, pr = {}, {}
                for p in lista:
                    i = idx[p]
                    if len(i) <= k:
                        continue
                    sel = rng.choice(i, k, replace=False)
                    rest = np.setdiff1d(i, sel)
                    base = pred[rest]
                    lr = np.log(y[sel] / pred[sel])
                    real_ab = y[rest].sum()
                    r_ab.setdefault("real", []).append(real_ab)
                    r_ab.setdefault("n_ab", []).append(len(rest))
                    pr.setdefault("A", []).append(base.sum())
                    for lam in LAMBDAS:
                        o = np.log(y[sel].sum() / pred[sel].sum()) * k / (k + lam)
                        pr.setdefault(f"lam{lam}", []).append(
                            (base * np.exp(o)).sum())
                    # D: nivel y pendiente en log(prediccion)
                    if k >= 3 and np.std(np.log(pred[sel])) > 1e-6:
                        b1, b0 = np.polyfit(np.log(pred[sel]), lr, 1)
                        b1 *= k / (k + 24)          # la pendiente se contrae mas
                        b0 *= k / (k + 12)
                        pr.setdefault("D", []).append(
                            (base * np.exp(b0 + b1 * np.log(base))).sum())
                    else:
                        pr.setdefault("D", []).append(
                            pr[f"lam12"][-1])
                    if etq_l == "43":
                        ce = np.where(hay[rest], corr[rest], pred[rest])
                        pr.setdefault("E", []).append(ce.sum())
                        o12 = np.log(y[sel].sum() / pred[sel].sum()) * k / (k + 12)
                        pr.setdefault("F", []).append((ce * np.exp(o12)).sum())
                R = np.array(r_ab["real"])
                for kk, v in pr.items():
                    acc.setdefault((etq_l, kk), []).append(
                        (p25(R, np.array(v)), mdm(R, np.array(v))))
                if etq_l == "114" and s == 0:
                    diag.append((k, np.array(r_ab["n_ab"]), R,
                                 np.array(pr["lam12"])))
        for (etq_l, kk), v in acc.items():
            filas.append(dict(bloque=etq_l, brazo=kk, k=k,
                              PRED25=float(np.mean([a for a, _ in v])),
                              MdMRE=float(np.mean([b for _, b in v]))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    NOM = {"A": "A · en frio", "lam12": "B · anclaje actual (lam=12)",
           "D": "D · nivel y pendiente", "E": "E · estimacion del usuario",
           "F": "F · estimacion + anclaje"}
    print(f"\n{S}\n±25 % DE LO QUE QUEDA ABIERTO · 114 proyectos\n{S}")
    d = t[(t.bloque == "114") & (t.brazo.isin(["A", "lam12", "D"]))]
    print(d.pivot(index="k", columns="brazo", values="PRED25").rename(
        columns=NOM).round(4).to_string())
    print(f"\n{S}\nEL LAMBDA QUE LE VA A LO ABIERTO (hoy es 12)\n{S}")
    d = t[(t.bloque == "114") & (t.brazo.str.startswith("lam"))]
    print(d.pivot(index="k", columns="brazo", values="PRED25").round(4).to_string())
    print(f"\n{S}\n¿ROMPE EL ESTANCAMIENTO LA ESTIMACION? · 43 proyectos\n{S}")
    d = t[(t.bloque == "43") & (t.brazo.isin(["A", "lam12", "E", "F"]))]
    print(d.pivot(index="k", columns="brazo", values="PRED25").rename(
        columns=NOM).round(4).to_string())

    print(f"\n{S}\n¿DEPENDE DE CUANTAS TAREAS QUEDAN ABIERTAS?\n{S}")
    print(f"  {'k':>3}{'n_ab <= 20':>13}{'21-60':>10}{'61-200':>10}{'>200':>10}")
    for k, nab, R, P in diag:
        g = np.digitize(nab, [21, 61, 201])
        fila = [f"{p25(R[g == j], P[g == j]):.3f}" if (g == j).sum() >= 8 else "  -"
                for j in range(4)]
        print(f"  {k:>3}" + "".join(f"{x:>13}" if i == 0 else f"{x:>10}"
                                    for i, x in enumerate(fila)))

    print(f"\n{S}\nLECTURA\n{S}")
    b = t[(t.bloque == "43") & (t.brazo == "lam12")].set_index("k").PRED25
    e = t[(t.bloque == "43") & (t.brazo == "E")].set_index("k").PRED25
    f = t[(t.bloque == "43") & (t.brazo == "F")].set_index("k").PRED25
    for k in KS:
        print(f"  k={k:2d}: anclaje {b[k]:.1%} · estimacion {e[k]:.1%} · "
              f"las dos {f[k]:.1%}")
    if e.mean() - b.mean() > .05:
        print("\n  La estimacion SI rompe el estancamiento de lo abierto.")
    else:
        print("\n  La estimacion tampoco rompe el estancamiento: lo abierto esta")
        print("  limitado por el ruido por tarea, no por el desplazamiento.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "con_estimacion": len(con_est),
               "sorteos": SORTEOS}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
