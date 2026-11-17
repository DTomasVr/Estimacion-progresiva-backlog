"""E85 - ¿Se suman las dos palancas? Estimacion del usuario MAS tareas cerradas.

E84 dejo claro que son mecanismos distintos:

  tareas cerradas          corrigen un desplazamiento GLOBAL -> el 11 % de varianza
                           que esta ENTRE proyectos
  estimacion del usuario   informacion POR TAREA -> parte del 89 % que esta DENTRO

Si atacan partes distintas de la varianza, deberian sumarse. Y si la estimacion del
usuario ya lleva implicita la escala del equipo -quien trabaja en tareas cortas las
estima cortas-, el anclaje aportaria menos encima de ella que encima del frio.

Las dos cosas son medibles, y la segunda ademas se puede comprobar directamente:
basta descomponer la varianza del residuo DESPUES de aplicar la estimacion del
usuario y ver si sigue habiendo un 11 % entre proyectos o si ya se ha ido.

  A  en frio
  B  solo k tareas cerradas
  C  solo la estimacion del usuario (todas las tareas)
  D  las dos a la vez

Mismo corrector de Poisson fuera de pliegue que E84, y la METRICA promediada sobre
los sorteos de que tareas se cerraron primero.
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
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/combinar_palancas")
S = "=" * 104
LAMBDA, SORTEOS, PLIEGUES, COB = 12, 60, 5, .8
KS = (3, 5, 10)


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def reparto(rho, idx, nom):
    """Que parte de la varianza del residuo esta ENTRE proyectos y que parte DENTRO."""
    n = np.array([len(idx[p]) for p in nom], float)
    med = np.array([rho[idx[p]].mean() for p in nom])
    gran = float(np.average(med, weights=n))
    entre = float(np.average((med - gran) ** 2, weights=n))
    dentro = float(np.average([rho[idx[p]].var() for p in nom], weights=n))
    return entre, dentro, entre / (entre + dentro)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
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
    frio = np.array([pred[idx[p]].sum() for p in nom])
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · {time.time() - t0:.0f} s",
          flush=True)

    X = np.hstack([V, np.log(np.where(hay, est, 1))[:, None],
                   np.log(pred)[:, None]]).astype(np.float32)
    corr = pred.copy()
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, real, proy)):
        a = a[hay[a]]
        mo = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                           l2_regularization=1., loss="poisson",
                                           random_state=0).fit(X[a], real[a])
        bb = b[hay[b]]
        corr[bb] = mo.predict(X[bb])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)
    corr = np.clip(corr, 1e-6, None)

    # ---- ¿sigue habiendo desplazamiento de proyecto despues de la estimacion? ----
    e1, d1, f1 = reparto(np.log(real / pred), idx, nom)
    e2, d2, f2 = reparto(np.log(real / corr), idx, nom)
    print(f"\n{S}\nQUE QUEDA DEL RESIDUO DESPUES DE LA ESTIMACION DEL USUARIO\n{S}")
    print(f"  {'':34s}{'entre proy.':>13}{'dentro':>10}{'% entre':>10}")
    print(f"  {'residuo del modelo en frio':34s}{e1:>13.4f}{d1:>10.4f}{f1:>9.1%}")
    print(f"  {'residuo tras la estimacion':34s}{e2:>13.4f}{d2:>10.4f}{f2:>9.1%}")
    print(f"  la varianza DENTRO baja un {(1 - d2 / d1):.0%}; la de ENTRE, un "
          f"{(1 - e2 / e1):.0%}")

    def ancla(base, k, rng, usar_real=True):
        t_ = np.empty(len(nom))
        for j, p in enumerate(nom):
            i = idx[p]
            if len(i) <= k:
                t_[j] = base[i].sum()
                continue
            sel = rng.choice(i, k, replace=False)
            resto = np.setdiff1d(i, sel)
            off = np.log(real[sel].sum() / base[sel].sum()) * k / (k + LAMBDA)
            t_[j] = real[sel].sum() + (base[resto] * np.exp(off)).sum()
        return t_

    filas = [dict(brazo="A · en frio", **mt(real_tot, frio)),
             dict(brazo="C · solo estimacion del usuario",
                  **mt(real_tot, np.array([corr[idx[p]].sum() for p in nom])))]
    for k in KS:
        for etq, base in ((f"B · solo {k} tareas cerradas", pred),
                          (f"D · estimacion + {k} cerradas", corr)):
            ms = []
            for s in range(SORTEOS):
                ms.append(mt(real_tot, ancla(base, k, np.random.default_rng(400 + s))))
            filas.append(dict(brazo=etq,
                              **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))
    t = pd.DataFrame(filas)
    print(f"\n{S}\n¿SE SUMAN LAS DOS PALANCAS? ({len(nom)} proyectos)\n{S}")
    print(t[["brazo", "PRED25", "MdMRE", "factor2", "MAE_h", "SSPB_pc"]].round(4)
          .to_string(index=False))

    a = float(t[t.brazo.str.startswith("A")].PRED25.iloc[0])
    c = float(t[t.brazo.str.startswith("C")].PRED25.iloc[0])
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  en frio {a:.1%} · solo estimacion {c:.1%} ({(c - a) * 100:+.1f})")
    for k in KS:
        b = float(t[t.brazo.eq(f"B · solo {k} tareas cerradas")].PRED25.iloc[0])
        d = float(t[t.brazo.eq(f"D · estimacion + {k} cerradas")].PRED25.iloc[0])
        suma = (b - a) + (c - a)
        real_g = d - a
        print(f"  k={k:2d}: cerradas {b:.1%} ({b - a:+.1%}) · estimacion "
              f"{c - a:+.1%} · JUNTAS {d:.1%} ({real_g:+.1%})")
        print(f"        si se sumaran darian {a + suma:.1%}; se consigue el "
              f"{real_g / suma * 100:.0f} % de esa suma")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "frac_entre_frio": f1,
               "frac_entre_corr": f2}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
