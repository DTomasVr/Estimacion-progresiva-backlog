"""E82 - Repesar por densidad inversa del objetivo (DenseWeight), barrido cruzado con
la cantidad de proyectos de entrenamiento.

El modelo del total se entrena con perdida de Poisson sobre horas, que pesa muchisimo
los valores grandes: equivocarse 0,1 h en una tarea de 0,14 h no le cuesta casi nada,
y por eso predice ~5 h para todo el tramo corto. La literatura de regresion
desbalanceada (DenseWeight / DenseLoss, Steininger et al.; LDS/FDS, Yang et al.)
propone lo contrario: pesar cada ejemplo por el INVERSO de la densidad de su objetivo.

    p~(y)  = densidad de log(y) estimada con nucleo, normalizada a [0,1]
    w(y)   = max(eps, 1 - alpha * p~(y))        alpha = 0 es el modelo de hoy

Se barre alpha y, a la vez, el NUMERO DE PROYECTOS de entrenamiento, que es la otra
pregunta abierta. La evaluacion es siempre sobre los 114, con GroupKFold por proyecto
entero: lo que cambia es cuantos proyectos entran a entrenar dentro de cada pliegue.

La densidad se estima SOLO con las etiquetas de entrenamiento de cada pliegue.

Se mira lo que tiene que mejorar (el sesgo en las tareas cortas) y lo que no se puede
estropear (el total del proyecto y su compresion de escala).
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

# 3 hilos por trabajador x 4 trabajadores = los 12 logicos, sin estorbarse
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "3"

import joblib  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/repeso_densidad")
S = "=" * 120
PLIEGUES, MIN_TAREAS, EPS = 5, 8, .05
ALFAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
PROYECTOS = (40, 70, 0)                 # 0 = todos los del pliegue
TRAMOS = [0, .25, .5, 1, 2, 4, 8, 16, 1e9]
ET = ["<0,25", "0,25-0,5", "0,5-1", "1-2", "2-4", "4-8", "8-16", ">16"]


def pesos(y_tr, alfa):
    """DenseWeight: peso inversamente proporcional a la densidad del objetivo."""
    if alfa <= 0:
        return np.ones(len(y_tr))
    ly = np.log(y_tr)
    p = gaussian_kde(ly)(ly)
    p = (p - p.min()) / max(p.max() - p.min(), 1e-12)
    w = np.maximum(EPS, 1.0 - alfa * p)
    return w / w.mean()


def remuestrea(a, y_tr, alfa, semilla):
    """Aplicar el peso REMUESTREANDO en vez de con sample_weight.

    Medido en esta maquina: ajustar con sample_weight tarda 147,5 s y remuestrear
    14,3 s, diez veces menos, y ademas con pesos el ajuste se cortaba antes (67
    iteraciones contra 200). Para un ensamble de arboles las dos cosas son
    equivalentes, asi que no hay motivo para pagar ese precio.
    """
    if alfa <= 0:
        return a
    w = pesos(y_tr, alfa)
    w = w / w.sum()
    rng = np.random.default_rng(semilla)
    return a[rng.choice(len(a), len(a), replace=True, p=w)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_pbe_v3.joblib")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tam = sw.groupby("project").size()
    k = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw = sw[k].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")[k]
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    X = np.hstack([E5, SV]).astype(np.float32)
    nom = list(pd.unique(proy))
    real_tot = pd.Series(y).groupby(proy).sum().reindex(nom).to_numpy()
    real_esc = np.log(pd.Series(y).groupby(proy).mean().reindex(nom).to_numpy())
    tramo = pd.cut(y, TRAMOS, labels=ET, right=False)
    print(f"{len(sw):,} tareas · {len(nom)} proyectos · X {X.shape} · "
          f"{time.time() - t0:.0f} s", flush=True)

    pliegues = list(GroupKFold(PLIEGUES).split(X, y, proy))
    n_cfg = len(ALFAS) * len(PROYECTOS)
    print(f"{n_cfg} configuraciones x {PLIEGUES} pliegues = {n_cfg * PLIEGUES} ajustes "
          f"· ~{n_cfg * PLIEGUES * 9 / 60:.0f} min estimados", flush=True)

    # Las 18 configuraciones son independientes: se reparten en procesos en vez de
    # correrlas en fila. Con 6 nucleos fisicos, 4 trabajadores x 3 hilos es lo que
    # cabe sin que se estorben entre si.
    def una_config(n_pro, alfa):
        p_oof = np.zeros(len(y))
        for nf, (a, b) in enumerate(pliegues):
            pr_tr = np.unique(proy[a])
            if n_pro and n_pro < len(pr_tr):
                rng = np.random.default_rng(1000 + nf)
                elegidos = set(rng.choice(pr_tr, n_pro, replace=False))
                a = a[np.array([q in elegidos for q in proy[a]])]
            a2 = remuestrea(a, y[a], alfa, 7000 + nf)
            m = HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=.06,
                l2_regularization=1., loss="poisson", random_state=0)
            m.fit(X[a2], y[a2])
            p_oof[b] = m.predict(X[b])
        p_oof = np.clip(p_oof, 1e-6, None)

        d = pd.DataFrame({"t": tramo, "y": y, "p": p_oof}).groupby("t", observed=True)
        sesgo = (d.p.mean() / d.y.mean()).reindex(ET)
        est_tot = pd.Series(p_oof).groupby(proy).sum().reindex(nom).to_numpy()
        r = np.abs(est_tot - real_tot) / real_tot
        mlog = float(np.median(np.log(est_tot / real_tot)))
        est_esc = np.log(pd.Series(p_oof).groupby(proy).mean().reindex(nom).to_numpy())
        fila = dict(
            alfa=alfa, proyectos=n_pro or len(nom),
            sesgo_corto=float(sesgo["<0,25"]), sesgo_05=float(sesgo["0,25-0,5"]),
            sesgo_1=float(sesgo["0,5-1"]), sesgo_largo=float(sesgo[">16"]),
            tarea_MdMRE=float(np.median(np.abs(p_oof - y) / y)),
            tarea_PRED25=float(np.mean(np.abs(p_oof - y) / y <= .25)),
            total_MdMRE=float(np.median(r)), total_PRED25=float(np.mean(r <= .25)),
            total_factor2=float(np.mean(np.maximum(est_tot / real_tot,
                                                   real_tot / est_tot) <= 2)),
            total_SSPB=float(np.sign(mlog) * (np.exp(abs(mlog)) - 1) * 100),
            esc_sd=float(est_esc.std()),
            esc_spearman=float(pd.Series(est_esc).corr(pd.Series(real_esc),
                                                       method="spearman")))
        print(f"  a={alfa}|p={n_pro or 'todos'}  total ±25% {fila['total_PRED25']:.3f} "
              f"· sesgo corto x{fila['sesgo_corto']:.1f} · sd escala "
              f"{fila['esc_sd']:.3f} · {time.time() - t0:.0f} s", flush=True)
        return fila, f"a={alfa}|p={n_pro or 'todos'}", p_oof.astype(np.float32)

    cfgs = [(n_pro, alfa) for n_pro in PROYECTOS for alfa in ALFAS]
    res = Parallel(n_jobs=4, backend="loky", verbose=0)(
        delayed(una_config)(n_pro, alfa) for n_pro, alfa in cfgs)
    filas = [x[0] for x in res]
    preds = {x[1]: x[2] for x in res}

    t = pd.DataFrame(filas)
    print(f"\n{S}\nBARRIDO: REPESO POR DENSIDAD x CANTIDAD DE PROYECTOS\n{S}")
    cols = ["alfa", "proyectos", "sesgo_corto", "sesgo_1", "sesgo_largo",
            "tarea_MdMRE", "tarea_PRED25", "total_PRED25", "total_MdMRE",
            "total_SSPB", "esc_sd", "esc_spearman"]
    print(t[cols].round(4).to_string(index=False))
    print(f"\n  referencia · sd REAL de la escala: {real_esc.std():.3f}")

    print(f"\n{S}\nLECTURA\n{S}")
    base = t[(t.alfa == 0) & (t.proyectos == len(nom))].iloc[0]
    print(f"  modelo de hoy (alpha=0, todos): sesgo corto x{base.sesgo_corto:.1f} · "
          f"total ±25 % {base.total_PRED25:.1%} · sd escala {base.esc_sd:.3f}")
    todos = t[t.proyectos == len(nom)]
    mej = todos.sort_values("total_PRED25", ascending=False).iloc[0]
    print(f"  mejor por total ±25 %: alpha={mej.alfa} -> {mej.total_PRED25:.1%} "
          f"({(mej.total_PRED25 - base.total_PRED25) * 100:+.1f} pts)")
    mejc = todos.sort_values("sesgo_corto").iloc[0]
    print(f"  mejor por sesgo en tareas cortas: alpha={mejc.alfa} -> "
          f"x{mejc.sesgo_corto:.1f} (hoy x{base.sesgo_corto:.1f}), y su total "
          f"±25 % es {mejc.total_PRED25:.1%}")
    print(f"\n  efecto de la cantidad de proyectos, a alpha=0:")
    for _, x in t[t.alfa == 0].sort_values("proyectos").iterrows():
        print(f"    {int(x.proyectos):>3} proyectos -> total ±25 % {x.total_PRED25:.1%} "
              f"· sd escala {x.esc_sd:.3f}")
    t.to_csv(OUT / "resumen.csv", index=False)
    np.savez_compressed(OUT / "predicciones.npz", **preds)
    json.dump({"alfas": list(ALFAS), "proyectos": list(PROYECTOS),
               "sd_real": float(real_esc.std()), "n": int(len(sw))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
