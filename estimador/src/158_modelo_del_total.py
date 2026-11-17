"""E52 - Un modelo entrenado PARA el total, no reciclado de la prediccion por tarea.

Toda la cadena esta entrenada para acertar una tarea suelta: error cuadratico
sobre log-horas, que da la mediana condicional. Para el total hace falta la MEDIA,
y hasta ahora se conseguia de dos formas de segunda mano: empujando la mediana con
un factor de smearing (E43) o metiendo un unico HGB con perdida de Poisson (E45).

Ese HGB de Poisson gano el MdMRE del total (0,368 frente a 0,438) siendo un solo
arbol contra un apilado de cuatro canales. Si un modelo suelto entrenado para el
objetivo correcto le gana a un conjunto entrenado para el objetivo equivocado, lo
que toca es montar el conjunto entero para el objetivo correcto.

  canales, todos con perdida de Poisson sobre HORAS (no sobre log):
    bolsa de 5 HGB sobre e5-large + TF-IDF reducido
    HGB sobre TF-IDF reducido de caracteres
    HGB sobre la vista de analogia (la mediana de los k vecinos como rasgo)
  apilado: Poisson sobre los canales, ajustado en una particion interna por
           proyecto para no tocar los proyectos de prueba

Se compara contra los tres estimadores que ya existen, sobre los mismos 114
proyectos y con la misma validacion cruzada por proyecto entero.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/modelo_total")
S = "=" * 122
MIN_TAREAS, PLIEGUES, N_BOLSA, K_VEC = 8, 5, 5, 25


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def poi(semilla=0, **kw):
    p = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
             loss="poisson", random_state=semilla)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_pbe_v3.joblib")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    X = np.hstack([E5, SV])
    print(f"{len(sw):,} tareas · {X.shape[1]} rasgos · {time.time() - t0:.0f} s", flush=True)

    pred = np.zeros(len(sw))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        rng = np.random.default_rng(nf)
        pr = np.unique(proy[a])
        pin = set(rng.choice(pr, max(2, int(.2 * len(pr))), replace=False))
        ain = a[np.array([p in pin for p in proy[a]])]
        afu = a[np.array([p not in pin for p in proy[a]])]
        bolsa = []
        for s in range(N_BOLSA):
            sub = rng.choice(afu, int(.85 * len(afu)), replace=False)
            bolsa.append(poi(s, max_features=.8).fit(X[sub], y[sub]))
        sv_only = poi().fit(SV[afu], y[afu])
        # canal de analogia: la mediana de los k vecinos como rasgo unico
        A = Vn[afu]
        yv = y[afu]

        def vecino(sel):
            out = np.zeros(len(sel))
            for i in range(0, len(sel), 2048):
                s = Vn[sel[i:i + 2048]] @ A.T
                j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
                out[i:i + 2048] = np.median(yv[j], axis=1)
            return out
        ana_fu, ana_in, ana_b = vecino(afu), vecino(ain), vecino(b)
        ana_m = poi().fit(ana_fu[:, None], y[afu])

        def canales(sel, ana):
            return np.column_stack([np.mean([m.predict(X[sel]) for m in bolsa], 0),
                                    sv_only.predict(SV[sel]),
                                    ana_m.predict(ana[:, None])])
        ap = poi(max_iter=120, max_depth=3).fit(canales(ain, ana_in), y[ain])
        pred[b] = ap.predict(canales(b, ana_b))
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    lp = np.load("results/analogia_cv/pred.npy")[:, 0]
    e = np.exp(ly - lp)
    tr = pd.qcut(lp, 6, labels=False, duplicates="drop")
    sm = np.exp(lp) * pd.DataFrame({"t": tr, "e": e}).groupby("t").e.mean(
    ).reindex(tr).to_numpy()
    d = pd.DataFrame({"p": proy, "real": y, "nuevo": np.clip(pred, 1e-6, None),
                      "sm": sm})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    tot = d.groupby("p").sum(numeric_only=True)
    ant = pd.read_csv("results/combinar_total/resumen.csv").set_index("brazo")

    filas = [dict(brazo="modelo entrenado PARA el total (Poisson apilado)",
                  **mt(tot.real, tot.nuevo)),
             dict(brazo="ascendente smearing (lo anterior)", **mt(tot.real, tot.sm))]
    t = pd.DataFrame(filas)
    print(f"\n{S}\nTOTAL DEL PROYECTO ({len(tot)} proyectos)\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    print("\n  para comparar, lo mejor que habia (E50):")
    for k in ("combinada, pesos para MdMRE", "ascendente Poisson"):
        if k in ant.index:
            r = ant.loc[k]
            print(f"    {k:34s} MdMRE {r.MdMRE:.4f} · R2 {r.R2_log:.4f} · "
                  f"PRED25 {r.PRED25:.4f} · factor2 {r.factor2:.4f}")
    np.save(OUT / "pred_total.npy", pred)
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n_bolsa": N_BOLSA, "k_vec": K_VEC}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
