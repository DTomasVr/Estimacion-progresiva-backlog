"""E120b - e5-large contra e5-small con la misma receta, todo fuera de pliegue.

Antes de reconstruir la cadena v3 -> v6 con el small, se mide si aguanta. Mismo
protocolo que el resto de la serie: GroupKFold(5) por proyecto entero sobre las
24.848 tareas sin Apache, objetivo log-horas.

Canales, cada uno para large y para small:
  base      HGB sobre el codificador sin ajustar + los dos SVD del TF-IDF
  C         HGB sobre el codificador ajustado POR PLIEGUE (160 / 248)
  analogia  mediana de horas de los 25 vecinos de entrenamiento (coseno, base)
  mezcla    media en log de base y C, y despues ajuste a la rejilla de 30 valores
            de cada pliegue de entrenamiento

Mismas filas y mismos pliegues para los dos: la diferencia es solo el codificador.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/large_contra_small")
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
K_VEC, N_REJ = 25, 30


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                spearman=float(spearmanr(p, y).statistic))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    p6 = joblib.load("models/estimador_agix_v6.joblib")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((p6["tfc"], p6["tfw"]), p6["svds"])])
    emb = {"large": (np.load(CACHE / "e5-large.npy"),
                     np.load(CACHE / "e5-ajustado-C-porpliegue.npy")),
           "small": (np.load(CACHE / "e5-small.npy"),
                     np.load(CACHE / "e5-small-C-porpliegue.npy"))}
    for k, (b, c) in emb.items():
        assert len(b) == len(c) == len(y), k
    print(f"{len(y):,} tareas · {time.time() - t0:.0f} s", flush=True)

    pliegues = list(GroupKFold(5).split(np.zeros(len(y)), y, proy))
    P = {}
    for k, (B, C) in emb.items():
        Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
        XB = np.hstack([B, SV])
        pb, pc, pa, pm = (np.zeros(len(y)) for _ in range(4))
        for nf, (a, t) in enumerate(pliegues):
            pb[t] = HistGradientBoostingRegressor(**PAR).fit(XB[a], ly[a]).predict(XB[t])
            pc[t] = HistGradientBoostingRegressor(**PAR).fit(C[a], ly[a]).predict(C[t])
            for i in range(0, len(t), 2048):
                s = Bn[t[i:i + 2048]] @ Bn[a].T
                j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
                pa[t[i:i + 2048]] = np.log(np.median(y[a][j], axis=1))
            rej = np.sort(pd.Series(y[a]).value_counts().index.to_numpy()[:N_REJ])
            m = (pb[t] + pc[t]) / 2
            pm[t] = rej[np.abs(np.log(rej)[None, :] - m[:, None]).argmin(1)]
            print(f"  {k} pliegue {nf + 1}/5 · {time.time() - t0:.0f} s", flush=True)
        P[k] = {"base": np.exp(pb), "C": np.exp(pc), "analogia": np.exp(pa),
                "mezcla": np.exp((pb + pc) / 2), "mezcla+rejilla": pm}

    filas = [{"codificador": k, "canal": c, **met(y, v)}
             for k, d in P.items() for c, v in d.items()]
    filas.append({"codificador": "-", "canal": "constante 2,45 h",
                  **met(y, np.full(len(y), 2.45))})
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "metricas.csv", index=False)
    np.savez_compressed(OUT / "predicciones.npz",
                        **{f"{k}_{c}": v for k, d in P.items() for c, v in d.items()})
    pd.set_option("display.width", 200)
    print(r.round(4).to_string(index=False))

    # remuestreo por proyecto de la diferencia en la mezcla con rejilla
    rng = np.random.default_rng(0)
    up = np.unique(proy)
    idx = {p: np.flatnonzero(proy == p) for p in up}
    dl = np.abs(P["large"]["mezcla+rejilla"] - y) / y <= .25
    ds = np.abs(P["small"]["mezcla+rejilla"] - y) / y <= .25
    el = np.abs(np.log(P["large"]["mezcla"] / y))
    es = np.abs(np.log(P["small"]["mezcla"] / y))
    bp, bm = [], []
    for _ in range(2000):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))])
        bp.append(ds[s].mean() - dl[s].mean())
        bm.append(el[s].mean() - es[s].mean())
    res = {"dif_PRED25_small_menos_large": float(ds.mean() - dl.mean()),
           "IC95_PRED25": [float(np.quantile(bp, .025)), float(np.quantile(bp, .975))],
           "dif_errorlog_large_menos_small": float(el.mean() - es.mean()),
           "IC95_errorlog": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]}
    json.dump(res, open(OUT / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"· {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
