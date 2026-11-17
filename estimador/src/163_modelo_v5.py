"""E57 - El modelo final v5: el apilado de siempre mas el codificador del brazo C.

E35 brazo C aporta, y la mejor configuracion medida es llevar LOS DOS
codificadores ajustados: el escalar cuadratico que ya habia y el nuevo, entrenado
con cabeza de clasificacion sobre la rejilla mas una perdida de ordenacion por
pares. Uno ordena mejor y el otro estima mejor la escala, y el apilado usa las
dos cosas.

Aqui se reconstruye el modelo final con ese canal anadido, se guarda, y se vuelve
a sacar el panel de metricas completo para que las cifras por tarea del deck
salgan del modelo que de verdad se despliega.

Se limita a 6 hilos porque esto corre mientras la GPU ajusta los codificadores
por pliegue, y saturar los 12 nucleos deja sin CPU al hilo que la alimenta.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "6"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.svm import LinearSVR  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/modelo_v5")
S = "=" * 126
N_BOLSA = 7


def azar(y, corridas=600, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(y)
    mars, err = [], []
    for c in range(corridas):
        i = rng.integers(0, n - 1, n)
        i += i >= np.arange(n)
        e = np.abs(y - y[i])
        mars.append(e.mean())
        if c < 40:
            err.append(e)
    return float(np.mean(mars)), float(np.std(np.concatenate(err)))


def panel(y, p, ref=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    e = np.abs(y - p)
    q = np.log(p / y)
    m = float(np.median(q))
    d = dict(MAE_h=float(e.mean()), RMSE_h=float(np.sqrt(np.mean((y - p) ** 2))),
             MdAPE_pc=float(np.median(e / y) * 100))
    if ref is not None:
        m0, s0 = ref
        d["SA"] = float(1 - e.mean() / m0)
        d["delta_Glass"] = float((m0 - e.mean()) / s0)
    d["zeta_pc"] = float((np.exp(np.median(np.abs(q))) - 1) * 100)
    d["SSPB_pc"] = float(np.sign(m) * (np.exp(abs(m)) - 1) * 100)
    d["R2_log"] = float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var())
    d["PRED25_pc"] = float(np.mean(e / y <= .25) * 100)
    d["factor2_pc"] = float(np.mean(np.maximum(p / y, y / p) <= 2) * 100)
    d["spearman"] = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    return d


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


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
    tr = (sw.split_t3 == "train").to_numpy()
    va = (sw.split_t3 == "val").to_numpy()
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    E5 = np.load(CACHE / "e5-large.npy")
    A = np.load(CACHE / "e5-ajustado.npy")
    C = np.load(CACHE / "e5-ajustado-C.npy")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    viejo = f["apilado"].predict(np.column_stack([
        np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
        f["ridge"].predict(Xs), f["svr"].predict(Xs), f["hgb_ajustado"].predict(A)]))
    print(f"datos listos · {time.time() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    bolsa = []
    for s in range(N_BOLSA):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
    lr = Ridge(alpha=2.).fit(Xs[tr], ly[tr])
    ls = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Xs[tr], ly[tr])
    ha = hgb().fit(A[tr], ly[tr])
    hc = hgb().fit(C[tr], ly[tr])
    canal = {"bolsa": np.mean([m.predict(Xd) for m in bolsa], 0),
             "ridge disperso": lr.predict(Xs), "SVR disperso": ls.predict(Xs),
             "e5 ajustado A": ha.predict(A), "e5 ajustado C": hc.predict(C)}
    Z = np.column_stack(list(canal.values()))
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va], ly[va])
    nuevo = ap.predict(Z)
    print("  pesos: " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(canal, ap.coef_)),
          flush=True)

    filas = []
    for b, k in bl.items():
        ref = azar(y[k])
        for etq, p in (("v3 · el que habia", np.exp(viejo)),
                       ("v5 · con el codificador C", np.exp(nuevo))):
            filas.append(dict(bloque=b, modelo=etq, n=int(k.sum()), **panel(y[k], p[k], ref)))
        filas.append(dict(bloque=b, modelo="adivinar al azar", n=int(k.sum()),
                          **panel(y[k], np.full(k.sum(), float(np.median(y[k]))), ref)))
    t = pd.DataFrame(filas)
    cols = ["modelo", "MAE_h", "RMSE_h", "SA", "delta_Glass", "zeta_pc", "SSPB_pc",
            "R2_log", "PRED25_pc", "factor2_pc", "spearman"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "panel.csv", index=False)
    paq = dict(f)
    paq.update({"bolsa": bolsa, "ridge": lr, "svr": ls, "hgb_ajustado": ha,
                "hgb_ajustado_C": hc, "apilado": ap,
                "codificador_ajustado_C": "models/e5_ajustado_C"})
    joblib.dump(paq, "models/estimador_pbe_v5.joblib", compress=3)
    json.dump({"pesos": dict(zip(canal, ap.coef_.tolist()))},
              open(OUT / "pesos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/panel.csv y models/estimador_pbe_v5.joblib · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
