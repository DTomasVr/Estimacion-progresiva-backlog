"""E39 - El apilado con TODAS las vistas que hay en disco, no con las tres de siempre.

El modelo final apila cuatro canales: la bolsa sobre e5-large, dos lineales
dispersos y el codificador ajustado. Pero en la cache hay seis codificadores y en
el proyecto hay dos vistas mas que nunca han entrado juntas:

  bge-m3, qwen3-0.6B, e5-base, mpnet   codificados en E4 y luego descartados
                                       por perder de uno en uno
  la descripcion                       E31 la deja en +0,015 de R2 donde existe
  la analogia (k vecinos)              en E22 saco el MEJOR PRED25 individual de
                                       todos los modelos (0,178 frente a 0,156 de
                                       la bolsa) con un R2 de 0,038

Esa ultima es la que mas interesa. Un modelo con R2 0,038 y PRED25 0,178 esta
acertando por un camino distinto al de la bolsa, y eso es exactamente lo que un
apilado puede aprovechar: en un conjunto no gana el que mas sabe, gana el que
sabe algo que los demas no. Descartar vistas por su cifra en solitario es el
error tipico, y aqui se ha cometido cuatro veces.

Se ajustan once modelos base sobre el entrenamiento, se aprenden los pesos en
validacion y se contrasta contra el modelo actual con McNemar sobre los aciertos.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/apilado_total")
S = "=" * 126


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def mcnemar(a, b):
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return "empate", 1.0, 0
    return ("mejor" if n01 > n10 else "peor" if n10 > n01 else "empate"), \
        float(binomtest(n01, n01 + n10, .5).pvalue), n01 - n10


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


def analogia(V, tr, ly, k=15):
    """Estimacion por analogia: la mediana de los k titulos mas parecidos del
    entrenamiento. Es la familia clasica de este problema y aqui aporta por una
    via distinta a la de los arboles."""
    N = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    A = N[tr]
    out = np.zeros(len(V))
    yt = ly[tr]
    for i in range(0, len(V), 2048):
        s = N[i:i + 2048] @ A.T
        j = np.argpartition(-s, k, axis=1)[:, :k]
        out[i:i + 2048] = np.median(yt[j], axis=1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
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
    AJ = np.load(CACHE / "e5-ajustado.npy")
    DE = np.load(CACHE / "e5-large-desc.npy")
    tiene_d = sw.description.fillna("").astype(str).str.split().str.len().gt(0).to_numpy()
    DE = np.where(tiene_d[:, None], DE, 0).astype(np.float32)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    actual = f["apilado"].predict(np.column_stack([
        np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
        f["ridge"].predict(Xs), f["svr"].predict(Xs), f["hgb_ajustado"].predict(AJ)]))
    print(f"datos listos · {time.time() - t0:.0f} s", flush=True)

    base = {}
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    bolsa = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
    base["bolsa sobre e5-large + TF-IDF"] = np.mean([m.predict(Xd) for m in bolsa], 0)
    print(f"  bolsa · {time.time() - t0:.0f} s", flush=True)
    base["ridge disperso"] = Ridge(alpha=2.).fit(Xs[tr], ly[tr]).predict(Xs)
    base["SVR disperso"] = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(
        Xs[tr], ly[tr]).predict(Xs)
    base["e5 ajustado"] = hgb().fit(AJ[tr], ly[tr]).predict(AJ)
    for n in ("bge-m3", "qwen3-0.6B", "e5-base", "mpnet"):
        V = np.load(CACHE / f"{n}.npy")
        base[f"HGB sobre {n}"] = hgb().fit(V[tr], ly[tr]).predict(V)
        print(f"  {n} · {time.time() - t0:.0f} s", flush=True)
    base["HGB sobre la descripcion"] = hgb().fit(DE[tr], ly[tr]).predict(DE)
    base["ridge denso sobre e5-large"] = Ridge(alpha=10.).fit(E5[tr], ly[tr]).predict(E5)
    base["analogia k=15"] = analogia(E5, tr, ly)
    print(f"  once canales listos · {time.time() - t0:.0f} s", flush=True)

    Z = np.column_stack(list(base.values()))
    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va], ly[va])
    print("\n  pesos: " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(base, ap.coef_)))
    total = {"0 · el modelo actual": actual,
             "1 · apilado de once canales": ap.predict(Z),
             "2 · media simple de los once": Z.mean(1)}
    # apilado solo con los cuatro de siempre, para aislar el efecto de anadir vistas
    cu = ["bolsa sobre e5-large + TF-IDF", "ridge disperso", "SVR disperso", "e5 ajustado"]
    Zc = np.column_stack([base[k] for k in cu])
    total["3 · apilado de los cuatro de siempre"] = RidgeCV(
        alphas=np.logspace(-2, 3, 20)).fit(Zc[va], ly[va]).predict(Zc)

    filas, contr = [], []
    for b, k in bl.items():
        b25 = np.abs(np.exp(actual[k]) - y[k]) / y[k] <= .25
        bf2 = np.maximum(np.exp(actual[k]) / y[k], y[k] / np.exp(actual[k])) <= 2
        for n, lpz in total.items():
            p = np.exp(lpz)
            filas.append(dict(bloque=b, modelo=n, n=int(k.sum()), **metricas(y[k], p[k])))
            v1, p1, d1 = mcnemar(np.abs(p[k] - y[k]) / y[k] <= .25, b25)
            v2, p2, d2 = mcnemar(np.maximum(p[k] / y[k], y[k] / p[k]) <= 2, bf2)
            contr.append(dict(bloque=b, modelo=n,
                              pred25_vs_actual=f"{v1} (p={p1:.3g}, {d1:+d})",
                              factor2_vs_actual=f"{v2} (p={p2:.3g}, {d2:+d})"))
        for n, v in base.items():
            filas.append(dict(bloque=b, modelo=f"    canal · {n}", n=int(k.sum()),
                              **metricas(y[k], np.exp(v[k]))))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contr)
    cols = ["modelo", "MAE_h", "MdAElog", "zeta_pc", "SSPB_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "contrastes.csv", index=False)
    json.dump({"pesos": dict(zip(base, ap.coef_.tolist()))},
              open(OUT / "pesos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
