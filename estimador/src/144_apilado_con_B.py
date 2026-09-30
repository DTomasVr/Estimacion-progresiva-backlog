"""Si el codificador nuevo aporta o no, se decide DENTRO del apilado.

Por su cuenta, el codificador con cabeza de clasificacion (brazo B de E35) es un
empate a medias contra el escalar que ya teniamos: mejor R2 en la prueba fija
(0,118 frente a 0,091), peor Spearman (0,386 frente a 0,393) y bastante peor en
proyectos nuevos.

Pero ese canal no se usa suelto. E10 dejo escrito por que esta en el apilado: NO
porque sea el mejor, sino porque ordena mejor y estima peor la escala, y esa
diferencia de perfil es justo lo que aporta a un conjunto donde la bolsa de
arboles ya estima bien la escala. Un codificador con el perfil contrario -- mejor
R2, peor orden -- podria ser mejor en solitario y aportar menos apilado.

Cuatro apilados, identicos salvo por que codificador ajustado llevan:

  0  el de siempre, con A
  1  con B en lugar de A
  2  con los dos
  3  sin ninguno, para ver cuanto aporta el canal en total
"""
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
OUT = Path("results/finetune_objetivos")
S = "=" * 122


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
    A = np.load(CACHE / "e5-ajustado.npy")
    B = np.load(CACHE / "e5-ajustado-B.npy")
    C = np.load(CACHE / "e5-ajustado-C.npy")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    actual = f["apilado"].predict(np.column_stack([
        np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
        f["ridge"].predict(Xs), f["svr"].predict(Xs), f["hgb_ajustado"].predict(A)]))

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    bolsa = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
    canal = {"bolsa": np.mean([m.predict(Xd) for m in bolsa], 0),
             "ridge disperso": Ridge(alpha=2.).fit(Xs[tr], ly[tr]).predict(Xs),
             "SVR disperso": LinearSVR(C=.3, max_iter=4000, random_state=0).fit(
                 Xs[tr], ly[tr]).predict(Xs),
             "A escalar": hgb().fit(A[tr], ly[tr]).predict(A),
             "B clasificacion": hgb().fit(B[tr], ly[tr]).predict(B),
             "C clasificacion + orden": hgb().fit(C[tr], ly[tr]).predict(C)}
    print(f"canales listos · {time.time() - t0:.0f} s", flush=True)

    L = ["bolsa", "ridge disperso", "SVR disperso"]
    comb = {"0 · el de siempre (con A)": L + ["A escalar"],
            "1 · con B en lugar de A": L + ["B clasificacion"],
            "2 · con C en lugar de A": L + ["C clasificacion + orden"],
            "3 · con A y C": L + ["A escalar", "C clasificacion + orden"],
            "4 · con los tres": L + ["A escalar", "B clasificacion",
                                     "C clasificacion + orden"],
            "5 · sin codificador ajustado": L}
    pred = {"referencia · el artefacto v3": actual}
    for n, ks in comb.items():
        Z = np.column_stack([canal[k] for k in ks])
        ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va], ly[va])
        pred[n] = ap.predict(Z)
        print(f"  {n}: " + " · ".join(f"{k} {w:+.3f}" for k, w in zip(ks, ap.coef_)))

    filas, contr = [], []
    for b, k in bl.items():
        r25 = np.abs(np.exp(actual[k]) - y[k]) / y[k] <= .25
        for n, lpz in pred.items():
            p = np.exp(lpz)
            filas.append(dict(bloque=b, apilado=n, n=int(k.sum()), **metricas(y[k], p[k])))
            v, pv, d = mcnemar(np.abs(p[k] - y[k]) / y[k] <= .25, r25)
            contr.append(dict(bloque=b, apilado=n, pred25_vs_v3=f"{v} (p={pv:.3g}, {d:+d})"))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contr)
    cols = ["apilado", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t.to_csv(OUT / "apilado_con_B.csv", index=False)
    print(f"\nescrito {OUT}/apilado_con_B.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
