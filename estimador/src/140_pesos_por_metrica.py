"""E40 - Elegir los pesos del apilado para la metrica que se reporta, no para el error cuadratico.

E39 deja un dato incomodo. En la prueba fija el apilado saca PRED25 0,167, y
estos canales sueltos, que forman parte de el, sacan mas:

    SVR disperso      0,1815
    e5 ajustado       0,1804
    analogia k=15     0,1775

O sea que el apilado esta EMPEORANDO la tasa de acierto de sus propios
componentes. No es una paradoja: los pesos se aprenden minimizando error
cuadratico en validacion, que es lo optimo para el R2 y no tiene por que serlo
para una tasa de acierto dentro de una ventana. Promediar acerca a la media
condicional, y acercarse a la media aleja del valor mas probable cuando la
distribucion es asimetrica.

Asi que se eligen los pesos para cada metrica, por busqueda sobre el simplice en
validacion. El riesgo evidente es sobreajustar la validacion: once pesos contra
3.183 filas y un criterio de cero-uno es facil de exprimir. Por eso se reporta
SIEMPRE la cifra de validacion al lado de la de prueba. Si la de validacion sube
mucho y la de prueba no la sigue, el brazo no vale y hay que decirlo.

Se incluye tambien el brazo de "quedarse con el mejor canal segun validacion",
que es la version honesta de la observacion de arriba: elegir el SVR disperso
mirando validacion es legitimo; elegirlo mirando la prueba no lo es.
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
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/pesos_metrica")
S = "=" * 128
N_SORTEOS = 4000


def tasa(y, p, cual):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    if cual == "PRED25":
        return float(np.mean(np.abs(p - y) / y <= .25))
    if cual == "factor2":
        return float(np.mean(np.maximum(p / y, y / p) <= 2))
    if cual == "MdAElog":
        return -float(np.median(np.abs(np.log(p / y))))
    return -float(np.mean((np.log(p) - np.log(y)) ** 2))


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
    N = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    A, yt, out = N[tr], ly[tr], np.zeros(len(V))
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
    td = sw.description.fillna("").astype(str).str.split().str.len().gt(0).to_numpy()
    DE = np.where(td[:, None], DE, 0).astype(np.float32)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    actual = f["apilado"].predict(np.column_stack([
        np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
        f["ridge"].predict(Xs), f["svr"].predict(Xs), f["hgb_ajustado"].predict(AJ)]))

    base = {}
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tr)
    bolsa = []
    for s in range(7):
        sub = rng.choice(idx, int(.85 * len(idx)), replace=False)
        bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
    base["bolsa"] = np.mean([m.predict(Xd) for m in bolsa], 0)
    base["ridge disperso"] = Ridge(alpha=2.).fit(Xs[tr], ly[tr]).predict(Xs)
    base["SVR disperso"] = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(
        Xs[tr], ly[tr]).predict(Xs)
    base["e5 ajustado"] = hgb().fit(AJ[tr], ly[tr]).predict(AJ)
    for n in ("bge-m3", "qwen3-0.6B", "e5-base", "mpnet"):
        V = np.load(CACHE / f"{n}.npy")
        base[n] = hgb().fit(V[tr], ly[tr]).predict(V)
    base["descripcion"] = hgb().fit(DE[tr], ly[tr]).predict(DE)
    base["ridge denso"] = Ridge(alpha=10.).fit(E5[tr], ly[tr]).predict(E5)
    base["analogia k=15"] = analogia(E5, tr, ly)
    Z = np.column_stack(list(base.values()))
    nombres = list(base)
    print(f"{Z.shape[1]} canales · {time.time() - t0:.0f} s", flush=True)

    def buscar(cual):
        """Sorteo sobre el simplice mas un refinado por coordenadas."""
        r = np.random.default_rng(7)
        mejor_w = np.ones(Z.shape[1]) / Z.shape[1]
        mejor = tasa(y[va], np.exp(Z[va] @ mejor_w), cual)
        for _ in range(N_SORTEOS):
            w = r.dirichlet(np.ones(Z.shape[1]) * .5)
            v = tasa(y[va], np.exp(Z[va] @ w), cual)
            if v > mejor:
                mejor, mejor_w = v, w
        for _ in range(3):
            for j in range(Z.shape[1]):
                for d in (-.15, -.05, .05, .15):
                    w = mejor_w.copy()
                    w[j] = max(0, w[j] + d)
                    if w.sum() == 0:
                        continue
                    w = w / w.sum()
                    v = tasa(y[va], np.exp(Z[va] @ w), cual)
                    if v > mejor:
                        mejor, mejor_w = v, w
        return mejor_w

    ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Z[va], ly[va])
    brazos = {"0 · el modelo actual": actual,
              "1 · apilado por error cuadratico": ap.predict(Z)}
    for cual in ("PRED25", "factor2", "MdAElog"):
        w = buscar(cual)
        brazos[f"2 · pesos para {cual}"] = Z @ w
        top = sorted(zip(nombres, w), key=lambda a: -a[1])[:4]
        print(f"  pesos para {cual}: " + " · ".join(f"{a} {b:.2f}" for a, b in top),
              flush=True)
    for cual in ("PRED25", "factor2"):
        j = int(np.argmax([tasa(y[va], np.exp(Z[va, i]), cual) for i in range(Z.shape[1])]))
        brazos[f"3 · mejor canal suelto para {cual} ({nombres[j]})"] = Z[:, j]

    filas, contr = [], []
    for n, lpz in brazos.items():
        p = np.exp(lpz)
        filas.append(dict(bloque="validacion (donde se eligio)", modelo=n,
                          n=int(va.sum()), **metricas(y[va], p[va])))
    for b, k in bl.items():
        b25 = np.abs(np.exp(actual[k]) - y[k]) / y[k] <= .25
        bf2 = np.maximum(np.exp(actual[k]) / y[k], y[k] / np.exp(actual[k])) <= 2
        for n, lpz in brazos.items():
            p = np.exp(lpz)
            filas.append(dict(bloque=b, modelo=n, n=int(k.sum()), **metricas(y[k], p[k])))
            v1, p1, d1 = mcnemar(np.abs(p[k] - y[k]) / y[k] <= .25, b25)
            v2, p2, d2 = mcnemar(np.maximum(p[k] / y[k], y[k] / p[k]) <= 2, bf2)
            contr.append(dict(bloque=b, modelo=n,
                              pred25_vs_actual=f"{v1} (p={p1:.3g}, {d1:+d})",
                              factor2_vs_actual=f"{v2} (p={p2:.3g}, {d2:+d})"))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contr)
    cols = ["modelo", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in ["validacion (donde se eligio)"] + list(bl):
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        if b in bl:
            print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "contrastes.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
