"""E28 · Afinar la regla de Bayes, y contrastarla contra el predictor constante.

E27 dejo dos cosas. La buena: aplicar la regla de Bayes para la perdida cero-uno
sube el PRED(25) de 0,167 a 0,198 en la prueba fija. La incomoda: un predictor
CONSTANTE de 1,25 h saca 0,177, o sea que a esa tolerancia el modelo apenas le
gana a no mirar el texto.

Antes de dar eso por bueno hay un fallo evidente que arreglar: el clasificador
que estima P(y = v | x) solo veia los vectores, no la prediccion del modelo de
regresion, que es el resumen mas informativo que tenemos del texto. Metersela
como rasgo deberia afilar mucho la distribucion condicional.

  A  clasificador solo con vectores           (lo de E27)
  B  + la prediccion del apilado como rasgo
  C  B con la temperatura calibrada en validacion
  D  B restringiendo los candidatos a un factor 3 alrededor de la prediccion,
     para que la regla no se vaya a la moda global cuando dude

Y esta vez se contrasta contra el constante con una prueba pareada de McNemar
sobre los aciertos, que es lo que corresponde cuando lo que se compara son
dos secuencias de aciertos y fallos sobre las mismas tareas.
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
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
ART = Path("models/estimador_pbe_v3.joblib")
OUT = Path("results/tasa_acierto")
SEP = "=" * 122
MIN_CUENTA = 60


def tasas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    r = np.abs(p - y) / y
    f = np.maximum(p / y, y / p)
    m = E.point_metrics(y, p)
    return dict(PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(f <= 2)), MdMRE=m["MdMRE"],
                MdAElog=float(np.median(np.abs(np.log(p) - np.log(y)))),
                R2=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()))


def mcnemar(a, b):
    """a y b son vectores de aciertos. Devuelve el veredicto de a frente a b."""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return "empate", 1.0, 0
    p = binomtest(n01, n01 + n10, .5).pvalue
    ver = "mejor" if n01 > n10 else "peor" if n10 > n01 else "empate"
    return ver, float(p), n01 - n10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load(ART)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
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
    Ms = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Xd = np.hstack([E5] + [sv.transform(tf.transform(tit)).astype(np.float32)
                           for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Ms), f["svr"].predict(Ms),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    rej = np.array(json.load(open(OUT / "meta.json", encoding="utf-8"))["rejilla"])
    clases = np.abs(np.log(rej)[None, :] - ly[:, None]).argmin(1)
    orden = np.array(sorted(np.unique(clases[tr])))
    val_clase = rej[orden]
    print(f"rejilla {len(rej)} · clases {len(orden)} · {time.time() - t0:.0f} s")

    def entrenar(X):
        c = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=.08,
                                           l2_regularization=1., random_state=0)
        c.fit(X[tr], clases[tr])
        return c.predict_proba(X)

    PA = entrenar(Xd)
    print(f"  A ajustado · {time.time() - t0:.0f} s", flush=True)
    Xb = np.hstack([Xd, lp[:, None], Z])
    PB = entrenar(Xb)
    print(f"  B ajustado · {time.time() - t0:.0f} s", flush=True)

    def ventana(tipo):
        return (0.8, 1 / 0.75) if tipo == "pred25" else (0.5, 2.0)

    def decidir(P, tipo, temp=1.0, tope=None):
        lo, hi = ventana(tipo)
        Q = P ** (1 / temp)
        Q = Q / Q.sum(1, keepdims=True)
        M = np.zeros((len(rej), len(val_clase)), bool)
        for i, c in enumerate(rej):
            M[i] = (val_clase >= c * lo) & (val_clase <= c * hi)
        masa = Q @ M.T
        if tope is not None:
            fuera = np.abs(np.log(rej)[None, :] - lp[:, None]) > np.log(tope)
            masa = np.where(fuera, -1, masa)
        return rej[masa.argmax(1)]

    # temperatura y tope elegidos en validación
    mejor_t, mejor_v = 1.0, -1
    for temp in (0.5, .75, 1.0, 1.5, 2.0, 3.0):
        p = decidir(PB, "pred25", temp)
        v = float(np.mean(np.abs(p[va] - y[va]) / y[va] <= .25))
        if v > mejor_v:
            mejor_t, mejor_v = temp, v
    mejor_k, mejor_kv = None, -1
    for tope in (1.5, 2.0, 3.0, 5.0, None):
        p = decidir(PB, "pred25", mejor_t, tope)
        v = float(np.mean(np.abs(p[va] - y[va]) / y[va] <= .25))
        if v > mejor_kv:
            mejor_k, mejor_kv = tope, v
    print(f"validación: temperatura {mejor_t} · tope ×{mejor_k} "
          f"(PRED25 val {mejor_kv:.4f})")

    cte25 = float(json.load(open(OUT / "meta.json", encoding="utf-8"))["constante_25"])
    cte2 = float(json.load(open(OUT / "meta.json", encoding="utf-8"))["constante_f2"])
    brazos = {
        "0 · el modelo tal cual": np.exp(lp),
        "A · Bayes con vectores (E27)": decidir(PA, "pred25"),
        "B · Bayes + predicción del modelo": decidir(PB, "pred25"),
        f"C · B con temperatura {mejor_t}": decidir(PB, "pred25", mejor_t),
        f"D · C con tope ×{mejor_k}": decidir(PB, "pred25", mejor_t, mejor_k),
        "B2 · Bayes factor 2 + predicción": decidir(PB, "factor2"),
        f"control · constante {cte25:.2f} h": np.full(len(sw), cte25),
        f"control · constante {cte2:.2f} h": np.full(len(sw), cte2),
    }
    filas, contrastes = [], []
    for b, k in bl.items():
        ref25 = np.abs(brazos[f"control · constante {cte25:.2f} h"][k] - y[k]) / y[k] <= .25
        ref2 = np.maximum(brazos[f"control · constante {cte2:.2f} h"][k] / y[k],
                          y[k] / brazos[f"control · constante {cte2:.2f} h"][k]) <= 2
        for nombre, p in brazos.items():
            filas.append(dict(bloque=b, brazo=nombre, n=int(k.sum()), **tasas(y[k], p[k])))
            h25 = np.abs(p[k] - y[k]) / y[k] <= .25
            h2 = np.maximum(p[k] / y[k], y[k] / p[k]) <= 2
            v1, p1, d1 = mcnemar(h25, ref25)
            v2, p2, d2 = mcnemar(h2, ref2)
            contrastes.append(dict(bloque=b, brazo=nombre,
                                   pred25_vs_cte=f"{v1} (p={p1:.3g}, {d1:+d})",
                                   factor2_vs_cte=f"{v2} (p={p2:.3g}, {d2:+d})"))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contrastes)
    cols = ["brazo", "PRED25", "PRED50", "factor2", "MdMRE", "MdAElog", "R2"]
    for b in bl:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t.to_csv(OUT / "afinado.csv", index=False)
    c.to_csv(OUT / "contrastes.csv", index=False)
    json.dump({"temperatura": mejor_t, "tope": mejor_k},
              open(OUT / "afinado.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/afinado.csv y contrastes.csv")


if __name__ == "__main__":
    main()
