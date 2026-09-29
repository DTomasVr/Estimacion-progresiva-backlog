"""E27 · Optimizar la TASA DE ACIERTO, que no es lo mismo que minimizar el error.

Toda la noche se optimizo error logaritmico. Pero PRED(25) y el "factor 2" no
son errores: son tasas de acierto dentro de una ventana. Y para esa perdida --
cero-uno con tolerancia -- el predictor optimo NO es la media ni la mediana.
Gneiting (2011, "Making and Evaluating Point Forecasts") lo deja escrito: bajo
la puntuacion cero-uno con tolerancia c, la regla de Bayes es **el punto medio
del intervalo modal de longitud 2c**. O sea: hay que colocar la prediccion
donde mas masa de probabilidad quepa en su ventana, no donde este el centro de
la distribucion.

Aqui eso importa el doble porque las horas estan MUY cuantizadas: los veinte
valores mas frecuentes cubren el 60,7 % de las tareas, y solo 1 h, 0,5 h y 2 h
ya son el 24,9 %. La distribucion condicional tiene atomos, y una prediccion de
1,83 h "desperdicia" la ventana comparada con una de 2,00 h.

Cuatro brazos, todos sobre el mismo modelo final:

  0  como esta                     el apilado, optimo en error logaritmico
  1  desplazamiento global         un solo factor, elegido en validacion para
                                   maximizar la tasa. La ventana de +-25 % es
                                   asimetrica en log ([-0,223, +0,288]), asi que
                                   quedarse corto ya ayuda
  2  ajuste a la rejilla           llevar la prediccion al valor frecuente mas
                                   cercano: aprovecha los atomos
  3  regla de Bayes                estimar P(y = v | x) con un clasificador
                                   sobre los valores frecuentes y elegir la p
                                   que maximiza la masa dentro de su ventana

Control obligatorio: el predictor CONSTANTE que maximiza la tasa sin mirar el
texto. Si un brazo no le gana, no esta aportando nada.

Se reporta a la vez el error, porque estos brazos lo empeoran a proposito y hay
que decirlo: es un cambio de objetivo, no una mejora gratis.
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
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
ART = Path("models/estimador_agix_v3.joblib")
OUT = Path("results/tasa_acierto")
SEP = "=" * 124
MIN_CUENTA = 60          # un valor entra en la rejilla con al menos estas apariciones


def tasas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    r = np.abs(p - y) / y
    f = np.maximum(p / y, y / p)
    m = E.point_metrics(y, p)
    return dict(PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(f <= 2)), factor3=float(np.mean(f <= 3)),
                MdMRE=m["MdMRE"],
                MdAElog=float(np.median(np.abs(np.log(p) - np.log(y)))),
                R2=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()))


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
    print(f"modelo cargado y aplicado · {time.time() - t0:.0f} s")

    # ------------------------------------------------ rejilla de valores
    cuentas = pd.Series(y[tr]).value_counts()
    rejilla = np.sort(cuentas[cuentas >= MIN_CUENTA].index.to_numpy())
    # rellenar huecos mayores de un factor 1,6 para que la rejilla cubra el rango
    extra = []
    for a, b in zip(rejilla[:-1], rejilla[1:]):
        k = int(np.floor(np.log(b / a) / np.log(1.6)))
        extra += list(np.exp(np.linspace(np.log(a), np.log(b), k + 2))[1:-1]) if k > 0 else []
    rejilla = np.sort(np.unique(np.concatenate([rejilla, extra,
                                                [y[tr].min(), y[tr].max()]])))
    cobertura = float(np.mean(np.isin(y[tr], cuentas[cuentas >= MIN_CUENTA].index)))
    print(f"rejilla: {len(rejilla)} valores de {rejilla[0]:.2f} a {rejilla[-1]:.0f} h · "
          f"los frecuentes cubren el {cobertura:.1%} de las tareas exactamente")

    # ------------------------------------------------ 1 · desplazamiento global
    def mejor_desp(mask, tol):
        ds = np.arange(-.8, .81, .01)
        hits = [np.mean(np.abs(np.exp(lp[mask] + d) - y[mask]) / y[mask] <= tol)
                for d in ds]
        return float(ds[int(np.argmax(hits))])
    d25 = mejor_desp(va, .25)
    d2 = float(np.arange(-.8, .81, .01)[int(np.argmax(
        [np.mean(np.maximum(np.exp(lp[va] + d) / y[va], y[va] / np.exp(lp[va] + d)) <= 2)
         for d in np.arange(-.8, .81, .01)]))])
    print(f"desplazamiento elegido en validación: ×{np.exp(d25):.3f} para ±25 % · "
          f"×{np.exp(d2):.3f} para factor 2")

    # ------------------------------------------------ 3 · regla de Bayes
    clases = np.abs(np.log(rejilla)[None, :] - ly[:, None]).argmin(1)
    print(f"clases en el entrenamiento: {len(np.unique(clases[tr]))}", flush=True)
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=.08,
                                         l2_regularization=1., random_state=0)
    clf.fit(Xd[tr], clases[tr])
    P = clf.predict_proba(Xd)
    orden = np.array(sorted(np.unique(clases[tr])))
    val_clase = rejilla[orden]
    print(f"clasificador ajustado · {time.time() - t0:.0f} s", flush=True)

    def bayes(tol_tipo):
        """Elige el valor de la rejilla que mete mas masa en su ventana."""
        if tol_tipo == "pred25":
            lo, hi = 0.8, 1 / 0.75
        else:
            lo, hi = 0.5, 2.0
        M = np.zeros((len(rejilla), len(val_clase)), bool)
        for i, c in enumerate(rejilla):
            M[i] = (val_clase >= c * lo) & (val_clase <= c * hi)
        masa = P @ M.T
        return rejilla[masa.argmax(1)]
    p_b25, p_b2 = bayes("pred25"), bayes("factor2")

    # ------------------------------------------------ 2 · ajuste a la rejilla
    def ajustar(v):
        return rejilla[np.abs(np.log(rejilla)[None, :] - np.log(v)[:, None]).argmin(1)]
    p_snap = ajustar(np.exp(lp))

    # ------------------------------------------------ control constante
    def constante(mask, tol_tipo):
        lo, hi = (0.8, 1 / 0.75) if tol_tipo == "pred25" else (0.5, 2.0)
        best, bv = rejilla[0], -1
        for c in rejilla:
            v = float(np.mean((y[mask] >= c * lo) & (y[mask] <= c * hi)))
            if v > bv:
                best, bv = c, v
        return best
    c25, c2 = constante(va, "pred25"), constante(va, "factor2")
    print(f"constante que más acierta en validación: {c25:.2f} h para ±25 % · "
          f"{c2:.2f} h para factor 2")

    brazos = {
        "0 · el modelo tal cual": np.exp(lp),
        f"1 · × {np.exp(d25):.3f} (desplazamiento para ±25 %)": np.exp(lp + d25),
        f"1b · × {np.exp(d2):.3f} (desplazamiento para factor 2)": np.exp(lp + d2),
        "2 · ajustado a la rejilla de valores": p_snap,
        "3 · regla de Bayes para ±25 %": p_b25,
        "3b · regla de Bayes para factor 2": p_b2,
        f"control · constante {c25:.2f} h": np.full(len(sw), c25),
        f"control · constante {c2:.2f} h": np.full(len(sw), c2),
    }
    filas = []
    for b, k in bl.items():
        for nombre, p in brazos.items():
            filas.append(dict(bloque=b, brazo=nombre, n=int(k.sum()), **tasas(y[k], p[k])))
    t = pd.DataFrame(filas)
    cols = ["brazo", "PRED25", "PRED50", "factor2", "factor3", "MdMRE", "MdAElog", "R2"]
    for b in bl:
        print(f"\n{SEP}\n{b}\n{SEP}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"rejilla": rejilla.tolist(), "desp_25": d25, "desp_f2": d2,
               "constante_25": float(c25), "constante_f2": float(c2),
               "min_cuenta": MIN_CUENTA},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    joblib.dump({"clasificador": clf, "rejilla": rejilla, "val_clase": val_clase},
                "models/bayes_tasa.joblib", compress=3)
    print(f"\nescrito {OUT}/resumen.csv y models/bayes_tasa.joblib")


if __name__ == "__main__":
    main()
