"""Paso 3: bandas condicionales en vez de una banda global.

La banda actual es un factor unico, x4,71, calibrado sobre todos los residuos de
validacion. Eso garantiza cobertura MEDIA del 80 %, no cobertura para cada tipo
de tarea: si el error crece con el tamano -- y aqui crece -- la banda sobra en
las tareas cortas y falta en las largas, y la cobertura por tramo se separa del
80 % en las dos direcciones. Es exactamente lo que la literatura de pronostico
probabilistico llama falta de calibracion condicional.

Tres brazos sobre los mismos bloques congelados:

  A  global          el factor unico de hoy
  B  cuantilica      tres HGB con perdida pinball en 0,10 / 0,50 / 0,90; el
                     intervalo es [q10, q90] directamente, sin factor
  C  conformal por   el factor se calibra por separado dentro de cada tramo de
     tramo           prediccion (Mondrian). El tramo se decide con la
                     prediccion, no con el valor real, porque en produccion el
                     real no se conoce

Se mide cobertura global, cobertura POR TRAMO REAL -- que es donde se ve el
defecto -- y anchura, porque una banda que cubre por ser enorme no sirve.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
TRAMOS = [(0, 1, "<1 h"), (1, 4, "1-4 h"), (4, 16, "4-16 h"), (16, 1e9, ">16 h")]
SEP = "=" * 128


def tramo(v):
    return pd.cut(v, [0, 1, 4, 16, np.inf], labels=[t[2] for t in TRAMOS], right=False)


def main():
    a = joblib.load(ART)
    reg, oh = a["regresor"], a["onehot"]
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]

    def rasgos(d):
        X = emb[d.fe.to_numpy()]
        if oh is not None:
            X = np.hstack([X, oh.transform(d[["task_kind"]].astype(str))])
        return X

    tr = sw[(sw.split_t3 == "train") & (~sw.apache)]
    va = sw[(sw.split_t3 == "val") & (~sw.apache)]
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    print(f"entrenamiento {len(tr):,} · calibración {len(va):,} · "
          + " · ".join(f"{k} {len(v):,}" for k, v in bloques.items()))

    # ------------------------------------------------ A · banda global
    lp_va = reg.predict(rasgos(va))
    res_va = np.abs(np.log(va.effort_hours.to_numpy()) - lp_va)
    q_global = float(np.quantile(res_va, .8))
    print(f"\nA · banda global: factor ×{np.exp(q_global):.2f}")

    # ------------------------------------------------ B · cuantilica
    cuant = {}
    for q in (.1, .5, .9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        m.fit(rasgos(tr), np.log(tr.effort_hours.to_numpy()))
        cuant[q] = m
        print(f"B · cuantílica q={q} ajustada")

    # ------------------------------------------------ C · conformal por tramo
    pred_va = np.exp(lp_va)
    tr_va = tramo(pred_va)
    q_tramo = {}
    for _, _, etq in TRAMOS:
        k = np.asarray(tr_va == etq)
        q_tramo[etq] = float(np.quantile(res_va[k], .8)) if k.sum() >= 50 else q_global
    print("C · conformal por tramo de predicción: "
          + " · ".join(f"{e} ×{np.exp(v):.2f}" for e, v in q_tramo.items()))

    # ------------------------------------------------ lectura
    filas = []
    for b, d in bloques.items():
        y = d.effort_hours.to_numpy()
        X = rasgos(d)
        lp = reg.predict(X)
        p = np.exp(lp)
        bandas = {
            "A · global": (np.exp(lp - q_global), np.exp(lp + q_global)),
            "B · cuantílica": (np.exp(cuant[.1].predict(X)), np.exp(cuant[.9].predict(X))),
            "C · conformal por tramo": (
                np.exp(lp - np.array([q_tramo[t] for t in tramo(p)])),
                np.exp(lp + np.array([q_tramo[t] for t in tramo(p)]))),
        }
        for nombre, (lo, hi) in bandas.items():
            hi = np.maximum(hi, lo * 1.0001)
            dentro = (y >= lo) & (y <= hi)
            fila = dict(bloque=b, banda=nombre, n=len(d), cobertura=float(dentro.mean()),
                        ancho_mediano_h=float(np.median(hi - lo)),
                        factor_mediano=float(np.median(hi / np.maximum(lo, 1e-9)) ** .5))
            for lo_t, hi_t, etq in TRAMOS:
                k = (y >= lo_t) & (y < hi_t)
                fila[f"cob_{etq}"] = float(dentro[k].mean()) if k.sum() else np.nan
            filas.append(fila)
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCOBERTURA (nominal 0,80) Y ANCHURA\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        print(t[t.bloque == b].drop(columns=["bloque"]).round(3).to_string(index=False))

    print(f"\n{SEP}\nLO QUE SE VE\n{SEP}")
    g = t[t.bloque == "prueba fija"].set_index("banda")
    cols = [f"cob_{e}" for _, _, e in TRAMOS]
    for nombre in g.index:
        v = g.loc[nombre, cols].astype(float)
        print(f"  {nombre:26s} cobertura por tramo {v.min():.3f}–{v.max():.3f} "
              f"(rango {v.max() - v.min():.3f}) · global {g.loc[nombre, 'cobertura']:.3f}")

    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/bandas_condicionales.csv", index=False)
    json.dump({"q_global": q_global, "factor_global": float(np.exp(q_global)),
               "q_por_tramo": q_tramo}, open("results/bandas.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    joblib.dump({q: m for q, m in cuant.items()}, "models/cuantiles_titulos.joblib",
                compress=3)
    print("\nescrito results/bandas_condicionales.csv, bandas.json y "
          "models/cuantiles_titulos.joblib")


if __name__ == "__main__":
    main()
