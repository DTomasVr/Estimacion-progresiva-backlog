"""Paso 5: abstenerse. Si el modelo no puede acertar siempre, que al menos sepa cuando falla.

Los pasos 3 y 4 dejaron una conclusion incomoda: el tamano de la tarea no esta
en el titulo. Ni una banda mas lista ni un clasificador dedicado encuentran las
tareas grandes. Queda una pregunta distinta y mas util para el producto: el
modelo, aunque falle, ¿sabe DONDE va a fallar? Si lo sabe, el artefacto puede
callarse ahi y pedirle la estimacion al usuario, que es la palanca que ya
medimos como la mayor.

Es regresion selectiva: se ordena por confianza, se responde el x % mas seguro y
se mide el error SOLO sobre lo respondido. La curva riesgo-cobertura tiene que
bajar mas rapido que abstenerse al azar; si no baja, la senal de confianza no
vale nada por muy razonable que suene.

Cuatro senales, todas disponibles en el momento de predecir:
  anchura      q90 - q10 en log, de los modelos cuantilicos (paso 3)
  confianza    probabilidad maxima del clasificador de tramo (paso 4)
  cercania     coseno al vecino mas cercano del entrenamiento
  prediccion   el propio valor predicho, por si el modelo solo es fiable en su
               tramo comodo de 1 a 4 h
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
COBERTURAS = [1.0, .9, .8, .7, .6, .5, .4, .3, .2]
SEP = "=" * 128


def main():
    a = joblib.load(ART)
    reg, oh = a["regresor"], a["onehot"]
    cuant = joblib.load("models/cuantiles_titulos.joblib")
    cl = joblib.load("models/clasificador_tramo.joblib")
    clf = cl["clasificador"]

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
    Xtr = emb[tr.fe.to_numpy()]
    Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}

    filas, curvas = [], []
    for b, d in bloques.items():
        X = rasgos(d)
        y = d.effort_hours.to_numpy()
        lp = reg.predict(X)
        p = np.exp(lp)
        err = np.abs(np.log(np.clip(p, .01, None) / y))          # error logarítmico
        V = emb[d.fe.to_numpy()]
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        cercania = (V @ Xtr.T).max(1)
        anchura = cuant[.9].predict(X) - cuant[.1].predict(X)
        confianza = clf.predict_proba(X).max(1)
        comodo = -np.abs(lp - np.log(2.0))                        # cerca de 2 h

        senales = {"anchura de la banda": -anchura, "confianza del clasificador": confianza,
                   "cercanía al entrenamiento": cercania, "predicción en zona cómoda": comodo}
        rng = np.random.default_rng(0)
        senales["azar (referencia)"] = rng.random(len(y))

        for nombre, s in senales.items():
            orden = np.argsort(-s)                                # más confiado primero
            for c in COBERTURAS:
                k = orden[:max(10, int(round(c * len(y))))]
                m = E.point_metrics(y[k], np.clip(p[k], .01, None))
                curvas.append(dict(bloque=b, senal=nombre, cobertura=c, n=len(k),
                                   MdAElog=float(np.median(err[k])), MdMRE=m["MdMRE"],
                                   PRED25=m["PRED25"],
                                   frac_grandes=float((y[k] > 16).mean())))
            # area bajo la curva riesgo-cobertura: menos es mejor
            v = [x["MdAElog"] for x in curvas if x["bloque"] == b and x["senal"] == nombre]
            filas.append(dict(bloque=b, senal=nombre, AURC=float(np.mean(v)),
                              MdAElog_100=v[0], MdAElog_50=v[COBERTURAS.index(.5)],
                              MdAElog_20=v[-1]))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(curvas)

    print(f"{SEP}\nCURVA RIESGO-COBERTURA · error logarítmico mediano sobre lo RESPONDIDO\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        piv = c[c.bloque == b].pivot_table(index="senal", columns="cobertura",
                                           values="MdAElog")
        print(piv[sorted(piv.columns, reverse=True)].round(4).to_string())
    print(f"\n{SEP}\nRESUMEN: ¿ALGUNA SEÑAL LE GANA AL AZAR?\n{SEP}")
    for b in bloques:
        v = t[t.bloque == b].sort_values("AURC")
        azar = float(v[v.senal == "azar (referencia)"].AURC.iloc[0])
        v = v.assign(vs_azar=(v.AURC - azar).round(4))
        print(f"\n{b} (azar = {azar:.4f})")
        print(v.drop(columns=["bloque"]).round(4).to_string(index=False))

    print(f"\n{SEP}\nQUE PASA CON LAS TAREAS GRANDES AL ABSTENERSE\n{SEP}")
    for b in bloques:
        v = c[(c.bloque == b) & (c.cobertura.isin([1.0, .5]))]
        piv = v.pivot_table(index="senal", columns="cobertura", values="frac_grandes")
        print(f"\n{b} · fracción de tareas >16 h entre las respondidas")
        print(piv[sorted(piv.columns, reverse=True)].round(4).to_string())

    Path("results").mkdir(exist_ok=True)
    c.to_csv("results/abstencion_curvas.csv", index=False)
    t.to_csv("results/abstencion_resumen.csv", index=False)
    json.dump({"coberturas": COBERTURAS}, open("results/abstencion.json", "w",
                                               encoding="utf-8"), indent=1)
    print("\nescrito results/abstencion_curvas.csv y abstencion_resumen.csv")


if __name__ == "__main__":
    main()
