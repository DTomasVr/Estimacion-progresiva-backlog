"""Paso 4: clasificar el tramo en vez de (o ademas de) regresionar las horas.

El paso 3 dejo claro que el problema no es la anchura de la banda sino su
centro: para una tarea real de mas de 16 h el modelo predice 3,4 h, y ninguna
banda razonable alrededor de 3,4 llega a 24. La cobertura en ese tramo es 0,18
con la banda global y no la arregla ni la cuantilica ni la conformal por tramo.
Para cubrir las tareas grandes hay que PREDECIR que son grandes.

Ademas, para planificar, el tramo es lo que se necesita: saber si algo es de una
tarde o de dos semanas cambia la decision; saber si son 2,1 o 2,4 horas no.

  1  un clasificador de 4 tramos sobre los mismos rasgos, y si le gana al propio
     regresor en identificar las tareas grandes
  2  dos formas de devolver horas a partir de el:
       esperanza      suma_k P(k) x mediana de horas del tramo k
       dos etapas     el tramo mas probable, y dentro de el la mediana del
                      entrenamiento
  3  que pasa con la cobertura si la banda se centra en el tramo predicho
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
CORTES = [0, 1, 4, 16, np.inf]
ETQ = ["<1 h", "1-4 h", "4-16 h", ">16 h"]
SEP = "=" * 128


def tramo(v):
    return pd.cut(np.asarray(v, float), CORTES, labels=ETQ, right=False)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    lr = np.log(p / y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))), PRED25=m["PRED25"],
                MAE_h=m["MAE_h"], sesgo=float(np.exp(np.median(lr))),
                suma_pred_real=float(p.sum() / y.sum()))


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
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}

    y_tr = np.asarray(tramo(tr.effort_hours))
    print(f"entrenamiento {len(tr):,} · reparto de tramos: "
          + " · ".join(f"{e} {(y_tr == e).mean():.1%}" for e in ETQ))
    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)
    clf.fit(rasgos(tr), y_tr)
    clases = list(clf.classes_)
    mediana_tramo = {e: float(np.median(tr.effort_hours[y_tr == e])) for e in ETQ}
    media_geo = {e: float(np.exp(np.mean(np.log(tr.effort_hours[y_tr == e])))) for e in ETQ}
    print("  mediana de horas por tramo en el entrenamiento: "
          + " · ".join(f"{e} {v:.2f} h" for e, v in mediana_tramo.items()))

    filas, conf = [], {}
    for b, d in bloques.items():
        X = rasgos(d)
        y = d.effort_hours.to_numpy()
        real = np.asarray(tramo(y))
        P = clf.predict_proba(X)
        pred_clf = clf.predict(X)
        p_reg = np.exp(reg.predict(X))
        pred_reg = np.asarray(tramo(p_reg))

        print(f"\n{SEP}\n{b} · n={len(d):,}\n{SEP}")
        print("clasificador de tramo:")
        print(classification_report(real, pred_clf, labels=ETQ, zero_division=0, digits=3))
        print("  matriz de confusión (filas = real, columnas = predicho):")
        cm = confusion_matrix(real, pred_clf, labels=ETQ)
        print(pd.DataFrame(cm, index=ETQ, columns=ETQ).to_string())
        f1c = f1_score(real, pred_clf, labels=ETQ, average="macro", zero_division=0)
        f1r = f1_score(real, pred_reg, labels=ETQ, average="macro", zero_division=0)
        rec_c = float((pred_clf[real == ">16 h"] == ">16 h").mean()) if (real == ">16 h").any() else np.nan
        rec_r = float((pred_reg[real == ">16 h"] == ">16 h").mean()) if (real == ">16 h").any() else np.nan
        print(f"  F1 macro · clasificador {f1c:.3f} vs tramo implícito del regresor {f1r:.3f}")
        print(f"  cobertura de las tareas >16 h · clasificador {rec_c:.3f} vs regresor {rec_r:.3f}")
        conf[b] = dict(f1_clf=f1c, f1_reg=f1r, recall16_clf=rec_c, recall16_reg=rec_r)

        idx = {c: clases.index(c) for c in ETQ if c in clases}
        esperanza = sum(P[:, idx[e]] * media_geo[e] for e in idx)
        dos_etapas = np.array([mediana_tramo[c] for c in pred_clf])
        mezcla = np.sqrt(p_reg * esperanza)
        for nombre, p in (("regresor (lo actual)", p_reg),
                          ("esperanza sobre tramos", esperanza),
                          ("dos etapas: tramo -> mediana", dos_etapas),
                          ("media geométrica regresor × esperanza", mezcla)):
            filas.append(dict(bloque=b, metodo=nombre, n=len(d), **metricas(y, p)))

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nHORAS A PARTIR DEL CLASIFICADOR\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        print(t[t.bloque == b].drop(columns=["bloque"]).round(4).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/clasificador_tramo.csv", index=False)
    json.dump({"confusion": conf, "mediana_tramo": mediana_tramo,
               "media_geo_tramo": media_geo,
               "reparto_train": {e: float((y_tr == e).mean()) for e in ETQ}},
              open("results/clasificador_tramo.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    joblib.dump({"clasificador": clf, "onehot": oh, "etiquetas": ETQ,
                 "mediana_tramo": mediana_tramo, "media_geo_tramo": media_geo},
                "models/clasificador_tramo.joblib", compress=3)
    print("\nescrito results/clasificador_tramo.csv, .json y models/clasificador_tramo.joblib")


if __name__ == "__main__":
    main()
