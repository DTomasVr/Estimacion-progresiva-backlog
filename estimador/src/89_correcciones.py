"""Paso 1 y 2: corregir la salida sin tocar el modelo.

1 · SMEARING DE DUAN (1983). El modelo ajusta log(horas) y se exponencia, lo que
    devuelve la MEDIANA condicional, no la media. Para cualquier cosa que se
    sume -- un backlog, un plan, un presupuesto -- eso subestima de forma
    sistematica. El estimador de smearing corrige multiplicando por la media de
    exp(residuos) del entrenamiento. No es una mejora del modelo: es la
    diferencia entre responder "la tarea tipica" y responder "cuanto suma esto".

2 · FACTOR DE ESTILO. El texto del generador infla las predicciones: medido en
    87, el sesgo pasa de 1,02 con titulos de Jira a 1,20 con el mismo contenido
    reescrito al estilo del generador. El factor se ajusta en un bloque y se
    verifica en el otro.

Se mide lo que cada correccion arregla y lo que rompe, porque no mueven las
mismas metricas: el smearing empeora el MdMRE (sube todas las predicciones) y
arregla la suma. Hay que decir las dos cosas.
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
pd.set_option("display.width", 220)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
REESCRITO = "results/reescritura/comparacion.csv"
GEN = Path("F:/Software Task generation Model/salida_gen.json")
SEP = "=" * 126


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

    # ---------------------------------------------- 1 · smearing, ajustado en entrenamiento
    tr = sw[(sw.split_t3 == "train") & (~sw.apache)]
    res = np.log(tr.effort_hours.to_numpy()) - reg.predict(rasgos(tr))
    S = float(np.mean(np.exp(res)))
    print(f"factor de smearing de Duan ajustado en el entrenamiento: S = {S:.4f}")
    print(f"  (residuos: sd {res.std():.3f} · si fueran normales seria exp(sd²/2) = "
          f"{np.exp(res.var() / 2):.4f})")

    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "prueba TAWOS": sw[(sw.split_t3 == "test") & (sw.source == "TAWOS")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    filas = []
    for b, d in bloques.items():
        y = d.effort_hours.to_numpy()
        p = np.exp(reg.predict(rasgos(d)))
        filas.append(dict(bloque=b, n=len(d), version="sin corregir", **metricas(y, p)))
        filas.append(dict(bloque=b, n=len(d), version="con smearing", **metricas(y, p * S)))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\n1 · SMEARING: QUE ARREGLA Y QUE ROMPE\n{SEP}")
    print(t.round(4).to_string(index=False))
    print("\n  el smearing sube todas las predicciones por igual: empeora el MdMRE y")
    print("  acerca la SUMA, que es lo que importa cuando se planifica un backlog entero.")

    # ---------------------------------------------- 2 · factor de estilo, cruzado
    r = pd.read_csv(REESCRITO)
    print(f"\n{SEP}\n2 · FACTOR DE ESTILO: AJUSTADO EN UN BLOQUE, VERIFICADO EN EL OTRO\n{SEP}")
    factores, filas2 = {}, []
    for ajuste in r.bloque.unique():
        ca = r[r.bloque == ajuste]
        k = float(np.exp(np.median(np.log(ca.effort_hours / ca.pred_reescrito))))
        factores[ajuste] = k
        ap = r[r.bloque != ajuste]
        y = ap.effort_hours.to_numpy()
        for etq, p in (("original (estilo Jira)", ap.pred_original.to_numpy()),
                       ("reescrito sin corregir", ap.pred_reescrito.to_numpy()),
                       (f"reescrito × {k:.3f}", ap.pred_reescrito.to_numpy() * k)):
            filas2.append(dict(ajustado_en=ajuste, evaluado_en=ap.bloque.iloc[0],
                               n=len(ap), version=etq, **metricas(y, p)))
    t2 = pd.DataFrame(filas2)
    print(t2.round(4).to_string(index=False))
    k_medio = float(np.exp(np.mean(np.log(list(factores.values())))))
    print(f"\n  factores: " + " · ".join(f"{a} ×{b:.3f}" for a, b in factores.items())
          + f"  ->  media geométrica ×{k_medio:.3f}")

    # ---------------------------------------------- 3 · efecto sobre el backlog generado
    print(f"\n{SEP}\n3 · EFECTO SOBRE EL BACKLOG QUE PRODUCE EL GENERADOR\n{SEP}")
    if GEN.exists():
        import torch
        from sentence_transformers import SentenceTransformer

        gen = json.load(open(GEN, encoding="utf-8"))
        tex = [en[i] for d in gen.values() for h in d["historias"]
               for i, en in [(i, h.get("tareas") or []) for i in
                             range(len(h.get("tareas_es") or []))]
               if i < len(en)]
        st = SentenceTransformer(a["codificador"],
                                 device="cuda" if torch.cuda.is_available() else "cpu")
        X = st.encode(tex, normalize_embeddings=False, show_progress_bar=False)
        if oh is not None:
            X = np.hstack([X, oh.transform(pd.DataFrame(
                {"task_kind": ["task"] * len(tex)}).astype(str))])
        p = np.exp(reg.predict(X))
        for etq, q in (("sin corregir", p), ("× estilo", p * k_medio),
                       ("× smearing", p * S), ("× estilo × smearing", p * k_medio * S)):
            print(f"  {etq:22s} mediana {np.median(q):5.2f} h · "
                  f"backlog {q.sum():6.0f} h ({q.sum() / 8:5.1f} días-persona)")
        print(f"\n  n = {len(tex)} tareas generadas")

    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/correccion_smearing.csv", index=False)
    t2.to_csv("results/correccion_estilo.csv", index=False)
    json.dump({"smearing": S, "factores_estilo": factores, "estilo_medio": k_medio},
              open("results/correcciones.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nescrito results/correccion_smearing.csv, correccion_estilo.csv, correcciones.json")


if __name__ == "__main__":
    main()
