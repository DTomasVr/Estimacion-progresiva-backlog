"""Where do the generated tasks land, compared with the corpus the model was trained on?

The task-generation module writes short Spanish imperatives ("Almacenar los datos
de pago en el perfil del miembro"). The effort corpus is 74 % Apache: long
English issue reports with stack traces. This script measures how far apart they
are, and what the deployed model actually answers when handed the real thing.

  1. how close a generated task gets to its nearest training task (cosine),
     against the same statistic for real held-out tasks -- the off-manifold test
  2. what the model predicts for them, and what their neighbours really took
  3. Spanish against English wording of the SAME task, to separate the language
     gap from the domain gap
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
GEN = Path("F:/Software Task generation Model/salida_gen.json")
DATA = "data/processed/pbe_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]
SEP = "=" * 118


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    gen = json.load(open(GEN, encoding="utf-8"))
    tareas = []
    for dominio, d in gen.items():
        for h in d["historias"]:
            es, en = h.get("tareas_es") or [], h.get("tareas") or []
            for i, t in enumerate(es):
                tareas.append({"dominio": dominio, "es": t,
                               "en": en[i] if i < len(en) else None})
    g = pd.DataFrame(tareas).dropna(subset=["en"]).reset_index(drop=True)
    print(f"tareas generadas: {len(g)} de {g.dominio.nunique()} dominios")
    print(f"  palabras por tarea: mediana {g.es.str.split().str.len().median():.0f} "
          f"(es) / {g.en.str.split().str.len().median():.0f} (en)")

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    tr = sw[sw.split_t3 == "train"]
    Xtr = emb[tr.fe.to_numpy()]
    Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    y_tr = tr.effort_hours.to_numpy()
    print(f"  corpus de entrenamiento: {len(tr):,} tareas, mediana {np.median(y_tr):.2f} h")

    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    # SIN normalizar: el corpus se codifico sin normalizar y el regresor se ajusto
    # sobre esos vectores. Normalizar aqui mete la entrada en otra escala y el
    # modelo responde cualquier cosa. La normalizacion se aplica aparte, solo
    # para los cosenos.
    G = {k: st.encode(g[k].tolist(), normalize_embeddings=False, show_progress_bar=False)
         for k in ("es", "en")}
    Gn = {k: v / np.linalg.norm(v, axis=1, keepdims=True) for k, v in G.items()}

    art = joblib.load("models/frio_v4.joblib")
    reg, oh, sc = art["regresor"], art["onehot"], art["escalador"]

    def predecir(X, task_kind="task", source="PBE"):
        """Lo que responderia el modelo desplegado: fuente desconocida, numericas a cero."""
        n = len(X)
        cat = oh.transform(pd.DataFrame({"task_kind": [task_kind] * n,
                                         "source": [source] * n}).astype(str))
        num = sc.transform(np.log1p(np.zeros((n, len(NUM)))))
        return np.exp(reg.predict(np.hstack([X, cat, num])))

    # ---------------------------------------------- 1 · distancia a la variedad
    def vecinos(X, k=10):
        s = X @ Xtr.T
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        top = np.take_along_axis(s, idx, 1)
        orden = np.argsort(-top, axis=1)
        idx = np.take_along_axis(idx, orden, 1)
        return np.take_along_axis(top, orden, 1), idx

    filas = []
    for nombre, k in (("generadas (español)", "es"), ("generadas (inglés)", "en")):
        s, idx = vecinos(Gn[k])
        X, p = G[k], predecir(G[k])
        filas.append(dict(conjunto=nombre, n=len(X), cos_max=float(np.median(s[:, 0])),
                          cos_top10=float(np.median(s.mean(1))),
                          horas_vecinos=float(np.median(np.median(y_tr[idx], axis=1))),
                          horas_predichas=float(np.median(p)),
                          p10=float(np.quantile(p, .1)), p90=float(np.quantile(p, .9))))
    ref = {"real · prueba SiP (comercial)": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
           "real · prueba TAWOS": sw[(sw.split_t3 == "test") & (sw.source == "TAWOS")],
           "real · externo (94 % Apache)": sw[sw.split_t3 == "externo"]}
    rng = np.random.default_rng(0)
    for nombre, d in ref.items():
        m = d.iloc[rng.choice(len(d), min(400, len(d)), replace=False)]
        X = emb[m.fe.to_numpy()]
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        s, idx = vecinos(X)
        p = predecir(X)
        filas.append(dict(conjunto=nombre, n=len(m), cos_max=float(np.median(s[:, 0])),
                          cos_top10=float(np.median(s.mean(1))),
                          horas_vecinos=float(np.median(np.median(y_tr[idx], axis=1))),
                          horas_predichas=float(np.median(p)),
                          p10=float(np.quantile(p, .1)), p90=float(np.quantile(p, .9)),
                          horas_reales=float(np.median(m.effort_hours))))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nDONDE CAEN, COMPARADO CON TAREAS REALES\n{SEP}")
    print(t.round(3).to_string(index=False))

    # ---------------------------------------------- 2 · de donde son los vecinos
    s, idx = vecinos(Gn["es"])
    fuente = tr.source.to_numpy()[idx[:, :5]].ravel()
    jira = tr.fuente_jira.fillna("-").to_numpy()[idx[:, :5]].ravel()
    print(f"\n{SEP}\nDE QUE PARTE DEL CORPUS SALEN LOS 5 VECINOS MAS CERCANOS\n{SEP}")
    print("  fuente:", pd.Series(fuente).value_counts(normalize=True).round(3).to_dict())
    print("  jira:  ", pd.Series(jira).value_counts(normalize=True).head(5).round(3).to_dict())

    # ---------------------------------------------- 3 · ejemplos
    p_es, p_en = predecir(G["es"]), predecir(G["en"])
    g["pred_es"], g["pred_en"], g["cos"] = p_es, p_en, s[:, 0]
    g["vecino"] = tr.title.to_numpy()[idx[:, 0]]
    g["vecino_h"] = y_tr[idx[:, 0]]
    print(f"\n{SEP}\nEJEMPLOS: TAREA GENERADA -> VECINO MAS CERCANO DEL CORPUS\n{SEP}")
    for _, r in g.sample(10, random_state=7).iterrows():
        print(f"  [{r.pred_es:5.2f} h es | {r.pred_en:5.2f} h en | cos {r.cos:.2f}] {r.es[:72]}")
        print(f"        vecino ({r.vecino_h:5.2f} h): {str(r.vecino)[:88]}")
    dif = np.abs(np.log(p_es / p_en))
    print(f"\nmisma tarea en español y en inglés: diferencia mediana "
          f"{np.exp(np.median(dif)):.3f}x, p90 {np.exp(np.quantile(dif, .9)):.3f}x")

    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/tareas_generadas.csv", index=False)
    g.to_csv("results/tareas_generadas_detalle.csv", index=False)
    print("\nescrito results/tareas_generadas.csv y tareas_generadas_detalle.csv")


if __name__ == "__main__":
    main()
