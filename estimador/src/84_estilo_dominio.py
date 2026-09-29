"""Three cheap interventions, measured on the frozen blocks before anything is generated.

They come from reading the real output of the task generator: 10-word Spanish
imperatives with no description, product/business domain.

  1. LANGUAGE   the generator already produces English internally and translates
                only for display. Feeding the estimator the English costs
                nothing, and this measures what feeding Spanish would cost, on
                the real Spanish translation of the v3 corpus.
  2. STYLE      train on TITLES ONLY (7-8 words) instead of title+description
                (44). Zero-cost version of "rewrite the corpus in the target
                format": no LLM, no invented labels.
  3. DOMAIN     drop Apache (74 % of training) or downweight it, since the
                domain fixes the hour scale and Apache's is 0,83 h against the
                2,00 h of commercial work.

Crossed with the feature set, because a deployment-honest model cannot use
`source` (Agix would be an unseen level) nor the link/component counts (they
accumulate while the task is worked, and are zero when it is created).

Nothing synthetic is involved and the test blocks are the frozen ones.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 230)
DATA = "data/processed/agix_effort_v4.parquet"
V3 = "data/processed/agix_effort_v3.parquet"
EMB_TXT = "data/processed/embeddings/v4_mpnet_congelado"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
EMB_V3_EN = "data/processed/embeddings/paraphrase-multilingual-mpnet-base-v2"
EMB_V3_ES = "data/processed/embeddings/paraphrase-multilingual-mpnet-base-v2.es"
GEN = Path("F:/Software Task generation Model/salida_gen.json")
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
SEP = "=" * 150


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    m = E.point_metrics(y, p)
    lr = np.log(p / np.asarray(y, float))
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))), PRED25=m["PRED25"],
                MAE_h=m["MAE_h"], sesgo=float(np.exp(np.median(lr))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def cargar(path):
    e = np.load(f"{path}.npy")
    i = np.load(f"{path}.ids.npy", allow_pickle=True)
    return e, {t: k for k, t in enumerate(i)}


# --------------------------------------------------------------- 1 · idioma
def idioma():
    """Lo que cuesta darle al modelo el español en vez del inglés, medido en v3."""
    df = pd.read_parquet(V3)
    sw = df[df.is_software_task == 1].copy()
    es = pd.read_parquet("data/processed/text_es.parquet")
    sw = sw[sw.task_id.isin(es.task_id)]
    en_e, en_p = cargar(EMB_V3_EN)
    # los vectores en español no llevan fichero de ids: van en el orden de
    # text_es.parquet, que es el de agix_effort_clean.parquet (ver 19_exp_bilingual)
    es_e = np.load(f"{EMB_V3_ES}.npy")
    es_p = {t: i for i, t in enumerate(es.task_id.to_numpy())}
    assert len(es_e) == len(es_p), "vectores en español desalineados con text_es"
    tr, te = sw[sw.split == "train"], sw[sw.split == "test"]
    Xtr = en_e[[en_p[t] for t in tr.task_id]]
    m = hgb().fit(Xtr, np.log(tr.effort_hours.to_numpy()))
    y = te.effort_hours.to_numpy()
    filas = []
    for nombre, e, p in (("entrada en inglés (como se entrenó)", en_e, en_p),
                         ("entrada en español (traducida)", es_e, es_p)):
        X = e[[p[t] for t in te.task_id]]
        filas.append(dict(entrada=nombre, n=len(te), **metricas(y, np.exp(m.predict(X)))))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\n1 · IDIOMA DE ENTRADA · modelo entrenado en inglés, prueba de v3\n{SEP}")
    print(t.round(4).to_string(index=False))
    a, b = t.MdMRE.iloc[0], t.MdMRE.iloc[1]
    print(f"  darle el español cuesta {b - a:+.4f} de MdMRE ({b / a:.3f}x) y "
          f"{t.MdAElog.iloc[1] / t.MdAElog.iloc[0]:.3f}x de error logarítmico")
    return t


# --------------------------------------------------- 2 y 3 · estilo y dominio
def rejilla():
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    vistas = {"título + descripción": cargar(EMB_TXT), "solo título": cargar(EMB_TIT)}

    gen = json.load(open(GEN, encoding="utf-8"))
    g_es, g_en = [], []
    for d in gen.values():
        for h in d["historias"]:
            es, en = h.get("tareas_es") or [], h.get("tareas") or []
            for i, t in enumerate(es):
                if i < len(en):
                    g_es.append(t)
                    g_en.append(en[i])
    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    Xg = {k: st.encode(v, normalize_embeddings=False, show_progress_bar=False)
          for k, v in (("es", g_es), ("en", g_en))}
    print(f"tareas generadas para la sonda: {len(g_es)}")

    tr_all = sw[sw.split_t3 == "train"]
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "prueba TAWOS": sw[(sw.split_t3 == "test") & (sw.source == "TAWOS")],
               "externo": sw[sw.split_t3 == "externo"],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    val = sw[sw.split_t3 == "val"]

    filas, sondas = [], []
    for vista, (emb, pos) in vistas.items():
        idx = lambda d: [pos[t] for t in d.task_id]
        oh_c = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=25).fit(tr_all[["task_kind"]].astype(str))
        oh_f = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=25).fit(tr_all[["task_kind", "source"]].astype(str))
        sc = StandardScaler().fit(np.log1p(tr_all[NUM].to_numpy(float)))
        rasgos = {
            "actual (vector+cat+fuente+num)":
                (lambda d: np.hstack([emb[idx(d)], oh_f.transform(d[["task_kind", "source"]].astype(str)),
                                      sc.transform(np.log1p(d[NUM].to_numpy(float)))]),
                 lambda X, n: np.hstack([X, oh_f.transform(pd.DataFrame(
                     {"task_kind": ["task"] * n, "source": ["Agix"] * n}).astype(str)),
                     sc.transform(np.log1p(np.zeros((n, len(NUM)))))])),
            "despliegue (vector+categoría)":
                (lambda d: np.hstack([emb[idx(d)], oh_c.transform(d[["task_kind"]].astype(str))]),
                 lambda X, n: np.hstack([X, oh_c.transform(pd.DataFrame(
                     {"task_kind": ["task"] * n}).astype(str))])),
        }
        for nombre_r, (f, fg) in rasgos.items():
            Xtr_all, ytr_all = f(tr_all), np.log(tr_all.effort_hours.to_numpy())
            corpus = {
                "todo": (tr_all, None),
                "sin Apache": (tr_all[~tr_all.apache], None),
                "Apache x0,25": (tr_all, np.where(tr_all.apache, .25, 1.)),
            }
            for nombre_c, (d, w) in corpus.items():
                if w is None and nombre_c != "todo":
                    X, y = f(d), np.log(d.effort_hours.to_numpy())
                    m = hgb().fit(X, y)
                else:
                    m = hgb().fit(Xtr_all, ytr_all, sample_weight=w)
                base = dict(vista=vista, rasgos=nombre_r, corpus=nombre_c, n_train=len(d))
                mv = metricas(val.effort_hours.to_numpy(), np.exp(m.predict(f(val))))
                for b, p in bloques.items():
                    filas.append(dict(**base, bloque=b, n=len(p), val_MdAElog=mv["MdAElog"],
                                      **metricas(p.effort_hours.to_numpy(),
                                                 np.exp(m.predict(f(p))))))
                for lang in ("es", "en"):
                    pr = np.exp(m.predict(fg(Xg[lang], len(g_es))))
                    sondas.append(dict(**base, idioma=lang, mediana_h=float(np.median(pr)),
                                       p10=float(np.quantile(pr, .1)),
                                       p90=float(np.quantile(pr, .9)),
                                       dispersion=float(np.quantile(pr, .9) / np.quantile(pr, .1))))
                print(f"  ajustado [{vista}] [{nombre_r}] [{nombre_c}] n={len(d):,}", flush=True)
    return pd.DataFrame(filas), pd.DataFrame(sondas)


def main():
    t_idioma = idioma()
    r, s = rejilla()
    Path("results").mkdir(exist_ok=True)
    t_idioma.to_csv("results/estilo_idioma.csv", index=False)
    r.to_csv("results/estilo_dominio.csv", index=False)
    s.to_csv("results/estilo_sonda_generadas.csv", index=False)

    cols = ["vista", "rasgos", "corpus", "n_train", "val_MdAElog", "MdMRE", "MdAElog",
            "PRED25", "MAE_h", "sesgo", "spearman"]
    for b in r.bloque.unique():
        v = r[r.bloque == b].sort_values("MdAElog")
        print(f"\n{SEP}\n{b} · n={v.n.iloc[0]:,}\n{SEP}")
        print(v[cols].round(4).to_string(index=False))

    print(f"\n{SEP}\nSONDA: QUE RESPONDE CADA CONFIGURACION A LAS 115 TAREAS GENERADAS\n{SEP}")
    print(s.sort_values(["vista", "rasgos", "corpus", "idioma"]).round(3).to_string(index=False))
    print("\nescrito results/estilo_idioma.csv, estilo_dominio.csv, estilo_sonda_generadas.csv")


if __name__ == "__main__":
    main()
