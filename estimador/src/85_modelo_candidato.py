"""Train and save the deployment candidate.

The configuration comes from what was measured in 81, 82, 83 and 84, not from a
preference:

  vista   TITULOS. The estimator will receive 10-word imperatives with no
          description. Training on title+description costs nothing on the frozen
          blocks but makes the model answer a median of 9 h to the real
          generated tasks, against 2,6 h when trained on titles.
  idioma  INGLES. The generator produces English internally and translates only
          for display; feeding the estimator the Spanish costs +0,0297 of MdMRE
          on the v3 test.
  corpus  SIN APACHE. Apache is 74 % of training and its hour scale (0,83 h) is
          not the scale of product work (2,00 h). Dropping it takes the bias on
          non-Apache held-out projects from 0,44 to 0,92, with 16.648 training
          rows instead of 63.926.
  rasgos  No `source` (Agix would be an unseen level, encoded as zeros) and no
          link/component counters (they accumulate while the task is worked and
          are zero when it is created). Whether the harmonised category is worth
          keeping is decided here, on the non-Apache blocks.

Selection note: the validation block is Apache-dominated too, so it ranks these
configurations the wrong way round. The candidate is chosen on the held-out
blocks that resemble the target -- SiP and the non-Apache external projects --
and that choice is stated rather than hidden.

Saves models/candidato_titulos.joblib with the regressor, the encoder, the
conformal quantiles and the metadata needed to serve it.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
CODIFICADOR = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GEN = Path("F:/Software Task generation Model/salida_gen.json")
SALIDA = Path("models/candidato_titulos.joblib")
SEP = "=" * 128


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    m = E.point_metrics(y, p)
    lr = np.log(p / np.asarray(y, float))
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))), PRED25=m["PRED25"],
                MAE_h=m["MAE_h"], sesgo=float(np.exp(np.median(lr))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]

    tr = sw[(sw.split_t3 == "train") & (~sw.apache)]
    va = sw[(sw.split_t3 == "val") & (~sw.apache)]
    va_todo = sw[sw.split_t3 == "val"]
    bloques = {"prueba fija": sw[sw.split_t3 == "test"],
               "prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "prueba TAWOS": sw[(sw.split_t3 == "test") & (sw.source == "TAWOS")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)],
               "externo (94 % Apache)": sw[sw.split_t3 == "externo"]}
    print(f"entrenamiento {len(tr):,} (sin Apache) · validación {len(va):,} sin Apache "
          f"de {len(va_todo):,}")
    print("bloques: " + " · ".join(f"{k} {len(v):,}" for k, v in bloques.items()))

    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(tr[["task_kind"]].astype(str))
    variantes = {
        "solo el texto del título": (lambda d: emb[d.fe.to_numpy()], None),
        "título + categoría": (
            lambda d: np.hstack([emb[d.fe.to_numpy()],
                                 oh.transform(d[["task_kind"]].astype(str))]), oh),
    }

    filas, modelos = [], {}
    for nombre, (f, _) in variantes.items():
        t0 = time.time()
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        m.fit(f(tr), np.log(tr.effort_hours.to_numpy()))
        modelos[nombre] = m
        print(f"  ajustado «{nombre}» en {time.time() - t0:.0f} s")
        for b, d in bloques.items():
            filas.append(dict(variante=nombre, bloque=b, n=len(d),
                              **metricas(d.effort_hours.to_numpy(), np.exp(m.predict(f(d))))))
    r = pd.DataFrame(filas)
    print(f"\n{SEP}\nLAS DOS VARIANTES SOBRE LOS BLOQUES CONGELADOS\n{SEP}")
    for b in bloques:
        v = r[r.bloque == b]
        print(f"\n{b} · n={v.n.iloc[0]:,}")
        print(v[["variante", "MdMRE", "MdAElog", "PRED25", "MAE_h", "sesgo",
                 "spearman"]].round(4).to_string(index=False))

    # el candidato se elige en los bloques que se parecen al destino, no en validación
    objetivo = r[r.bloque.isin(["prueba SiP", "externo no Apache"])]
    elegido = objetivo.groupby("variante").MdAElog.mean().idxmin()
    f = variantes[elegido][0]
    m = modelos[elegido]
    print(f"\ncandidato elegido sobre SiP + externo no Apache: «{elegido}»")

    # bandas conformes: calibradas en la validación que se parece al destino
    bandas = {}
    for nombre, d in (("validación sin Apache", va), ("validación completa", va_todo)):
        res = np.abs(np.log(d.effort_hours.to_numpy()) - m.predict(f(d)))
        bandas[nombre] = {f"factor{int(q * 100)}": float(np.exp(np.quantile(res, q)))
                          for q in (.5, .8, .9)}
    print("\nbandas conformes (factor multiplicativo):")
    for k, v in bandas.items():
        print(f"  {k:24s} " + " · ".join(f"{a} ×{b:.2f}" for a, b in v.items()))
    cob = []
    q80 = np.log(bandas["validación sin Apache"]["factor80"])
    for b, d in bloques.items():
        lp = m.predict(f(d))
        y = d.effort_hours.to_numpy()
        cob.append(dict(bloque=b, cobertura80=float(np.mean(
            (y >= np.exp(lp - q80)) & (y <= np.exp(lp + q80))))))
    print("cobertura real de la banda del 80 %: " +
          " · ".join(f"{c['bloque']} {c['cobertura80']:.3f}" for c in cob))

    # sonda: lo que responderia a las tareas que genera el otro modulo
    sonda = {}
    if GEN.exists():
        import torch
        from sentence_transformers import SentenceTransformer

        gen = json.load(open(GEN, encoding="utf-8"))
        tex = {"es": [], "en": []}
        for d in gen.values():
            for h in d["historias"]:
                es, en = h.get("tareas_es") or [], h.get("tareas") or []
                for i, t in enumerate(es):
                    if i < len(en):
                        tex["es"].append(t)
                        tex["en"].append(en[i])
        st = SentenceTransformer(CODIFICADOR,
                                 device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nsonda sobre {len(tex['en'])} tareas generadas:")
        for lang in ("en", "es"):
            X = st.encode(tex[lang], normalize_embeddings=False, show_progress_bar=False)
            if variantes[elegido][1] is not None:
                X = np.hstack([X, oh.transform(pd.DataFrame(
                    {"task_kind": ["task"] * len(X)}).astype(str))])
            p = np.exp(m.predict(X))
            sonda[lang] = dict(mediana=float(np.median(p)), p10=float(np.quantile(p, .1)),
                               p90=float(np.quantile(p, .9)))
            print(f"  {lang}: mediana {sonda[lang]['mediana']:.2f} h · "
                  f"p10–p90 {sonda[lang]['p10']:.2f}–{sonda[lang]['p90']:.2f} h")

    SALIDA.parent.mkdir(exist_ok=True)
    joblib.dump({
        "regresor": m, "onehot": variantes[elegido][1], "variante": elegido,
        "codificador": CODIFICADOR, "vista": "titulo", "idioma_esperado": "ingles",
        "objetivo": "log(horas)", "normalizar_vectores": False,
        "bandas": bandas, "corpus": "v4 sin Apache", "n_train": int(len(tr)),
        "mediana_global_h": float(np.median(tr.effort_hours)),
        "sha_datos": E.data_fingerprint(DATA), "fecha": time.strftime("%Y-%m-%d"),
    }, SALIDA, compress=3)
    r.to_csv("results/candidato.csv", index=False)
    json.dump({"elegido": elegido, "bandas": bandas, "cobertura80": cob,
               "sonda_generadas": sonda, "n_train": int(len(tr))},
              open("results/candidato.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {SALIDA} ({SALIDA.stat().st_size / 1e6:.2f} MB), "
          f"results/candidato.csv y candidato.json")
    print("""
uso:
    import joblib, numpy as np, pandas as pd
    from sentence_transformers import SentenceTransformer
    a = joblib.load("models/candidato_titulos.joblib")
    st = SentenceTransformer(a["codificador"])
    textos = ["Store payment data in the member profile"]     # INGLES, titulo solo
    X = st.encode(textos, normalize_embeddings=False)          # SIN normalizar
    if a["onehot"] is not None:                                # variante elegida
        cat = pd.DataFrame({"task_kind": ["task"] * len(textos)}).astype(str)
        X = np.hstack([X, a["onehot"].transform(cat)])
    horas = np.exp(a["regresor"].predict(X))
    k = a["bandas"]["validación sin Apache"]["factor80"]
    bajo, alto = horas / k, horas * k""")


if __name__ == "__main__":
    main()
