"""Does the encoder actually place Spanish where the English sits?

A smoke test, not the full validation. 24 real test-set tasks were translated
into Spanish by hand (data/processed/probe_es.tsv), keeping the code-switching
a real Spanish-speaking team would use -- identifiers, product names and
error terms stay in English, as they do in practice.

Two things are measured:
  1. cosine(EN, ES) in embedding space -- is the representation aligned?
  2. the ratio between the hours predicted from EN and from ES -- does the
     downstream model actually behave the same?

The second is what matters. A model can have decent cosine alignment and still
shift its prediction, because the regressor may key on directions where the
languages differ.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
_enc = __import__("15_encode_sbert")

PROBE = "data/processed/probe_es.tsv"


def main(model_name=_enc.DEFAULT):
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(E.DATA)
    sw = df[df.is_software_task == 1]
    probe = pd.read_csv(PROBE, sep="\t")
    probe = probe.merge(sw[["task_id", "text", "effort_hours", "task_kind",
                            "source", "split"]], on="task_id", how="inner")
    print(f"pares de sonda: {len(probe)}  (todos en la particion "
          f"{sorted(probe.split.unique())})")

    emb = _enc.load_embeddings(model_name)
    index = {t: i for i, t in enumerate(sw.task_id.to_numpy())}
    # note: index must map over the FULL dataset order used when encoding
    ids_all = pd.read_parquet(E.DATA, columns=["task_id"]).task_id.to_numpy()
    index = {t: i for i, t in enumerate(ids_all)}

    model = SentenceTransformer(model_name)
    model.max_seq_length = 256
    e_es = model.encode(probe.text_es.tolist(), convert_to_numpy=True,
                        normalize_embeddings=True, batch_size=16)
    e_en = np.vstack([emb[index[t]] for t in probe.task_id])

    cos = (e_en * e_es).sum(axis=1)
    print(f"\n--- 1. alineacion del espacio vectorial ---")
    print(f"  cos(EN, ES): mediana {np.median(cos):.3f}  min {cos.min():.3f}  "
          f"max {cos.max():.3f}")
    print(f"  pares con cos >= 0.80: {(cos>=.80).sum()}/{len(cos)}")
    print(f"  pares con cos >= 0.70: {(cos>=.70).sum()}/{len(cos)}")

    # reference: how similar are two UNRELATED tasks in this space?
    rng = np.random.default_rng(0)
    sample = emb[rng.choice(len(emb), 4000, replace=False)]
    rand_cos = (sample[:2000] * sample[2000:]).sum(axis=1)
    print(f"  referencia, dos tareas sin relacion: mediana {np.median(rand_cos):.3f}")

    # ---- 2. does the prediction move? ----
    tr, va, te = E.load("split")
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xtr = np.hstack([emb[[index[t] for t in tr.task_id]],
                     oh.fit_transform(tr[["task_kind", "source"]])])
    model_r = Ridge(alpha=1.0).fit(Xtr, np.log(tr.effort_hours.to_numpy()))

    meta = oh.transform(probe[["task_kind", "source"]])
    p_en = np.exp(model_r.predict(np.hstack([e_en, meta])))
    p_es = np.exp(model_r.predict(np.hstack([e_es, meta])))
    ratio = p_es / p_en
    logdrift = np.abs(np.log(ratio))

    print(f"\n--- 2. desplazamiento de la prediccion (ES frente a EN) ---")
    print(f"  razon ES/EN: mediana {np.median(ratio):.3f}  "
          f"p10 {np.quantile(ratio,.1):.3f}  p90 {np.quantile(ratio,.9):.3f}")
    print(f"  desviacion absoluta mediana: {np.exp(np.median(logdrift)):.3f}x")
    print(f"  predicciones dentro de +-10%: {(np.abs(ratio-1)<=.10).sum()}/{len(ratio)}")
    print(f"  predicciones dentro de +-25%: {(np.abs(ratio-1)<=.25).sum()}/{len(ratio)}")

    y = probe.effort_hours.to_numpy()
    for lang, p in [("EN", p_en), ("ES", p_es)]:
        mre = np.abs(y - p) / y
        print(f"  MdMRE sobre la sonda ({lang}): {np.median(mre):.3f}")

    out = probe[["task_id", "task_kind", "effort_hours"]].copy()
    out["cos"] = cos.round(3)
    out["pred_en"] = p_en.round(2)
    out["pred_es"] = p_es.round(2)
    out["ratio"] = ratio.round(3)
    print(f"\n{out.sort_values('cos').to_string(index=False)}")
    out.to_csv("results/crosslingual_probe.csv", index=False)
    print("\nescrito results/crosslingual_probe.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else _enc.DEFAULT)
