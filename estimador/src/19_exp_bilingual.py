"""Does training on both languages remove the cross-lingual drift?

Compares two ways of training the same head on the same rows:
  en  -- trained on English embeddings only (the current arm)
  bi  -- trained on English AND the Spanish translation of every training row,
         both carrying the same effort label

Each is then scored twice: on the English test set and on the Spanish
translation of that same test set. The number that matters is not accuracy but
the gap between them -- a model useful in production must give the same answer
for the same task regardless of the language it was written in.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
_enc = __import__("15_encode_sbert")

ES_TEXT = "data/processed/text_es.parquet"
ES_CACHE = "data/processed/embeddings/{}.es.npy"


def encode_es(model_name):
    slug = model_name.split("/")[-1]
    path = ES_CACHE.format(slug)
    es = pd.read_parquet(ES_TEXT)
    if os.path.exists(path):
        emb = np.load(path)
        if len(emb) == len(es):
            print(f"cache ES al dia: {path}")
            return es, emb
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"codificando {len(es):,} textos ES en {dev} ...")
    m = SentenceTransformer(model_name, device=dev)
    m.max_seq_length = 256
    emb = m.encode(es.text_es.fillna("").tolist(), batch_size=128 if dev == "cuda" else 32,
                   convert_to_numpy=True, normalize_embeddings=True,
                   show_progress_bar=True)
    np.save(path, emb.astype(np.float32))
    print(f"guardado {path}")
    return es, emb


def main(model_name=_enc.DEFAULT):
    emb_en = _enc.load_embeddings(model_name)
    ids = pd.read_parquet(E.DATA, columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(ids)}

    es_df, emb_es_all = encode_es(model_name)
    assert (es_df.task_id.to_numpy() == ids).all(), "orden ES desalineado"

    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        print(f"\n=== split={split_col} "
              f"train {len(tr):,} val {len(va):,} test {len(te):,} ===")

        def rows(d, emb):
            return emb[[pos[t] for t in d.task_id]]

        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        meta_tr = oh.fit_transform(tr[["task_kind", "source"]])
        meta_va = oh.transform(va[["task_kind", "source"]])
        meta_te = oh.transform(te[["task_kind", "source"]])

        Xtr_en = np.hstack([rows(tr, emb_en), meta_tr])
        Xva_en = np.hstack([rows(va, emb_en), meta_va])
        Xte_en = np.hstack([rows(te, emb_en), meta_te])
        Xte_es = np.hstack([rows(te, emb_es_all), meta_te])
        Xva_es = np.hstack([rows(va, emb_es_all), meta_va])
        Xtr_es = np.hstack([rows(tr, emb_es_all), meta_tr])

        ytr = np.log(tr.effort_hours.to_numpy())
        y_te, y_va = te.effort_hours.to_numpy(), va.effort_hours.to_numpy()

        Xtr_bi = np.vstack([Xtr_en, Xtr_es])
        ytr_bi = np.concatenate([ytr, ytr])

        for head_name, make in [("ridge", lambda: Ridge(alpha=1.0)),
                                ("knn", lambda: KNeighborsRegressor(
                                    n_neighbors=10, metric="cosine",
                                    weights="distance"))]:
            for tag, (X, y, Xv) in [
                ("en", (Xtr_en, ytr, Xva_en)),
                ("bi", (Xtr_bi, ytr_bi, np.vstack([Xva_en, Xva_es]))),
            ]:
                mdl = make().fit(X, y)
                lp_en, lp_es = mdl.predict(Xte_en), mdl.predict(Xte_es)
                yv = y_va if tag == "en" else np.concatenate([y_va, y_va])
                lpv = mdl.predict(Xv)

                m_en = E.point_metrics(y_te, np.exp(lp_en))
                m_en.update(E.interval_metrics(lpv, yv, lp_en, y_te))
                drift = np.exp(lp_es - lp_en)
                m_en["drift_median"] = float(np.median(drift))
                m_en["drift_abs_median"] = float(np.exp(np.median(np.abs(lp_es - lp_en))))
                m_en["drift_within10"] = float(np.mean(np.abs(drift - 1) <= .10))
                m_en["MdMRE_es"] = float(np.median(
                    np.abs(y_te - np.exp(lp_es)) / y_te))
                E.show(E.record("bilingual", f"{head_name}_{tag}", split_col, m_en,
                                config={"model": model_name,
                                        "train_rows": int(len(y))}))
                print(f"    deriva ES/EN mediana {m_en['drift_median']:.3f}  "
                      f"|deriva| {m_en['drift_abs_median']:.3f}x  "
                      f"dentro +-10% {m_en['drift_within10']:.1%}  "
                      f"MdMRE(ES) {m_en['MdMRE_es']:.3f}")

    print("\n--- resumen: deriva cross-lingual ---")
    t = E.table()
    b = t[t.arm == "bilingual"]
    print(b[["variant", "split", "MdMRE", "MdMRE_es", "drift_median",
             "drift_abs_median", "drift_within10"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else _enc.DEFAULT)
