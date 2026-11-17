"""Frozen vs fine-tuned SBERT on the TEST projects (never seen in training
nor in model selection). Retrieval of a parent's true children among all test
children; en and es queries; per relation. Paired bootstrap over queries for a
95 % CI of the nDCG@10 difference -- the adoption criterion is fixed up front:
adopt the fine-tuned model iff the CI lower bound is > 0 for BOTH en and es on
the pooled test set.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
MODELOS = {"congelado": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
           "ajustado": f"{BASE}/models/sbert_pbe_v1"}
if len(sys.argv) > 1:
    MODELOS["ajustado"] = sys.argv[1]
K = 10
B = 2000
SEED = 20260922


def ndcg_por_query(sims, rel_sets):
    top = torch.topk(sims, K, dim=1).indices.cpu().numpy()
    disc = 1 / np.log2(np.arange(2, K + 2))
    out = np.empty(len(rel_sets))
    rec = np.empty(len(rel_sets))
    for i, (fila, rel) in enumerate(zip(top, rel_sets)):
        g = np.array([j in rel for j in fila], float)
        ideal = disc[:min(len(rel), K)].sum()
        out[i] = (g * disc).sum() / ideal
        rec[i] = g.sum() / min(len(rel), K)
    return out, rec


def main():
    P = pd.read_parquet(f"{BASE}/data/processed/sbert_pairs.parquet")
    T = P[P.split == "test"]
    corpus = T.positive.drop_duplicates().tolist()
    cid = {t: i for i, t in enumerate(corpus)}
    Q = T.groupby("parent_uid").agg(en=("anchor_en", "first"), es=("anchor_es", "first"),
                                    rel=("relation", "first"),
                                    pos=("positive", lambda s: {cid[x] for x in s}))
    Q = Q[Q.es.notna()]
    print(f"test: {len(Q):,} consultas, corpus {len(corpus):,}")

    res = {}
    for nombre, ruta in MODELOS.items():
        m = SentenceTransformer(ruta, device="cuda")
        m.max_seq_length = 128
        C = m.encode(corpus, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
        for lang in ["en", "es"]:
            q = m.encode(Q[lang].tolist(), batch_size=128, convert_to_tensor=True,
                         normalize_embeddings=True)
            res[(nombre, lang)] = ndcg_por_query(q @ C.T, Q.pos.tolist())
        del m, C
        torch.cuda.empty_cache()

    rng = np.random.default_rng(SEED)
    filas = []
    for lang in ["en", "es"]:
        for rel in ["todas", "epic", "subtask"]:
            m = np.ones(len(Q), bool) if rel == "todas" else (Q.rel == rel).values
            a, ra = res[("congelado", lang)][0][m], res[("congelado", lang)][1][m]
            b, rb = res[("ajustado", lang)][0][m], res[("ajustado", lang)][1][m]
            d = b - a
            idx = rng.integers(0, len(d), (B, len(d)))
            bs = d[idx].mean(1)
            filas.append(dict(lang=lang, relacion=rel, n=int(m.sum()),
                              ndcg10_congelado=a.mean(), ndcg10_ajustado=b.mean(),
                              recall10_congelado=ra.mean(), recall10_ajustado=rb.mean(),
                              delta=d.mean(), ic95_lo=np.quantile(bs, .025), ic95_hi=np.quantile(bs, .975)))
    R = pd.DataFrame(filas).round(4)
    print(R.to_string(index=False))
    R.to_csv(f"{BASE}/results/07_sbert_test.csv", index=False)
    adopta = bool((R[(R.relacion == "todas")].ic95_lo > 0).all())
    print(f"\nCRITERIO (IC95 inferior > 0 en en y es): {'ADOPTAR ajustado' if adopta else 'MANTENER congelado'}")
    json.dump(dict(adoptar_ajustado=adopta, modelo=MODELOS["ajustado"]),
              open(f"{BASE}/results/07_sbert_decision.json", "w"), indent=1)


if __name__ == "__main__":
    main()
