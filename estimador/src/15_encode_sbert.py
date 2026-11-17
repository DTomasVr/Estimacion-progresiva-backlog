"""Encode the dataset with a cross-lingually aligned sentence encoder.

Embeddings are cached to disk keyed by model name, so the encoding cost is paid
once and every downstream experiment reuses it.

Model choice rationale: the corpus is English but PBE serves Spanish, so the
encoder must place a sentence and its translation at (nearly) the same point.
`paraphrase-multilingual-*` are distilled with exactly that objective
(Reimers & Gurevych, 2020), which a Spanish-only encoder cannot provide since
it shares no space with the English training corpus.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

DATA = "data/processed/pbe_effort_clean.parquet"
CACHE = "data/processed/embeddings"
DEFAULT = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def cache_path(model_name):
    slug = model_name.split("/")[-1]
    return f"{CACHE}/{slug}.npy", f"{CACHE}/{slug}.ids.npy"


def encode(model_name=DEFAULT, batch_size=32, max_seq=256):
    from sentence_transformers import SentenceTransformer

    os.makedirs(CACHE, exist_ok=True)
    emb_p, ids_p = cache_path(model_name)
    df = pd.read_parquet(DATA, columns=["task_id", "text"])
    if os.path.exists(emb_p):
        ids = np.load(ids_p, allow_pickle=True)
        if len(ids) == len(df) and (ids == df.task_id.to_numpy()).all():
            print(f"cache al dia: {emb_p}")
            return np.load(emb_p)
        print("cache obsoleta, recodificando")

    print(f"cargando {model_name} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq
    print(f"  cargado en {time.time()-t0:.0f}s, dim={model.get_sentence_embedding_dimension()}")

    texts = df.text.fillna("").tolist()
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    print(f"  {len(texts):,} textos en {time.time()-t0:.0f}s "
          f"({len(texts)/max(time.time()-t0,1):.0f}/s)")

    np.save(emb_p, emb.astype(np.float32))
    np.save(ids_p, df.task_id.to_numpy())
    print(f"guardado {emb_p}  shape={emb.shape}  "
          f"{os.path.getsize(emb_p)/1e6:.0f} MB")
    return emb


def load_embeddings(model_name=DEFAULT):
    """Embeddings aligned to the row order of the dataset."""
    emb_p, ids_p = cache_path(model_name)
    emb = np.load(emb_p)
    ids = np.load(ids_p, allow_pickle=True)
    df = pd.read_parquet(DATA, columns=["task_id"])
    assert (ids == df.task_id.to_numpy()).all(), "cache desalineada con los datos"
    return emb


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    encode(name)
