"""Encode v4 with the frozen multilingual encoder, reusing what we already have.

Only the JiraPublico rows are new, so the existing cache is reused verbatim for
every v3 row. That matters beyond speed: reusing the exact same vectors keeps
the A/B honest, since a re-encode could shift v3's own embeddings by a floating
point hair and muddy the comparison we are about to run.

The encoder stays FROZEN. The fine-tuned one was fitted on v3's temporal-train
only, so applying it to v4 would carry v3-specific adaptation into a comparison
about data volume -- and would silently re-open the leakage question we closed.
"""
import os
import time

import numpy as np
import pandas as pd

V4 = "data/processed/agix_effort_v4.parquet"
CACHE = "data/processed/embeddings"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SALIDA = f"{CACHE}/v4_mpnet_congelado"


def main():
    df = pd.read_parquet(V4, columns=["task_id", "text", "source"])
    ids_v4 = df.task_id.to_numpy()

    viejo_emb = np.load(f"{CACHE}/paraphrase-multilingual-mpnet-base-v2.npy")
    viejo_ids = np.load(f"{CACHE}/paraphrase-multilingual-mpnet-base-v2.ids.npy",
                        allow_pickle=True)
    pos = {t: i for i, t in enumerate(viejo_ids)}
    print(f"cache existente: {viejo_emb.shape}")

    tiene = np.array([t in pos for t in ids_v4])
    print(f"filas de v4 ya codificadas: {tiene.sum():,} de {len(df):,}")
    faltan = df[~tiene]
    print(f"por codificar: {len(faltan):,}  ({faltan.source.value_counts().to_dict()})")

    dim = viejo_emb.shape[1]
    emb = np.zeros((len(df), dim), dtype=np.float32)
    emb[tiene] = viejo_emb[[pos[t] for t in ids_v4[tiene]]]

    if len(faltan):
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{dev}] cargando {MODELO}")
        m = SentenceTransformer(MODELO, device=dev)
        m.max_seq_length = 256
        t0 = time.time()
        nuevo = m.encode(faltan.text.fillna("").tolist(), batch_size=256,
                         show_progress_bar=True, convert_to_numpy=True,
                         normalize_embeddings=True)
        seg = time.time() - t0
        print(f"  {len(faltan):,} textos en {seg:.0f}s ({len(faltan)/seg:.0f}/s)")
        emb[~tiene] = nuevo.astype(np.float32)

    os.makedirs(CACHE, exist_ok=True)
    np.save(f"{SALIDA}.npy", emb)
    np.save(f"{SALIDA}.ids.npy", ids_v4)
    print(f"\nguardado {SALIDA}.npy  shape={emb.shape}  "
          f"{os.path.getsize(SALIDA + '.npy')/1e6:.0f} MB")

    # las filas reutilizadas deben ser identicas bit a bit
    j = np.where(tiene)[0][:5]
    ok = all(np.array_equal(emb[i], viejo_emb[pos[ids_v4[i]]]) for i in j)
    print(f"filas reutilizadas identicas al cache: {ok}")
    n = np.linalg.norm(emb, axis=1)
    print(f"norma: min {n.min():.4f}  max {n.max():.4f}  (deben ser 1)")


if __name__ == "__main__":
    main()
