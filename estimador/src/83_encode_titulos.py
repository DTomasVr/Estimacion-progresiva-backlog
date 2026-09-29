"""Encode the titles alone, as a second view of the same corpus.

The tasks the generator hands the estimator are 10-word imperatives with no
description. The corpus's `text` is title + description: 44 words in the public
Jira block, 47 in TAWOS. The titles, though, are 7-8 words -- already the length
of what we will actually receive. Encoding them gives the zero-cost version of
"rewrite the corpus in the target format", with no LLM and no invented labels.

Same frozen encoder and the same id alignment convention as 53_encode_v4.py, so
the two views are interchangeable downstream.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
DATA = "data/processed/agix_effort_v4.parquet"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SALIDA = "data/processed/embeddings/v4_mpnet_titulos"


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(DATA, columns=["task_id", "title", "source"])
    t = df.title.fillna("").astype(str)
    vacios = int((t.str.strip() == "").sum())
    print(f"{len(df):,} titulos, {vacios} vacios, "
          f"mediana {t.str.split().str.len().median():.0f} palabras")

    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    emb = st.encode(t.tolist(), batch_size=64, show_progress_bar=False,
                    convert_to_numpy=True).astype(np.float32)
    np.save(f"{SALIDA}.npy", emb)
    np.save(f"{SALIDA}.ids.npy", df.task_id.to_numpy())
    print(f"escrito {SALIDA}.npy {emb.shape} en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
