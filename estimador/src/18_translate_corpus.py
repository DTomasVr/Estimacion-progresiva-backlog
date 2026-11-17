"""Translate the corpus to Spanish so the heads can be trained on both languages.

The cross-lingual probe showed that embedding alignment alone leaves a
systematic +23.5% drift in the predicted hours. Training the downstream heads on
Spanish surface forms as well as English is the direct fix: instead of trusting
the encoder's generic alignment, the regressor sees both languages of the same
labelled task.

Uses Helsinki-NLP/opus-mt-en-es (MarianMT). Placeholder tokens introduced during
cleaning (<NUM>, <URL>, <STACK>, <HASH>, <EMAIL>) are masked out before
translation and restored afterwards, because MT models mangle them.
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

DATA = "data/processed/pbe_effort_clean.parquet"
OUT = "data/processed/text_es.parquet"
MODEL = "Helsinki-NLP/opus-mt-en-es"
PLACEHOLDER = re.compile(r"<(NUM|URL|STACK|HASH|EMAIL)>")
MAX_CHARS = 1200  # Marian degrades badly past its 512-token window


def mask(text):
    """Replace placeholders with tokens MT will not touch, keeping their order."""
    kinds = []

    def sub(m):
        kinds.append(m.group(1))
        return f" QQ{len(kinds)-1}QQ "

    return PLACEHOLDER.sub(sub, text), kinds


def unmask(text, kinds):
    def sub(m):
        i = int(m.group(1))
        return f"<{kinds[i]}>" if i < len(kinds) else ""
    return re.sub(r"QQ\s*(\d+)\s*QQ", sub, text)


def main():
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    df = pd.read_parquet(DATA, columns=["task_id", "text"])
    print(f"textos a traducir: {len(df):,}")

    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()

    masked, kinds_all = [], []
    for t in df.text.fillna(""):
        m, k = mask(t[:MAX_CHARS])
        masked.append(m)
        kinds_all.append(k)

    # sort by length so each batch is uniformly sized -- far less padding waste
    order = np.argsort([len(s) for s in masked])
    out = [None] * len(masked)
    batch = 64 if device == "cuda" else 16
    t0 = time.time()

    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        chunk = [masked[i] for i in idx]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = unmask(d, kinds_all[i])
        done = start + len(idx)
        if start % (batch * 20) == 0 or done >= len(order):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"\r  {done:,}/{len(order):,}  {rate:.0f}/s  "
                  f"ETA {(len(order)-done)/max(rate,1e-9)/60:.1f} min",
                  end="", file=sys.stderr)

    print(file=sys.stderr)
    res = pd.DataFrame({"task_id": df.task_id, "text_es": out})
    res.to_parquet(OUT, compression="zstd", index=False)
    print(f"\nescrito {OUT}  ({len(res):,} filas, {time.time()-t0:.0f}s)")

    print("\n--- muestra ---")
    for i in range(5):
        print(f"  EN: {df.text.iloc[i][:78]}")
        print(f"  ES: {res.text_es.iloc[i][:78]}\n")

    kept = sum(1 for a, b in zip(df.text, res.text_es)
               if len(PLACEHOLDER.findall(a)) == len(PLACEHOLDER.findall(b)))
    print(f"marcadores preservados en {kept:,}/{len(res):,} textos "
          f"({kept/len(res)*100:.1f}%)")
    empty = res.text_es.fillna("").str.strip().eq("").sum()
    print(f"traducciones vacias: {empty}")


if __name__ == "__main__":
    main()
