"""Rewrite the INPUT of story->tasks examples as a user story, with a local
LLM (mistral:7b, temperature 0). The targets -- the real tasks teams wrote --
are never touched, so this is paraphrase augmentation of the input side, not
synthetic training data.

Measured: ~1.2-1.5 s per item with mistral:7b, 98.7 % come out in the
right form. Saves every FLUSH items and resumes, so it can be interrupted.

out: data/processed/historias_reescritas.parquet  (parent_uid, texto, historia)
"""
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

BASE = r"F:\Software Task generation Model"
OUT = f"{BASE}/data/processed/historias_reescritas_mistral.parquet"
P_TRAIN = 0.5      # training mixes both styles 50/50, so half of train is enough
OLLAMA = "http://127.0.0.1:11434/api/generate"
# mistral is Apache 2.0: its outputs carry no naming/derivative condition, unlike
# the Llama 3.1 community licence (which would force a "Llama-" model name).
LLM = "mistral:7b"
FLUSH = 250
PROMPT = ("Rewrite this software backlog item as ONE user story, in exactly this form:\n"
          "As a <role>, I want <goal>, so that <benefit>.\n"
          "Keep the same scope and the same technical terms. Do not add new features. "
          "Output only the user story, nothing else.\n\nBacklog item: ")


def reescribir(texto):
    body = json.dumps({"model": LLM, "prompt": PROMPT + texto, "stream": False,
                       "options": {"temperature": 0, "num_predict": 120}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))["response"].strip()
    return re.sub(r"\s+", " ", r.split("\n")[0].strip().strip('"'))


def main():
    partes = []
    for sp in ["train", "val", "test"]:
        d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_{sp}.parquet",
                            columns=["parent_uid", "mitad", "src"])
        d = d[d.mitad == "tasks"]
        d["texto"] = [s.split(" | examples: ")[0].replace("decompose story into tasks: ", "")
                      for s in d.src]
        d["split"] = sp
        partes.append(d[["parent_uid", "split", "texto"]])
    T = pd.concat(partes, ignore_index=True).drop_duplicates("parent_uid")
    tr = T[T.split == "train"].sample(frac=P_TRAIN, random_state=20260922)
    T = pd.concat([tr, T[T.split != "train"]], ignore_index=True)

    hecho = {}
    if os.path.exists(OUT):
        y = pd.read_parquet(OUT)
        hecho = dict(zip(y.parent_uid, y.historia))
        print(f"reanudando: {len(hecho):,} ya reescritas")
    pend = T[~T.parent_uid.isin(hecho)].reset_index(drop=True)
    print(f"{len(T):,} entradas, faltan {len(pend):,}", flush=True)

    t0 = time.time()
    for i, (uid, txt) in enumerate(zip(pend.parent_uid, pend.texto), 1):
        try:
            hecho[uid] = reescribir(txt[:1200])
        except Exception as e:                      # keep going; retry on the next run
            print(f"  fallo en {uid}: {e}", flush=True)
            continue
        if i % FLUSH == 0 or i == len(pend):
            T.assign(historia=T.parent_uid.map(hecho)).dropna(subset=["historia"]).to_parquet(OUT, index=False)
            eta = (len(pend) - i) * (time.time() - t0) / i
            ok = sum(bool(re.match(r"(?i)^as an? .+ i want", v)) for v in hecho.values()) / len(hecho)
            print(f"  {i:>6,}/{len(pend):,}  {(time.time()-t0)/60:5.1f} min  ETA {eta/60:5.1f} min  "
                  f"formato ok {ok:.1%}", flush=True)
    print(f"listo: {len(hecho):,} reescritas en {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
