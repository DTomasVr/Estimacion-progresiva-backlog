"""Contrastive pairs for the retrieval SBERT: (parent -> child) from both
hierarchy relations (epic -> child, parent -> sub-task), split by project.

Reactive work (bugs, suggestions, support) is dropped on either side: the
model is for "what planned work goes under this", not for bug triage.

Spanish: every unique parent summary is translated once with the cached
Helsinki-NLP/opus-mt-en-es. In train, 30 % of pairs get the Spanish anchor
(replacing, not adding, so volume is unchanged); val/test keep both versions
so retrieval can be scored es->en, which is how PBE queries arrive.

Output: data/processed/sbert_pairs.parquet
        (split, relation, parent_uid, child_uid, anchor_en, anchor_es, positive)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from textclean import limpiar, texto_issue  # noqa: E402

BASE = r"F:\Software Task generation Model"
SEED = 20260922
P_ES = 0.30
REACTIVO = ("bug", "suggestion", "question", "support", "incident", "problem", "rca")
t0 = time.time()


def fase(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def reactivo(t):
    return isinstance(t, str) and any(k in t.lower() for k in REACTIVO)


E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
S = pd.read_parquet(f"{BASE}/data/processed/split_projects.parquet").set_index("project_uid").split
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet",
                    columns=["uid", "project_uid", "issuetype", "summary", "description"])
N = N[N.uid.isin(set(E.parent_uid) | set(E.child_uid))].set_index("uid")
fase(f"nodos implicados {len(N):,}")

E["ptype"] = E.parent_uid.map(N.issuetype)
E["ctype"] = E.child_uid.map(N.issuetype)
E["split"] = E.parent_uid.map(N.project_uid).map(S)
E["csplit"] = E.child_uid.map(N.project_uid).map(S)
n0 = len(E)
E = E[~E.ptype.map(reactivo) & ~E.ctype.map(reactivo)]
fase(f"sin reactivos {len(E):,} de {n0:,}")
E = E[E.split == E.csplit]
fase(f"mismo split padre/hijo {len(E):,}")

N["texto"] = [texto_issue(s, d, 300) for s, d in zip(N.summary, N.description)]
N["titulo"] = [limpiar(s) for s in N.summary]
E["anchor_en"] = E.parent_uid.map(N.texto)
E["positive"] = E.child_uid.map(N.texto)
E = E[(E.positive.str.len() >= 10) & (E.anchor_en.str.len() >= 10)]
E = E.drop_duplicates(["anchor_en", "positive"])
fase(f"pares finales {len(E):,}")

# ------------------------------------------------------------- translation
from transformers import MarianMTModel, MarianTokenizer  # noqa: E402

padres = E.parent_uid.unique()
titulos = N.loc[padres, "titulo"]
unicos = pd.Series(titulos.unique())
unicos = unicos[unicos.str.len() > 0]
fase(f"traduciendo {len(unicos):,} titulos unicos de padres")
tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", dtype=torch.float16).cuda().eval()
orden = unicos.str.len().sort_values().index      # length-sorted batches: no padding waste
trad = {}
BS = 128
lista = unicos.loc[orden].tolist()
t1 = time.time()
for i in range(0, len(lista), BS):
    lote = lista[i:i + BS]
    x = tok(lote, return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
    with torch.no_grad():
        y = mt.generate(**x, num_beams=1, max_new_tokens=128)
    for a, b in zip(lote, tok.batch_decode(y, skip_special_tokens=True)):
        trad[a] = b
    if (i // BS) % 40 == 0:
        hecho = i + len(lote)
        eta = (len(lista) - hecho) / (hecho / (time.time() - t1))
        print(f"    {hecho:>7,}/{len(lista):,}  ETA {eta/60:4.1f} min", flush=True)
E["anchor_es"] = E.parent_uid.map(N.titulo).map(trad)
fase("traduccion lista")

rng = np.random.default_rng(SEED)
E["usa_es"] = (E.split == "train") & (rng.random(len(E)) < P_ES) & E.anchor_es.notna()
cols = ["split", "relation", "parent_uid", "child_uid", "ptype", "ctype",
        "anchor_en", "anchor_es", "positive", "usa_es"]
E[cols].to_parquet(f"{BASE}/data/processed/sbert_pairs.parquet", index=False)
print(E.groupby(["split", "relation"]).size().unstack().to_string())
print(E.sample(6, random_state=1)[["relation", "anchor_en", "anchor_es", "positive"]].to_string())
fase("fin")
