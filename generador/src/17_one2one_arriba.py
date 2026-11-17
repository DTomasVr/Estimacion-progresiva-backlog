"""Upper half as One2One: (feature description -> ONE item), not one row per
project. That is what makes the data-starved half trainable: 9.7k epics with
100k children become 100k training rows, and Dalpiaz's 22 projects become
~1.6k rows instead of 22.

Two styles, marked in the prefix, so the style can be requested at inference:
  "feature -> issue:"      a normal Jira child (the bulk)
  "feature -> user story:" the child really is written as "As a ..., I want ..."
                           (2.9k real epic children + Dalpiaz stories)
Story-form rows are repeated OVERSAMPLE times in train so they are not drowned
by the majority style (195 examples taught the model nothing last time).

Dalpiaz is split BY PROJECT: 15 train / 3 val / 4 held out for the final test,
which is never trained on and is what the thesis reports.

out: data/processed/s2s_o2o_{train,val,test}.parquet + s2s_o2o_dalpiaz_test.parquet
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch

sys.path.insert(0, r"F:\Software Task generation Model\src")
from textclean import limpiar  # noqa: E402

BASE = r"F:\Software Task generation Model"
SBERT = f"{BASE}/models/sbert_agix_v1"
SEED = 20260922
K_VECINOS = 2
HIJOS_POR_VECINO = 4
OVERSAMPLE = 3
MAX_IN, MAX_OUT = 256, 96
EOS = 1
PREF = {"issue": "feature -> issue: ", "historia": "feature -> user story: "}
t0 = time.time()


def fase(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def primeras_frases(t, n=2, max_chars=250):
    f = re.split(r"(?<=[.!?])\s+", t)
    o = " ".join(f[:n])
    return o if len(o) <= max_chars else o[:o.rfind(" ", 0, max_chars)]


# ------------------------------------------------------------- epic -> child
E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
E = E[E.relation == "epic"]
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet",
                    columns=["uid", "project_uid", "issuetype", "summary", "description"])
N = N[N.uid.isin(set(E.parent_uid) | set(E.child_uid))].set_index("uid")
S = pd.read_parquet(f"{BASE}/data/processed/split_projects.parquet").set_index("project_uid").split
H = set(pd.read_parquet(f"{BASE}/data/interim/historias_formato.parquet").uid)
REACTIVO = ("bug", "suggestion", "question", "support", "incident", "problem", "rca")
E = E[~E.parent_uid.map(N.issuetype).map(lambda t: isinstance(t, str) and any(k in t.lower() for k in REACTIVO))]
E = E[~E.child_uid.map(N.issuetype).map(lambda t: isinstance(t, str) and any(k in t.lower() for k in REACTIVO))]
E["objetivo"] = E.child_uid.map(N.summary).map(limpiar)
E = E[E.objetivo.str.len() >= 10]
E["proj"] = E.parent_uid.map(N.project_uid)
E["split"] = E.proj.map(S)
E["estilo"] = np.where(E.child_uid.isin(H), "historia", "issue")
fase(f"pares epic->hijo: {len(E):,} ({E.estilo.value_counts().to_dict()})")

pad = N.loc[E.parent_uid.unique()]
entrada = {u: (f"{limpiar(s)}. {primeras_frases(limpiar(d))}".strip(". ") if isinstance(d, str) and d
               else limpiar(s)) for u, s, d in zip(pad.index, pad.summary, pad.description)}
E["entrada"] = E.parent_uid.map(entrada)
E = E[E.entrada.str.len() >= 10]
hijos = E.groupby("parent_uid").objetivo.apply(list)

# --------------------------------------------------------------- dalpiaz
DP = pd.read_parquet(f"{BASE}/data/interim/dalpiaz_projects.parquet")
DS = pd.read_parquet(f"{BASE}/data/interim/dalpiaz_stories.parquet").drop_duplicates(["project", "story"])
rng = np.random.default_rng(SEED)
orden = DP.project.sample(frac=1, random_state=SEED).tolist()
dsplit = {p: ("train" if i < 15 else "val" if i < 18 else "test") for i, p in enumerate(orden)}
DP["split"] = DP.project.map(dsplit)
PROCEDENCIA = re.compile(r"licen|github|trello|can be found|dataset|data set|collected|collection|"
                         r"retrieved|extracted|sample|originat|available online|document|record|"
                         r"deliverable|wayback|user stories|we can|the curator", re.I)
DP["entrada"] = [(" ".join([f for f in re.split(r"(?<=[.!?])\s+", t) if not PROCEDENCIA.search(f)][:2])
                  or t[:200]) for t in DP.curator_text]
D = DS.merge(DP[["project", "entrada", "split"]], on="project")
D["objetivo"] = D.story.map(limpiar)
D = D[D.objetivo.str.len() >= 15]
D["estilo"] = "historia"
D["proj"] = "dalpiaz:" + D.project
fase(f"dalpiaz: {len(D):,} pares, proyectos por split {DP.split.value_counts().to_dict()}")

P = pd.concat([E[["entrada", "objetivo", "estilo", "proj", "split", "parent_uid"]],
               D[["entrada", "objetivo", "estilo", "proj", "split"]].assign(parent_uid="dalpiaz:" + D.project)],
              ignore_index=True).drop_duplicates(["entrada", "objetivo"])
fase(f"pares totales {len(P):,}")

# ------------------------------------------------- retrieval-augmented input
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(SBERT, device="cuda")
sb.max_seq_length = 128
U = P.drop_duplicates("parent_uid").reset_index(drop=True)
emb = sb.encode(U.entrada.tolist(), batch_size=256, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False)
fase(f"embeddings de {len(U):,} entradas unicas")

pool = np.where(U.split == "train")[0]
proj_cod = torch.tensor(pd.factorize(U.proj)[0], device="cuda")
vec = {}
for a in range(0, len(U), 4096):
    q = np.arange(a, min(a + 4096, len(U)))
    s = (emb[q] @ emb[pool].T).float()
    s[proj_cod[q][:, None] == proj_cod[pool][None, :]] = -2      # never the same project
    _, i = torch.topk(s, K_VECINOS, dim=1)
    for qi, ii in zip(q, i.cpu().numpy()):
        vec[U.parent_uid[qi]] = [U.parent_uid[pool[x]] for x in ii]

hijos_d = D.groupby("parent_uid").objetivo.apply(list) if "parent_uid" in D else {}
todos_hijos = {**hijos.to_dict(), **{("dalpiaz:" + k): v for k, v in
                                     D.groupby("project").objetivo.apply(list).items()}}
ent = dict(zip(U.parent_uid, U.entrada))


def contexto(uid):
    partes = []
    for v in vec.get(uid, []):
        hs = todos_hijos.get(v, [])[:HIJOS_POR_VECINO]
        if hs:
            partes.append(f"{ent[v][:80]} => {' ; '.join(hs)}")
    return " || ".join(partes)


P["src"] = [PREF[e] + en + " | examples: " + contexto(u)
            for e, en, u in zip(P.estilo, P.entrada, P.parent_uid)]
fase("entradas con ejemplos recuperados")

# ----------------------------------------------------------------- tokenize
sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
P["input_ids"] = [x[:MAX_IN - 1] + [EOS] for x in sp.encode(P.src.tolist())]
P["labels"] = [x[:MAX_OUT - 1] + [EOS] for x in sp.encode(P.objetivo.tolist())]
L = np.array([len(x) for x in P.labels])
print(f"  longitud de salida: p50 {np.quantile(L,.5):.0f}  p90 {np.quantile(L,.9):.0f}  "
      f"p99 {np.quantile(L,.99):.0f}  (tope {MAX_OUT})")

dal_test = P[P.proj.str.startswith("dalpiaz:") & (P.split == "test")]
dal_test.to_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet", index=False)
for split in ["train", "val", "test"]:
    d = P[P.split == split]
    if split == "train":
        d = pd.concat([d] + [d[d.estilo == "historia"]] * (OVERSAMPLE - 1), ignore_index=True)
    d.to_parquet(f"{BASE}/data/processed/s2s_o2o_{split}.parquet", index=False)
    print(f"  {split}: {len(d):,} pares  {d.estilo.value_counts().to_dict()}  "
          f"proyectos {d.proj.nunique():,}")
fase("escrito")
print("\nejemplo historia:\n", P[P.estilo == "historia"].src.iloc[0][:300],
      "\n  ->", P[P.estilo == "historia"].objetivo.iloc[0][:150])
