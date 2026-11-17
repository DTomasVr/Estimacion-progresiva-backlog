"""Preprocessing for the seq2seq generator (One2Seq), both halves:

  lower  "tasks"   : story/task parent  -> its sub-tasks        (relation subtask)
  upper  "stories" : epic (title + 1-2 sentences) -> its children (relation epic)
  dalpiaz          : curator project description -> user stories (upper TEST only)

Phases (each prints its wall time):
  P1 build examples   clean text, drop reactive work, >= 2 children, creation order,
                      within-target dedup
  P2 embeddings       sbert_agix_v1, float16 on disk; near-duplicate inputs
                      (cos >= 0.97, same project, same half) collapsed
  P3 retrieval aug    k=2 neighbours from TRAIN examples of OTHER projects
  P4 spanish inputs   30 % of train inputs use the translated parent title
  P5 tokenize         mT5 tokenizer once; caps from measured quantiles within the
                      VRAM budget; targets cut at whole-task boundaries
  P6 write + report   data/processed/s2s_{train,val,test,dalpiaz}.parquet
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from textclean import limpiar, texto_issue  # noqa: E402

BASE = r"F:\Software Task generation Model"
SBERT = f"{BASE}/models/sbert_agix_v1"
MT5 = "google/mt5-small"
CACHE = f"{BASE}/models/hf_cache"
SEED = 20260922
REACTIVO = ("bug", "suggestion", "question", "support", "incident", "problem", "rca")
SEP = " ; "
K_VECINOS = 2
TAREAS_POR_VECINO = 6
P_ES = 0.30
MAX_IN_TOPE, MAX_OUT_TOPE = 256, 192     # VRAM budget measured in 09_probe (fp32+autocast, bs8 out192 = 6.46 GB)
t0 = time.time()
tiempos = {}


def fase(nombre, inicio):
    tiempos[nombre] = round(time.time() - inicio, 1)
    print(f"[{time.time()-t0:6.1f}s] {nombre} lista en {tiempos[nombre]:.1f}s", flush=True)


def reactivo(t):
    return isinstance(t, str) and any(k in t.lower() for k in REACTIVO)


def primeras_frases(texto, n=2, max_chars=250):
    frases = re.split(r"(?<=[.!?])\s+", texto)
    out = " ".join(frases[:n])
    return out if len(out) <= max_chars else out[:out.rfind(" ", 0, max_chars)]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ============================================================== P1 examples
ti = time.time()
E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
S = pd.read_parquet(f"{BASE}/data/processed/split_projects.parquet").set_index("project_uid").split
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet",
                    columns=["uid", "project_uid", "issuetype", "summary", "description", "created"])
N = N[N.uid.isin(set(E.parent_uid) | set(E.child_uid))].set_index("uid")
E["ptype"] = E.parent_uid.map(N.issuetype)
E["ctype"] = E.child_uid.map(N.issuetype)
E = E[~E.ptype.map(reactivo) & ~E.ctype.map(reactivo)]
E = E[(E.relation == "subtask") | (E.ptype == "Epic")]
E["pproj"] = E.parent_uid.map(N.project_uid)
E = E[E.pproj == E.child_uid.map(N.project_uid)]            # cross-project links would leak splits
E["tarea"] = E.child_uid.map(N.summary).map(limpiar).str.replace(";", ",", regex=False)
E["creado"] = E.child_uid.map(N.created)
E = E[E.tarea.str.len() >= 5].sort_values(["parent_uid", "creado", "child_uid"])
E["tnorm"] = E.tarea.map(norm)
E = E.drop_duplicates(["parent_uid", "relation", "tnorm"])

g = E.groupby(["parent_uid", "relation"]).agg(tareas=("tarea", list), proj=("pproj", "first")).reset_index()
g["n_hijos"] = g.tareas.str.len()
n_antes = g.relation.value_counts().to_dict()
g = g[g.n_hijos >= 2].copy()
g["mitad"] = g.relation.map({"subtask": "tasks", "epic": "stories"})
g["split"] = g.proj.map(S)
P = N.loc[g.parent_uid]
titulo = P.summary.map(limpiar).values
desc = P.description.map(limpiar).values
g["entrada_en"] = [texto_issue(s, d, 300) if r == "subtask"
                   else (f"{t}. {primeras_frases(dd)}".strip(". ") if dd else t)
                   for r, s, d, t, dd in zip(g.relation, P.summary.values, P.description.values, titulo, desc)]
g["titulo_en"] = titulo
g = g[g.entrada_en.str.len() >= 10].reset_index(drop=True)
print(f"  padres con >=1 hijo: {n_antes}  ->  con >=2 hijos y texto: {g.mitad.value_counts().to_dict()}")
fase("P1 ejemplos", ti)

# ============================================================ P2 embeddings
ti = time.time()
from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(SBERT, device="cuda")
sb.max_seq_length = 128
emb = sb.encode(g.entrada_en.tolist(), batch_size=256, convert_to_tensor=True,
                normalize_embeddings=True, show_progress_bar=False).half()
np.save(f"{BASE}/data/processed/s2s_entrada_emb_f16.npy", emb.cpu().numpy())
g[["parent_uid", "relation"]].to_parquet(f"{BASE}/data/processed/s2s_entrada_emb_ids.parquet", index=False)

# near duplicates: same project + same half, cos >= .97 -> keep the one with most children
quitar = set()
for (proj, mitad), idx in g.groupby(["proj", "mitad"]).groups.items():
    idx = np.array(sorted(idx, key=lambda i: -g.n_hijos[i]))
    if len(idx) < 2:
        continue
    x = emb[idx]
    sim = (x @ x.T).float()
    sim.fill_diagonal_(0)
    sim = torch.triu(sim)            # i < j: j is the smaller one (sorted by n_hijos desc)
    dup = (sim >= 0.97).any(0).cpu().numpy()
    quitar.update(idx[dup].tolist())
print(f"  casi-duplicados quitados: {len(quitar):,} "
      f"({g.loc[list(quitar)].mitad.value_counts().to_dict() if quitar else {}})")
keep = np.array([i not in quitar for i in range(len(g))])
g = g[keep].reset_index(drop=True)
emb = emb[torch.from_numpy(keep).cuda()]
fase("P2 embeddings+dedup", ti)

# ======================================================= P3 retrieval aug
ti = time.time()
proj_cod = torch.tensor(pd.factorize(g.proj)[0], device="cuda")
g["vecinos"] = [[] for _ in range(len(g))]
g["vecinos_sim"] = [[] for _ in range(len(g))]
for mitad in ["tasks", "stories"]:
    pool = np.where((g.mitad == mitad) & (g.split == "train"))[0]
    todos = np.where(g.mitad == mitad)[0]
    P_ = emb[pool]
    pp = proj_cod[pool]
    for a in range(0, len(todos), 4096):
        q = todos[a:a + 4096]
        s = (emb[q] @ P_.T).float()
        s[proj_cod[q][:, None] == pp[None, :]] = -2            # never a neighbour from the same project
        v, i = torch.topk(s, K_VECINOS, dim=1)
        for qi, vi, ii in zip(q, v.cpu().numpy(), i.cpu().numpy()):
            g.at[qi, "vecinos"] = pool[ii].tolist()
            g.at[qi, "vecinos_sim"] = vi.round(3).tolist()


def ejemplo(j):
    r = g.loc[j]
    return f"{r.titulo_en} => {SEP.join(r.tareas[:TAREAS_POR_VECINO])}"


g["contexto"] = [" || ".join(ejemplo(j) for j in vs) for vs in g.vecinos]
fase("P3 recuperacion", ti)

# ======================================================== P4 spanish inputs
ti = time.time()
SP = pd.read_parquet(f"{BASE}/data/processed/sbert_pairs.parquet", columns=["parent_uid", "anchor_es"])
es = SP.dropna().drop_duplicates("parent_uid").set_index("parent_uid").anchor_es
g["titulo_es"] = g.parent_uid.map(es)
rng = np.random.default_rng(SEED)
g["usa_es"] = (g.split == "train") & g.titulo_es.notna() & (rng.random(len(g)) < P_ES)
g["entrada"] = np.where(g.usa_es, g.titulo_es, g.entrada_en)
print(f"  cobertura de traduccion: {g.titulo_es.notna().mean():.1%}; train con entrada es: "
      f"{g[g.split=='train'].usa_es.mean():.1%}")
fase("P4 espanol", ti)


def fuente(mitad, entrada, contexto):
    tarea = "decompose story into tasks" if mitad == "tasks" else "decompose feature into user stories"
    return f"{tarea}: {entrada} | examples: {contexto}"


g["src"] = [fuente(m, e, c) for m, e, c in zip(g.mitad, g.entrada, g.contexto)]

# ========================================================= Dalpiaz (test)
DP = pd.read_parquet(f"{BASE}/data/interim/dalpiaz_projects.parquet")
DS = pd.read_parquet(f"{BASE}/data/interim/dalpiaz_stories.parquet")
PROCEDENCIA = re.compile(r"licen|github|trello|can be found|dataset|data set|collected|collection|"
                         r"retrieved|extracted|sample|originat|available online|document|record|"
                         r"deliverable|wayback|user stories|we can|the curator", re.I)


def descripcion_sistema(t):
    """curator paragraph -> the 1-2 sentences about the system (rule, no rewriting)"""
    frases = [f for f in re.split(r"(?<=[.!?])\s+", t) if f]
    buenas = [f for f in frases if not PROCEDENCIA.search(f)] or frases[:1]
    return " ".join(buenas[:2])


DP["descripcion"] = DP.curator_text.map(descripcion_sistema)
st = DS.drop_duplicates(["project", "story"]).groupby("project").story.apply(list)
DP["tareas"] = DP.project.map(st)
qd = sb.encode(DP.descripcion.tolist(), convert_to_tensor=True, normalize_embeddings=True).half()
pool = np.where((g.mitad == "stories") & (g.split == "train"))[0]
v, i = torch.topk((qd @ emb[pool].T).float(), K_VECINOS, dim=1)
DP["vecinos"] = [pool[x].tolist() for x in i.cpu().numpy()]
DP["contexto"] = [" || ".join(ejemplo(j) for j in vs) for vs in DP.vecinos]
DP["src"] = [fuente("stories", d, c) for d, c in zip(DP.descripcion, DP.contexto)]
DP["mitad"], DP["split"], DP["relation"] = "stories", "dalpiaz", "dalpiaz"
del sb
torch.cuda.empty_cache()

# ============================================================ P5 tokenize
ti = time.time()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MT5, cache_dir=CACHE)
src_ids = tok(g.src.tolist(), add_special_tokens=True).input_ids
L_in = np.array([len(x) for x in src_ids])
tareas_tok = [tok(t, add_special_tokens=False).input_ids for t in g.tareas]
sep_ids = tok(SEP.strip(), add_special_tokens=False).input_ids
L_out_full = np.array([sum(len(x) for x in tt) + len(sep_ids) * (len(tt) - 1) + 1 for tt in tareas_tok])
q = {f"p{p}": (int(np.quantile(L_in, p / 100)), int(np.quantile(L_out_full, p / 100))) for p in (50, 90, 95, 99)}
print(f"  longitudes (entrada, salida) por cuantil: {q}")
MAX_IN = min(MAX_IN_TOPE, int(np.quantile(L_in, .95)))
MAX_OUT = min(MAX_OUT_TOPE, int(np.quantile(L_out_full, .95)))
print(f"  topes elegidos: entrada {MAX_IN}, salida {MAX_OUT}")


def objetivo(tt):
    ids, n = [], 0
    for k, t in enumerate(tt):
        extra = (sep_ids if k else []) + t
        if len(ids) + len(extra) + 1 > MAX_OUT:
            break
        ids += extra
        n += 1
    return ids + [tok.eos_token_id], n


obj = [objetivo(tt) for tt in tareas_tok]
g["input_ids"] = [x[:MAX_IN - 1] + [tok.eos_token_id] if len(x) > MAX_IN else x for x in src_ids]
g["labels"] = [o[0] for o in obj]
g["n_tareas_obj"] = [o[1] for o in obj]
g["entrada_truncada"] = L_in > MAX_IN
g["tareas_cortadas"] = g.n_hijos - g.n_tareas_obj
g = g[g.n_tareas_obj >= 1]
DP["input_ids"] = [x[:MAX_IN - 1] + [tok.eos_token_id] if len(x) > MAX_IN else x
                   for x in tok(DP.src.tolist()).input_ids]
fase("P5 tokenizacion", ti)

# ======================================================== P6 write + report
ti = time.time()
cols = ["split", "mitad", "relation", "parent_uid", "proj", "entrada_en", "titulo_es", "usa_es",
        "src", "tareas", "n_hijos", "n_tareas_obj", "vecinos_sim", "input_ids", "labels",
        "entrada_truncada", "tareas_cortadas"]
for sp in ["train", "val", "test"]:
    g[g.split == sp][cols].to_parquet(f"{BASE}/data/processed/s2s_{sp}.parquet", index=False)
DP[["split", "mitad", "project", "category", "descripcion", "src", "tareas", "input_ids"]].to_parquet(
    f"{BASE}/data/processed/s2s_dalpiaz.parquet", index=False)

rep = g.groupby(["split", "mitad"]).agg(ejemplos=("src", "size"), proyectos=("proj", "nunique"),
                                        hijos_med=("n_hijos", "median"),
                                        entrada_trunc=("entrada_truncada", "mean"),
                                        con_tareas_cortadas=("tareas_cortadas", lambda s: (s > 0).mean()),
                                        sim_vecino_med=("vecinos_sim", lambda s: np.median([x[0] for x in s])))
print(rep.round(3).to_string())
rep.to_csv(f"{BASE}/results/08_preprocess_resumen.csv")

m = g[g.split == "train"].sample(20, random_state=SEED)
with open(f"{BASE}/results/08_muestra_20.txt", "w", encoding="utf-8") as fh:
    for _, r in m.iterrows():
        fh.write(f"=== {r.mitad} | {r.parent_uid} | es={r.usa_es} | sim={r.vecinos_sim}\n"
                 f"SRC: {r.src}\nTGT: {tok.decode(r.labels, skip_special_tokens=True)}\n\n")
fase("P6 escritura", ti)
json.dump(dict(tiempos=tiempos, max_in=MAX_IN, max_out=MAX_OUT, cuantiles=q, sep=SEP,
               k_vecinos=K_VECINOS, p_es=P_ES, total_s=round(time.time() - t0)),
          open(f"{BASE}/results/08_preprocess.json", "w"), indent=1)
print(f"TOTAL {(time.time()-t0)/60:.1f} min")
