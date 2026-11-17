"""Frozen, blinded sample for the human study.

Everything is already generated: results/20_banco.jsonl holds, for the SAME 28
held-out projects, the backlog produced by our pipeline and by three local LLMs.
So the study costs no new generation -- only sampling, blinding and translation.

Two rating tasks come out of this file:
  items    individual user stories, balanced across systems, shuffled: is it
           plausible for this project, would you keep it in the backlog?
  conjuntos  per project, the three backlogs side by side as A/B/C: which is
           best overall, which repeats itself most

Everything the raters see is in Spanish (that is the language of PBE and of
the people rating), translated with the same protected-term translation the
product uses. The A/B/C -> system key is written to a SEPARATE file so the
ratings stay blind until analysis.

out: results/30_estudio_items.json, 30_estudio_conjuntos.json, 30_estudio_clave.json
"""
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
SEED = 20260924
N_PROY = 6
N_HIST_CONJ = 8          # stories shown per backlog in the set-level task
N_ITEMS = 60             # individual stories to rate
SISTEMAS = ["pbe", "llama3.1:8b", "mistral:7b"]
rng = np.random.default_rng(SEED)

R = pd.DataFrame([json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")])
R = R[(R.sistema == "pbe") | (R.condicion == "ejemplos")]      # the LLMs' strongest condition
R = R[R.sistema.isin(SISTEMAS)]
completos = R.groupby("caso").sistema.nunique()
casos = completos[completos == len(SISTEMAS)].index
dal = [c for c in casos if c.startswith("dalpiaz")]
jira = [c for c in casos if not c.startswith("dalpiaz")]
# one epic per project: two epics of the same project would probe the same domain twice
vistos, jira_div = set(), []
for c in rng.permutation(jira):
    proj = str(c).rsplit("-", 1)[0]
    if proj not in vistos:
        vistos.add(proj)
        jira_div.append(str(c))
elegidos = [str(x) for x in rng.choice(dal, min(3, len(dal)), replace=False)] + \
           jira_div[:N_PROY - min(3, len(dal))]
print(f"proyectos elegidos: {elegidos}")

# --- descriptions of those projects, to show the rater what the backlog is for
O = pd.concat([pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet"),
               pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")])
desc = O.groupby("parent_uid").entrada.first().to_dict()
refs = O[O.estilo == "historia"].groupby("parent_uid").objetivo.apply(list).to_dict()

p = pl.Pipeline.__new__(pl.Pipeline)          # translation only, no generators
from transformers import MarianMTModel, MarianTokenizer  # noqa: E402
tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", dtype=torch.float16).cuda().eval()


def a_es(textos, bs=64):
    out = []
    for a in range(0, len(textos), bs):
        lote = textos[a:a + bs]
        prot = [pl.proteger(t) for t in lote]
        x = tok([q for q, _ in prot], return_tensors="pt", padding=True, truncation=True,
                max_length=160).to("cuda")
        with torch.no_grad():
            y = mt.generate(**x, num_beams=1, max_new_tokens=160)
        out += [pl.restaurar(s, f) for s, (_, f) in
                zip(tok.batch_decode(y, skip_special_tokens=True), prot)]
    return out


# ---------------------------------------------------------------- set-level
conjuntos, clave, items = [], {}, []
for caso in elegidos:
    sub = R[R.caso == caso].set_index("sistema")
    letras = list("ABC")
    rng.shuffle(letras)
    asignacion = dict(zip(SISTEMAS, letras))
    clave[caso] = {v: k for k, v in asignacion.items()}
    backlogs = []
    for sis in SISTEMAS:
        r = sub.loc[sis]
        hist = list(r.historias)[:N_HIST_CONJ]
        tar = [list(t)[:4] for t in list(r.tareas)[:N_HIST_CONJ]]
        backlogs.append(dict(letra=asignacion[sis], historias=hist, tareas=tar))
        for h, tt in zip(hist, tar):
            items.append(dict(caso=caso, sistema=sis, historia=h, tareas=tt))
    conjuntos.append(dict(caso=caso, descripcion=desc.get(caso, ""), backlogs=backlogs,
                          n_ref=len(refs.get(caso, []))))

# ---------------------------------------------------------------- item-level
I = pd.DataFrame(items).drop_duplicates(["caso", "historia"])
por_sis = min(N_ITEMS // len(SISTEMAS), I.groupby("sistema").size().min())
# pandas 3 drops the grouping column inside apply, so sample group by group instead
I = pd.concat([g.sample(por_sis, random_state=SEED) for _, g in I.groupby("sistema")])
I = I.sample(frac=1, random_state=SEED).reset_index(drop=True)
I["id"] = [f"i{n:03d}" for n in range(len(I))]
print(f"items a valorar: {len(I)} ({I.sistema.value_counts().to_dict()})")

# ---------------------------------------------------------------- translate
todo = list(I.historia) + [t for tt in I.tareas for t in tt] + \
       [h for c in conjuntos for b in c["backlogs"] for h in b["historias"]] + \
       [c["descripcion"] for c in conjuntos]
unicos = list(dict.fromkeys(todo))
print(f"traduciendo {len(unicos)} textos unicos al espanol...", flush=True)
mapa = dict(zip(unicos, a_es(unicos)))

I["historia_es"] = I.historia.map(mapa)
I["tareas_es"] = [[mapa.get(t, t) for t in tt] for tt in I.tareas]
for c in conjuntos:
    c["descripcion_es"] = mapa.get(c["descripcion"], c["descripcion"])
    for b in c["backlogs"]:
        b["historias_es"] = [mapa.get(h, h) for h in b["historias"]]

json.dump(I.drop(columns=["sistema"]).to_dict("records"),
          open(f"{BASE}/results/30_estudio_items.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump([{k: v for k, v in c.items()} for c in conjuntos],
          open(f"{BASE}/results/30_estudio_conjuntos.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(dict(items={r.id: r.sistema for r in I.itertuples()}, conjuntos=clave, semilla=SEED),
          open(f"{BASE}/results/30_estudio_clave.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"escrito: {len(I)} items, {len(conjuntos)} conjuntos; la clave A/B/C va aparte")
print("\nejemplo de item:")
r = I.iloc[0]
print(f"  {r.id}: {r.historia_es[:140]}\n    tareas: {'; '.join(r.tareas_es[:3])[:160]}")
