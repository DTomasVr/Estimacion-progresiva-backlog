"""Build the story-aware training set.

lower half: every story->tasks example is kept as it is, and the ones whose
input was rewritten as a user story (14_reescribir_historias.py, mistral) are
ADDED as a second example with the SAME real tasks as target. The model then
sees both input styles mapping to the same output instead of only Jira titles.

upper half: examples whose children are actually written as user stories get a
distinct prefix ("... (story form)"), so the style can be requested at
inference without ever inventing a target -- every target stays a real,
human-written issue.

Retrieved examples inside the prompt stay Jira-style: at inference they come
from the real corpus too.

out: data/processed/s2s_hist_{train,val,test}.parquet  (+ column `estilo`)
"""
import re

import pandas as pd
import sentencepiece as spm

BASE = r"F:\Software Task generation Model"
MAX_IN, MAX_OUT = 256, 192
EOS = 1
PREF_TAREAS = "decompose story into tasks: "
PREF_HIST = "decompose feature into user stories: "
PREF_HIST_FMT = "decompose feature into user stories (story form): "

sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
sep = sp.encode(";")
R = pd.read_parquet(f"{BASE}/data/processed/historias_reescritas_mistral.parquet")
hist = dict(zip(R.parent_uid, R.historia))
H = pd.read_parquet(f"{BASE}/data/interim/historias_formato.parquet")
E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
E = E[E.relation == "epic"]
E["fmt"] = E.child_uid.isin(set(H.uid))
frac = E.groupby("parent_uid").fmt.mean()
epics_fmt = set(frac[frac >= 0.5].index)


def objetivo(tareas):
    ids = []
    for k, t in enumerate(tareas):
        extra = (sep if k else []) + sp.encode(t)
        if len(ids) + len(extra) + 1 > MAX_OUT:
            break
        ids += extra
    return ids + [EOS]


def entrada(src):
    return [x[:MAX_IN - 1] + [EOS] for x in sp.encode(src)]


resumen = []
for split in ["train", "val", "test"]:
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_{split}.parquet")
    d["estilo"] = "jira"
    # upper half: mark the ones whose children really are user stories
    es_fmt = (d.mitad == "stories") & d.parent_uid.isin(epics_fmt)
    d.loc[es_fmt, "src"] = [PREF_HIST_FMT + s[len(PREF_HIST):]
                            for s in d.src[es_fmt]]
    d.loc[es_fmt, "estilo"] = "historia_real"

    # lower half: add the rewritten-input copy, same real target
    tiene = (d.mitad == "tasks") & d.parent_uid.isin(hist)
    extra = d[tiene].copy()
    ejemplos = [s.split(" | examples: ", 1) for s in extra.src]
    extra["src"] = [PREF_TAREAS + hist[u] + (" | examples: " + p[1] if len(p) > 1 else "")
                    for u, p in zip(extra.parent_uid, ejemplos)]
    extra["estilo"] = "historia_reescrita"
    d = pd.concat([d, extra], ignore_index=True)

    d["input_ids"] = entrada(d.src.tolist())
    d["labels"] = [objetivo(list(t)) for t in d.tareas]
    d.to_parquet(f"{BASE}/data/processed/s2s_hist_{split}.parquet", index=False)
    resumen.append(d.groupby(["mitad", "estilo"]).size().rename(split))

print(pd.concat(resumen, axis=1).fillna(0).astype(int).to_string())
d = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_train.parquet")
for e in ["historia_reescrita", "historia_real"]:
    s = d[d.estilo == e]
    if len(s):
        r = s.iloc[0]
        print(f"\n[{e}] SRC: {r.src[:230]}\n          TGT: {' ; '.join(r.tareas[:4])[:200]}")
