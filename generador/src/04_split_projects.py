"""Project-level train/val/test split shared by EVERY later stage (SBERT
fine-tune, seq2seq, retrieval index, evaluation). Issue-level splits leak: a
project in train and test hands the model its own tasks.

Per source instance, projects are shuffled (fixed seed) and assigned to test
until ~10 % of that instance's hierarchy edges are covered, then val to ~20 %,
the rest train. A project holding > 25 % of its instance's edges is forced to
train so one giant project can't be the whole test set. Projects without any
hierarchy edge are split 80/10/10 at random (they still matter for retrieval).
"""
import numpy as np
import pandas as pd

BASE = r"F:\Software Task generation Model"
SEED = 20260922

PR = pd.read_parquet(f"{BASE}/data/interim/projects.parquet")
E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet", columns=["uid", "project_uid"])
proj = N.set_index("uid").project_uid
E["project_uid"] = E.parent_uid.map(proj)
aristas = E.groupby("project_uid").size()
PR["aristas"] = PR.project_uid.map(aristas).fillna(0).astype(int)

rng = np.random.default_rng(SEED)
split = {}
for src, g in PR.groupby("source"):
    con = g[g.aristas > 0]
    total = con.aristas.sum()
    grande = con.aristas > 0.25 * total
    for p in con.project_uid[grande]:
        split[p] = "train"
    resto = con[~grande].sample(frac=1, random_state=int(rng.integers(1 << 31)))
    acum = 0
    for p, a in zip(resto.project_uid, resto.aristas):
        split[p] = "test" if acum < 0.10 * total else ("val" if acum < 0.20 * total else "train")
        acum += a
    sin = g[g.aristas == 0].project_uid
    for p, u in zip(sin, rng.random(len(sin))):
        split[p] = "test" if u < 0.1 else ("val" if u < 0.2 else "train")

PR["split"] = PR.project_uid.map(split)
PR[["project_uid", "source", "project_key", "aristas", "n_issues", "split"]].to_parquet(
    f"{BASE}/data/processed/split_projects.parquet", index=False)

t = PR.pivot_table(index="source", columns="split", values="aristas", aggfunc="sum", fill_value=0)
t["test_%"] = (100 * t.test / t.sum(axis=1)).round(1)
print(t.to_string())
r = PR.groupby("split").agg(proyectos=("project_uid", "size"), aristas=("aristas", "sum"),
                            issues=("n_issues", "sum"))
r["aristas_%"] = (100 * r.aristas / r.aristas.sum()).round(1)
print(r.to_string())
r.to_csv(f"{BASE}/results/04_split_resumen.csv")
