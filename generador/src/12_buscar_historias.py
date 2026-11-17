"""Do we already own user stories in "As a <role>, I want <goal>, so that <benefit>" form?

Scans all 2.85M issues (streamed in batches so RAM stays flat) for the pattern
in the title and in the description, and reports how many of those issues sit
in the hierarchy (i.e. can be used as story -> tasks training pairs, or as the
target of epic -> stories).
"""
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, r"F:\Software Task generation Model\src")
from textclean import limpiar  # noqa: E402

BASE = r"F:\Software Task generation Model"
PAT = re.compile(r"\bas\s+(?:an?|the)\s+[^,.\n]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                 r"(?:want|would\s+like|need|wish|should|can|must|am\s+able)", re.I)
PAT_SO = re.compile(r"\bso\s+that\b|\bin\s+order\s+to\b", re.I)
t0 = time.time()

E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
padres = set(E.parent_uid)
hijos = set(E.child_uid)
hijos_epic = set(E.loc[E.relation == "epic", "child_uid"])

filas = []
pf = pq.ParquetFile(f"{BASE}/data/interim/nodes.parquet")
n = 0
for lote in pf.iter_batches(batch_size=200_000,
                            columns=["uid", "source", "project_uid", "issuetype", "summary", "description"]):
    d = lote.to_pandas()
    s = d.summary.fillna("")
    desc = d.description.fillna("").str.slice(0, 600)
    en_titulo = s.str.contains(PAT, na=False)
    en_desc = desc.str.contains(PAT, na=False)
    m = en_titulo | en_desc
    if m.any():
        sub = d[m].copy()
        sub["en_titulo"] = en_titulo[m]
        sub["con_so_that"] = (s[m] + " " + desc[m]).str.contains(PAT_SO, na=False)
        sub["texto"] = np.where(sub.en_titulo, s[m], desc[m])
        filas.append(sub[["uid", "source", "project_uid", "issuetype", "en_titulo",
                          "con_so_that", "texto"]])
    n += len(d)
    print(f"  {n:>9,} incidencias  {time.time()-t0:5.0f}s  historias {sum(map(len, filas)):,}", flush=True)

H = pd.concat(filas, ignore_index=True)
H["es_padre"] = H.uid.isin(padres)
H["es_hijo"] = H.uid.isin(hijos)
H["hijo_de_epic"] = H.uid.isin(hijos_epic)
H.to_parquet(f"{BASE}/data/interim/historias_formato.parquet", index=False)

print(f"\nTOTAL con formato de historia: {len(H):,} de {n:,} ({100*len(H)/n:.2f} %)")
print(f"  en el titulo: {H.en_titulo.sum():,}   con 'so that': {H.con_so_that.sum():,}")
print(f"  con hijos (=> historia -> tareas): {H.es_padre.sum():,}")
print(f"  hijas de un epic (=> epic -> historias): {H.hijo_de_epic.sum():,}")
print("\npor fuente:\n", H.groupby("source").agg(n=("uid", "size"), en_titulo=("en_titulo", "sum"),
                                                 padres=("es_padre", "sum"),
                                                 proyectos=("project_uid", "nunique")
                                                 ).sort_values("n", ascending=False).head(15).to_string())
print("\npor tipo:\n", H.issuetype.value_counts().head(12).to_string())
print("\nproyectos con mas historias:\n",
      H.groupby("project_uid").agg(n=("uid", "size"), en_titulo=("en_titulo", "sum"),
                                   padres=("es_padre", "sum")).sort_values("n", ascending=False).head(15).to_string())
print("\nejemplos con titulo en formato:")
for t in H[H.en_titulo].texto.sample(min(8, int(H.en_titulo.sum())), random_state=1):
    print("  -", t[:180])
print("\nejemplos con descripcion en formato (padres con hijos):")
for t in H[(~H.en_titulo) & H.es_padre].texto.sample(8, random_state=1):
    print("  -", limpiar(t)[:180])
H.groupby(["source", "issuetype"]).size().sort_values(ascending=False).to_csv(
    f"{BASE}/results/12_historias_formato.csv")
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
