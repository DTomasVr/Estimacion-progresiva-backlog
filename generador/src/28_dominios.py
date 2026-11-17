"""Group the projects into domains, so a whole domain can be held out.

The thesis has to answer "this only works for the kinds of projects it was
trained on". Splitting by project already measures new projects; this measures
new KINDS of project. Domains are k-means clusters over project vectors (mean
SBERT embedding of the project's own issue text), not the Jira instance, which
would only say "another company".

out: data/processed/dominios.parquet  (project_uid, dominio)
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

BASE = r"F:\Software Task generation Model"
K = 6
SEED = 20260922

H = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_train.parquet",
                    columns=["proj", "entrada_en", "mitad"])
V = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_val.parquet", columns=["proj", "entrada_en", "mitad"])
T = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_test.parquet", columns=["proj", "entrada_en", "mitad"])
D = pd.concat([H, V, T], ignore_index=True)
D = D[D.mitad == "tasks"].drop_duplicates(["proj", "entrada_en"])
print(f"{D.proj.nunique():,} proyectos, {len(D):,} textos")

sb = SentenceTransformer(f"{BASE}/models/sbert_pbe_v1", device="cuda")
sb.max_seq_length = 128
E = sb.encode(D.entrada_en.tolist(), batch_size=256, convert_to_tensor=True,
              normalize_embeddings=True, show_progress_bar=False)
P = pd.DataFrame({"proj": D.proj.values})
idx = P.groupby("proj").indices
proys = sorted(idx)
M = torch.stack([E[list(idx[p])].mean(0) for p in proys])
M = torch.nn.functional.normalize(M, dim=1).cpu().numpy()

km = KMeans(K, random_state=SEED, n_init=10).fit(M)
dom = pd.DataFrame({"project_uid": proys, "dominio": km.labels_})
tam = D.groupby("proj").size().rename("textos")
dom["textos"] = dom.project_uid.map(tam)

# name each cluster by the words that are distinctive of it
import re  # noqa: E402
from collections import Counter  # noqa: E402
palabras = D.assign(dominio=D.proj.map(dict(zip(dom.project_uid, dom.dominio))))
glob = Counter()
por_dom = {}
for d, g in palabras.groupby("dominio"):
    c = Counter(w for t in g.entrada_en for w in re.findall(r"[a-z]{4,}", t.lower()))
    por_dom[d] = c
    glob.update(c)
for d in sorted(por_dom):
    c = por_dom[d]
    n = sum(c.values())
    marca = sorted(((v / n) / (glob[w] / sum(glob.values())), w) for w, v in c.items() if v >= 40)[-8:]
    fila = dom[dom.dominio == d]
    print(f"  dominio {d}: {len(fila):>3} proyectos, {fila.textos.sum():>6,} textos  "
          f"| {', '.join(w for _, w in reversed(marca))}")
    print(f"      ejemplos: {', '.join(fila.sort_values('textos', ascending=False).project_uid.head(5))}")

dom.to_parquet(f"{BASE}/data/processed/dominios.parquet", index=False)
print(f"\nescrito; tamanos: {dom.groupby('dominio').textos.sum().to_dict()}")
