"""E-274 - Are the tasks of a backlog redundant among themselves? PBE against the LLMs.

The generation bench shows that PBE generates about 35 tasks per backlog and the LLMs
between 110 and 200. Before saying that PBE's tasks "are few but do not repeat" it has
to be measured. For each backlog of the 64 held-out projects:

  near_duplicates   share of tasks whose most similar neighbour IN THE SAME backlog
                    exceeds a cosine of 0,90 (and of 0,80, looser)
  mean_similarity   mean cosine with the most similar neighbour
  distinct_tasks    number of groups when tasks with cosine >= 0,90 are joined

The embeddings are from paraphrase-multilingual-mpnet-base-v2 NOT fine-tuned, not from
our own SBERT, so that the judge is not one of our models. The comparison with PBE is
paired by project, with a bootstrap CI95.

The raw JSONL runs keep their field names (sistema, condicion, caso, tareas).
Outputs: results/redundancia_tareas/por_backlog.csv and resumen.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GEN = Path("F:/Software Task generation Model/results/20_banco.jsonl")
OUT = Path("results/redundancia_tareas")
JUDGE = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def groups(S, threshold):
    n = len(S)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(S >= threshold, 1))):
        parent[root(i)] = root(j)
    return len({root(i) for i in range(n)})


def main():
    from sentence_transformers import SentenceTransformer
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    R = [r for r in R if r["sistema"] != "pbe" and r["condicion"] == "cero"]
    for p in sorted(Path("results/banco_generacion").glob("*.jsonl")):
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    rows, texts = [], []
    for r in R:
        t = [x.strip() for h in r["tareas"] for x in h if x and x.strip()]
        if len(t) >= 2:
            rows.append((r["sistema"], r["caso"], len(texts), len(t)))
            texts += t
    print(f"{len(rows)} backlogs · {len(texts):,} tasks · {time.time() - t0:.0f} s", flush=True)
    sb = SentenceTransformer(JUDGE, device="cuda")
    E = sb.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    print(f"embeddings ready · {time.time() - t0:.0f} s", flush=True)
    out = []
    for sys_, case, a, n in rows:
        S = E[a:a + n] @ E[a:a + n].T
        np.fill_diagonal(S, -1)
        v = S.max(1)
        out.append(dict(system=sys_, case=case, tasks=n, dup90=float(np.mean(v >= .90)),
                        dup80=float(np.mean(v >= .80)), similarity=float(v.mean()), distinct=groups(S, .90)))
    d = pd.DataFrame(out)
    d.to_csv(OUT / "por_backlog.csv", index=False)
    g = d.groupby("system").agg(n=("case", "size"), tasks=("tasks", "mean"), near_duplicates_090=("dup90", "mean"),
                                near_duplicates_080=("dup80", "mean"), mean_similarity=("similarity", "mean"),
                                distinct_tasks=("distinct", "mean")).reset_index()
    rng = np.random.default_rng(274)
    P = d.pivot(index="case", columns="system", values="dup90")
    pair = {}
    for s in P.columns:
        if s == "pbe":
            continue
        x = (P[s] - P["pbe"]).dropna().to_numpy()
        b = rng.choice(x, (4000, len(x))).mean(1)
        pair[s] = dict(diff_dup90=float(x.mean()), ci95_lo=float(np.quantile(b, .025)), ci95_hi=float(np.quantile(b, .975)))
    g = g.merge(pd.DataFrame(pair).T.rename_axis("system").reset_index(), on="system", how="left")
    g.to_csv(OUT / "resumen.csv", index=False)
    pd.set_option("display.width", 220)
    print(g.round(3).to_string(index=False))
    print(f"written {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
