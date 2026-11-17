"""E-274 - ¿Son redundantes entre si las tareas de un backlog? PBE frente a los LLM.

El banco de generacion muestra que PBE genera unas 35 tareas por backlog y los LLM
entre 110 y 200. Antes de decir que las de PBE «son pocas pero no se repiten» hay
que medirlo. Para cada backlog de los 64 proyectos retenidos:

  casi duplicadas   proporcion de tareas cuya vecina mas parecida DEL MISMO backlog
                    supera un coseno de 0,90 (y de 0,80, mas laxo)
  parecido medio    coseno medio con la vecina mas parecida
  tareas distintas  numero de grupos al unir las tareas con coseno >= 0,90

Los embeddings son de paraphrase-multilingual-mpnet-base-v2 SIN afinar, y no del
SBERT propio, para que el juez no sea un modelo nuestro. La comparacion con PBE es
pareada por proyecto, con IC95 por remuestreo.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GEN = Path("F:/Software Task generation Model/results/20_banco.jsonl")
OUT = Path("results/redundancia_tareas")
JUEZ = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def grupos(S, umbral):
    n = len(S)
    padre = list(range(n))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i, j in zip(*np.where(np.triu(S >= umbral, 1))):
        padre[raiz(i)] = raiz(j)
    return len({raiz(i) for i in range(n)})


def main():
    from sentence_transformers import SentenceTransformer
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = [json.loads(l) for l in open(GEN, encoding="utf-8")]
    R = [r for r in R if r["sistema"] != "pbe" and r["condicion"] == "cero"]
    for p in sorted(Path("results/banco_generacion").glob("*.jsonl")):
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    filas, textos = [], []
    for r in R:
        t = [x.strip() for h in r["tareas"] for x in h if x and x.strip()]
        if len(t) >= 2:
            filas.append((r["sistema"], r["caso"], len(textos), len(t)))
            textos += t
    print(f"{len(filas)} backlogs · {len(textos):,} tareas · {time.time() - t0:.0f} s", flush=True)
    sb = SentenceTransformer(JUEZ, device="cuda")
    E = sb.encode(textos, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    print(f"embeddings listos · {time.time() - t0:.0f} s", flush=True)
    out = []
    for sis, caso, a, n in filas:
        S = E[a:a + n] @ E[a:a + n].T
        np.fill_diagonal(S, -1)
        v = S.max(1)
        out.append(dict(sistema=sis, caso=caso, tareas=n, dup90=float(np.mean(v >= .90)),
                        dup80=float(np.mean(v >= .80)), parecido=float(v.mean()), distintas=grupos(S, .90)))
    d = pd.DataFrame(out)
    d.to_csv(OUT / "por_backlog.csv", index=False)
    g = d.groupby("sistema").agg(n=("caso", "size"), tareas=("tareas", "mean"), casi_duplicadas_090=("dup90", "mean"),
                                 casi_duplicadas_080=("dup80", "mean"), parecido_medio=("parecido", "mean"),
                                 tareas_distintas=("distintas", "mean")).reset_index()
    rng = np.random.default_rng(274)
    P = d.pivot(index="caso", columns="sistema", values="dup90")
    par = {}
    for s in P.columns:
        if s == "pbe":
            continue
        x = (P[s] - P["pbe"]).dropna().to_numpy()
        b = rng.choice(x, (4000, len(x))).mean(1)
        par[s] = dict(dif_dup90=float(x.mean()), ic95_lo=float(np.quantile(b, .025)), ic95_hi=float(np.quantile(b, .975)))
    g = g.merge(pd.DataFrame(par).T.rename_axis("sistema").reset_index(), on="sistema", how="left")
    g.to_csv(OUT / "resumen.csv", index=False)
    pd.set_option("display.width", 220)
    print(g.round(3).to_string(index=False))
    print(f"escrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
