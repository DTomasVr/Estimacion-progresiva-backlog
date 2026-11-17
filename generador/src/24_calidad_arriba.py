"""Why do the generated stories talk about the neighbours' projects?

Symptom: for a public-spending project the model writes "As a Titanium
Developer...", for DSpace "As a fabric developer...". The retrieved examples
are meant to ground the generation, but with a weak neighbour (cosine ~0.45)
the model copies their vocabulary instead.

Two metrics, both reported for every variant:
  sF1 / precision / coverage  as everywhere else
  fuga  share of generated stories carrying a DISTINCTIVE term that comes from
        the retrieved examples and is absent from the description (distinctive
        = appears in < 0.5 % of the training descriptions, so "user" or "data"
        do not count)

Variants (no retraining):
  actual        k=2 neighbours, ranked by votes
  sin_ejemplos  description only
  umbral_X      neighbours only if cosine >= X, else none
  k4            4 neighbours, to dilute any single project's vocabulary
  relevancia    ranked by votes x similarity to the description
"""
import importlib.util
import re
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("g", f"{BASE}/src/18_generar_historias.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
ev = G.ev
MODELO = f"{BASE}/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1"
SEED = 20260922
N_MUESTRAS, UMBRAL, K = 30, 0.85, 12
sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")


def palabras(t):
    return set(w for w in re.findall(r"[a-z]{4,}", (t or "").lower()))


def main():
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    tok, m = ev.cargar_modelo(MODELO)
    A = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_train.parquet",
                        columns=["parent_uid", "entrada", "objetivo", "proj", "split"])
    A = A.groupby("parent_uid").agg(entrada=("entrada", "first"), proj=("proj", "first"),
                                    hijos=("objetivo", list)).reset_index()
    EA = torch.tensor(np.load(f"{BASE}/data/processed/pipeline_emb.npz")["A"]).cuda()

    # document frequency over training descriptions -> what counts as distinctive
    df = Counter()
    for t in A.entrada:
        df.update(palabras(t))
    n_doc = len(A)
    distintiva = lambda w: df.get(w, 0) / n_doc < 0.005            # noqa: E731

    V = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_val.parquet")
    V = V[V.estilo == "historia"]
    P = G.por_proyecto(V).head(45).reset_index(drop=True)
    refs = list(P.refs)
    q = sb.encode(P.entrada.tolist() if "entrada" in P else
                  [s.split(" | examples: ")[0].split(": ", 1)[1] for s in P.src],
                  convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    desc = [s.split(" | examples: ")[0].split(": ", 1)[1] for s in P.src]
    sims, idx = torch.topk(q @ EA.T, 4, dim=1)
    sims, idx = sims.cpu().numpy(), idx.cpu().numpy()

    def prompt(i, k=2, umbral=0.0):
        partes = []
        for j, s in zip(idx[i][:k], sims[i][:k]):
            if s < umbral:
                continue
            r = A.iloc[int(j)]
            partes.append(f"{r.entrada[:80]} => {' ; '.join(list(r.hijos)[:4])}")
        return f"feature -> user story: {desc[i]} | examples: {' || '.join(partes)}", partes

    filas = []

    def evaluar(nombre, srcs, ctxs, por_relevancia=False):
        ids = [x[:255] + [1] for x in sp.encode(srcs)]
        cands = G.candidatos(tok, m, ids, N_MUESTRAS)
        if por_relevancia:
            gens = []
            for i, c in enumerate(cands):
                conj = G.a_conjunto(sb, [c], UMBRAL, 50)[0]
                if conj:
                    e = sb.encode(conj, convert_to_tensor=True, normalize_embeddings=True,
                                  show_progress_bar=False)
                    rel = (e @ q[i]).cpu().numpy()
                    conj = [t for _, t in sorted(zip(rel, conj), key=lambda x: -x[0])][:K]
                gens.append(conj)
        else:
            gens = G.a_conjunto(sb, cands, UMBRAL, K)
        s = ev.puntuar(sb, gens, refs)
        fugas = []
        for g, c, dsc in zip(gens, ctxs, desc):
            ctx_w = {w for p in c for w in palabras(p) if distintiva(w)} - palabras(dsc)
            fugas.append(np.mean([bool(palabras(t) & ctx_w) for t in g]) if g else 0)
        filas.append(dict(variante=nombre, sF1=s["sF1"], precision=s["sPrec"], cobertura=s["sCob"],
                          fuga=float(np.mean(fugas)), historias=float(np.mean([len(g) for g in gens]))))
        print(f"  {nombre:<14} sF1 {s['sF1']:.4f}  prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  "
              f"fuga {np.mean(fugas):.1%}  historias {filas[-1]['historias']:.1f}", flush=True)
        return gens

    base = [prompt(i) for i in range(len(P))]
    evaluar("actual", [b[0] for b in base], [b[1] for b in base])
    vacio = [(f"feature -> user story: {d} | examples: ", []) for d in desc]
    evaluar("sin_ejemplos", [v[0] for v in vacio], [v[1] for v in vacio])
    for u in (0.5, 0.55):
        pr = [prompt(i, 2, u) for i in range(len(P))]
        evaluar(f"umbral{u}", [x[0] for x in pr], [x[1] for x in pr])
    pr4 = [prompt(i, 4) for i in range(len(P))]
    evaluar("k4", [x[0] for x in pr4], [x[1] for x in pr4])
    evaluar("relevancia", [b[0] for b in base], [b[1] for b in base], por_relevancia=True)

    R = pd.DataFrame(filas).round(4)
    R.to_csv(f"{BASE}/results/24_calidad_arriba.csv", index=False)
    print("\n" + R.sort_values("sF1", ascending=False).to_string(index=False))
    print(f"\nsimilitud del vecino 1: mediana {np.median(sims[:, 0]):.3f}  "
          f"p10 {np.quantile(sims[:, 0], .1):.3f}  p90 {np.quantile(sims[:, 0], .9):.3f}")


if __name__ == "__main__":
    main()
