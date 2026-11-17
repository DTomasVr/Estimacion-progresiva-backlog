"""Did the Spanish inputs help the upper half, and did English suffer?

Compares the model before (English-only inputs) and after (half the training
inputs translated) on the SAME test projects, in both languages:
  en  the original English description
  es  its opus-mt translation, which is the shape Agix actually sends

Reports sF1 / precision / coverage, the share of output in user-story form, and
the leakage metric from 24_calidad_arriba.py. Prompts carry no retrieved
examples, per that same experiment.

usage: python src/26_eval_es.py MODELO_ANTES MODELO_DESPUES
"""
import importlib.util
import sys
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("q", f"{BASE}/src/24_calidad_arriba.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
G, ev = Q.G, Q.ev
SEED = 20260922
K, N_MUESTRAS, UMBRAL = 12, 30, 0.85
sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")


def casos():
    T = pd.read_parquet(f"{BASE}/data/processed/s2s_o2oes_test.parquet")
    T = T[(T.estilo == "historia") & (T.idioma == "en")]
    g = T.groupby("parent_uid").agg(entrada=("entrada", "first"), refs=("objetivo", list),
                                    proj=("proj", "first")).reset_index()
    g = g[g.refs.str.len() >= 3]
    jira = g[~g.proj.str.startswith("dalpiaz")].sample(60, random_state=SEED)
    dal = g[g.proj.str.startswith("dalpiaz")]
    tr = pd.read_parquet(f"{BASE}/data/processed/traducciones_arriba.parquet")
    es = dict(zip(tr.parent_uid, tr.entrada_es))
    out = pd.concat([jira.assign(fuente="jira"), dal.assign(fuente="dalpiaz")], ignore_index=True)
    out["entrada_es"] = out.parent_uid.map(es)
    return out[out.entrada_es.notna()].reset_index(drop=True)


def main():
    antes, despues = sys.argv[1], sys.argv[2]
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    C = casos()
    print(f"{len(C)} proyectos ({C.fuente.value_counts().to_dict()})", flush=True)
    A = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_train.parquet", columns=["parent_uid", "entrada"])
    df = Counter()
    for t in A.drop_duplicates("parent_uid").entrada:
        df.update(Q.palabras(t))
    nd = A.parent_uid.nunique()

    filas, por_ej = [], {}
    for etq, ruta in [("antes", antes), ("despues", despues)]:
        tok, m = ev.cargar_modelo(ruta)
        for idioma in ["en", "es"]:
            desc = list(C.entrada if idioma == "en" else C.entrada_es)
            q = sb.encode(desc, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            ids = [x[:255] + [1] for x in sp.encode([f"feature -> user story: {d} | examples: " for d in desc])]
            gens = []
            for i, c in enumerate(G.candidatos(tok, m, ids, N_MUESTRAS)):
                conj = G.a_conjunto(sb, [c], UMBRAL, 50)[0]
                if conj:
                    e = sb.encode(conj, convert_to_tensor=True, normalize_embeddings=True,
                                  show_progress_bar=False)
                    conj = [t for _, t in sorted(zip((e @ q[i]).cpu().numpy(), conj), key=lambda x: -x[0])][:K]
                gens.append(conj)
            s = ev.puntuar(sb, gens, list(C.refs))
            fmt = np.mean([np.mean([bool(G.PAT.match(t)) for t in g]) if g else 0 for g in gens])
            por_ej[(etq, idioma)] = s["mat"][:, 0]
            filas.append(dict(modelo=etq, idioma=idioma, sF1=s["sF1"], precision=s["sPrec"],
                              cobertura=s["sCob"], formato=fmt,
                              historias=float(np.mean([len(g) for g in gens]))))
            print(f"  {etq:<8} {idioma}  sF1 {s['sF1']:.4f}  prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  "
                  f"formato {fmt:.0%}", flush=True)
            if etq == "despues" and idioma == "es":
                with open(f"{BASE}/results/26_muestras_es.txt", "w", encoding="utf-8") as fh:
                    for (_, r), g in zip(C.iterrows(), gens):
                        fh.write(f"=== {r.parent_uid}\nES: {r.entrada_es[:200]}\n"
                                 f"REF: {' ; '.join(r.refs[:4])}\nGEN: {' ; '.join(g[:6])}\n\n")
        del m
        torch.cuda.empty_cache()

    R = pd.DataFrame(filas).round(4)
    print("\n" + R.to_string(index=False))
    R.to_csv(f"{BASE}/results/26_eval_es.csv", index=False)
    rng = np.random.default_rng(SEED)
    for idioma in ["en", "es"]:
        d = por_ej[("despues", idioma)] - por_ej[("antes", idioma)]
        bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        print(f"  {idioma}: despues - antes {d.mean():+.4f}  "
              f"IC95 [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")


if __name__ == "__main__":
    main()
