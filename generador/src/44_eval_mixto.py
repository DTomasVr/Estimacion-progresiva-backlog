"""Phase 5 of the distillation pilot: did training on LLM-written product backlogs
help on consumer domains WITHOUT wrecking the real-data performance?

Three questions, three populations:

  A. regression check -- real test sets (Jira story->tasks, Dalpiaz description->
     stories). If the mixed model loses here, the distillation cost too much.
  B. consumer domains -- the 40 held-out synthetic descriptions. Caveat: their
     references were written by the teacher, so this measures agreement with the
     teacher, not ground truth.
  C. reading material -- full backlogs for descriptions that failed before
     (audiobook store, cleaning app), old model vs new, printed side by side so a
     human can judge what the metrics cannot.

  python src/44_eval_mixto.py [abajo|arriba|leer]
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ABAJO_VIEJO = f"{BASE}/models/mt5-small-pruned_hist_pbe_v1"
ABAJO_NUEVO = f"{BASE}/models/mt5-small-pruned_hist_pbe_v1_mix_pbe_v1"
ARRIBA_VIEJO = f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1"
ARRIBA_NUEVO = f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_mixa_pbe_v1"
N = 300
CFG = ev.CFGS["beam4_actual"]
que = sys.argv[1] if len(sys.argv) > 1 else "abajo"
sb = SentenceTransformer(ev.SBERT, device="cuda")


def pareado(a, b, sb, gens_a, gens_b, refs, etiqueta):
    sa = ev.puntuar(sb, gens_a, refs)
    sbb = ev.puntuar(sb, gens_b, refs)
    d = sbb["mat"][:, 0] - sa["mat"][:, 0]
    rng = np.random.default_rng(ev.SEED)
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    lo, hi = np.quantile(bs, .025), np.quantile(bs, .975)
    veredicto = "MEJORA" if lo > 0 else ("EMPEORA" if hi < 0 else "empate")
    print(f"  {etiqueta:<34} {a}: {sa['sF1']:.4f}  {b}: {sbb['sF1']:.4f}  "
          f"dif {d.mean():+.4f} IC95 [{lo:+.4f}, {hi:+.4f}]  -> {veredicto}")
    return dict(conjunto=etiqueta, viejo=round(sa["sF1"], 4), nuevo=round(sbb["sF1"], 4),
                delta=round(float(d.mean()), 4), ic95_lo=round(float(lo), 4),
                ic95_hi=round(float(hi), 4), veredicto=veredicto, n=len(refs))


def generar_con(ruta, D):
    tok, m = ev.cargar_modelo(ruta)
    g = ev.generar(tok, m, D, CFG, bs=16)
    del m
    import torch
    torch.cuda.empty_cache()
    return g


filas = []
if que == "abajo":
    print("=== A. no romper lo real (historia -> tareas, proyectos de prueba de Jira)")
    T = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_test.parquet")
    T = T[(T.mitad == "tasks") & (T.estilo == "jira")].sample(N, random_state=ev.SEED).reset_index(drop=True)
    refs = [list(x) for x in T.tareas]
    filas.append(pareado("viejo", "nuevo", sb, generar_con(ABAJO_VIEJO, T),
                         generar_con(ABAJO_NUEVO, T), refs, "real: historia -> tareas"))

    print("=== B. dominios de consumo (referencias del profesor, no verdad absoluta)")
    S = pd.read_parquet(f"{BASE}/data/processed/s2s_mix_test.parquet")
    S = S[S.estilo == "sintetico"].reset_index(drop=True)
    if len(S) > N:
        S = S.sample(N, random_state=ev.SEED).reset_index(drop=True)
    refs = [list(x) for x in S.tareas]
    filas.append(pareado("viejo", "nuevo", sb, generar_con(ABAJO_VIEJO, S),
                         generar_con(ABAJO_NUEVO, S), refs, "consumo: historia -> tareas"))

elif que == "arriba":
    print("=== A. no romper lo real (descripcion -> historias)")
    for nombre, ruta in [("real: o2o test", f"{BASE}/data/processed/s2s_o2ofmt_test.parquet"),
                         ("real: Dalpiaz reservado", f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet")]:
        D = pd.read_parquet(ruta)
        if "objetivo" not in D.columns:
            continue
        D = D.sample(min(N, len(D)), random_state=ev.SEED).reset_index(drop=True)
        refs = [[o] for o in D.objetivo]
        filas.append(pareado("viejo", "nuevo", sb, generar_con(ARRIBA_VIEJO, D),
                             generar_con(ARRIBA_NUEVO, D), refs, nombre))
    print("=== B. dominios de consumo (referencias del profesor)")
    S = pd.read_parquet(f"{BASE}/data/processed/s2s_mixa_test.parquet")
    S = S[S.estilo == "sintetico"].reset_index(drop=True)
    S = S.sample(min(N, len(S)), random_state=ev.SEED).reset_index(drop=True)
    if "objetivo" in S.columns and S.objetivo.notna().all():
        refs = [[o] for o in S.objetivo]
    else:
        # the mixed parquet only carries the tokenized target; decode it back
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
        refs = [[sp.decode([int(x) for x in l if int(x) > 1])] for l in S.labels]
    if refs:
        filas.append(pareado("viejo", "nuevo", sb, generar_con(ARRIBA_VIEJO, S),
                             generar_con(ARRIBA_NUEVO, S), refs, "consumo: descripcion -> historia"))

if filas:
    R = pd.DataFrame(filas)
    salida = f"{BASE}/results/44_mixto_{que}.csv"
    R.to_csv(salida, index=False)
    print(f"\nescrito {salida}")
