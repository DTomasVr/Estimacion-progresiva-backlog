"""What exactly do the two lower-half models contribute, and where do they differ?

The ensemble was adopted because the union scores better (23_cobertura.py), but
nothing said WHY. This measures it on the same test examples for both sides:

  * each model alone: sF1, precision, coverage, tasks produced
  * overlap: how much of model 2's output is a near-duplicate of model 1's
  * the tasks only model 2 proposes: are they useful (close to a real task) or noise?
  * where each one wins, split by how many real tasks the story had

Also checks the shipped pipeline: inside one story, the two models' outputs are
only de-duplicated by exact text, while across stories the pipeline collapses at
cosine 0.90. This measures how many near-duplicates that leaves in the bench output.

out: results/38_dos_modelos.csv  (+ prints)
"""
import importlib.util
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

M1 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1"                    # bs16 lr1.4e-3
M2 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1_bs8_lr9.2e-4"       # bs8  lr9.2e-4
N = 300
UMBRAL_DUP = 0.90      # same threshold the pipeline uses
UMBRAL_UTIL = 0.60     # "this generated task resembles a real one"
CFG = ev.CFGS["beam4_actual"]

T = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_test.parquet")
T = T[T.mitad == "tasks"].sample(N, random_state=ev.SEED).reset_index(drop=True)
refs = [list(x) for x in T.tareas]
print(f"{len(T)} ejemplos de prueba, {np.mean([len(r) for r in refs]):.1f} tareas reales de media")

sb = SentenceTransformer(ev.SBERT, device="cuda")


def generar(ruta):
    tok, m = ev.cargar_modelo(ruta)
    g = ev.generar(tok, m, T, CFG, bs=16)
    del m
    torch.cuda.empty_cache()
    return g


g1, g2 = generar(M1), generar(M2)
print(f"VRAM pico {torch.cuda.max_memory_allocated()/2**30:.2f} GB")


def unir(a, b, umbral=UMBRAL_DUP):
    """semantic union, first-seen order -- what 23_cobertura measured"""
    out = []
    for x, y in zip(a, b):
        plano = list(x) + list(y)
        if not plano:
            out.append([])
            continue
        E = sb.encode(plano, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        S = (E @ E.T).float().cpu().numpy()
        vistos, sel = np.zeros(len(plano), bool), []
        for i in range(len(plano)):
            if vistos[i]:
                continue
            vistos |= (S[i] >= umbral)
            sel.append(plano[i])
        out.append(sel)
    return out


filas = []
for nombre, gen in [("modelo 1 (bs16, lr1.4e-3)", g1), ("modelo 2 (bs8, lr9.2e-4)", g2),
                    ("union semantica 0.90", unir(g1, g2)),
                    ("concatenacion sin dedup", [list(a) + list(b) for a, b in zip(g1, g2)])]:
    s = ev.puntuar(sb, gen, refs)
    filas.append(dict(sistema=nombre, sF1=round(s["sF1"], 4), precision=round(s["sPrec"], 4),
                      cobertura=round(s["sCob"], 4), tareas=round(np.mean([len(x) for x in gen]), 2)))
R = pd.DataFrame(filas)
print("\n=== cada modelo por separado y unidos (mismos 300 ejemplos)")
print(R.to_string(index=False))

# --------------------------------------------------- overlap and contribution
dup, nuevas, nuevas_utiles, solo1, solo2 = [], [], [], [], []
for x, y, r in zip(g1, g2, refs):
    if not x or not y:
        continue
    Ex = sb.encode(list(x), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    Ey = sb.encode(list(y), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    Er = sb.encode(list(r), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    S = (Ey @ Ex.T).float().cpu().numpy()          # each task of m2 against all of m1
    es_dup = S.max(1) >= UMBRAL_DUP
    dup.append(es_dup.mean())
    nuevas.append((~es_dup).sum())
    if (~es_dup).any():
        util = (Ey[~torch.from_numpy(es_dup).cuda()] @ Er.T).float().cpu().numpy().max(1)
        nuevas_utiles.append((util >= UMBRAL_UTIL).mean())
    # the reverse direction, to see if the asymmetry matters
    solo1.append((( Ex @ Ey.T).float().cpu().numpy().max(1) < UMBRAL_DUP).sum())
    solo2.append((~es_dup).sum())

print(f"\n=== solapamiento entre los dos modelos (n={len(dup)})")
print(f"  tareas del modelo 2 que repiten una del modelo 1 (cos >= {UMBRAL_DUP}): {100*np.mean(dup):.1f} %")
print(f"  tareas nuevas que aporta el modelo 2 por historia: {np.mean(nuevas):.2f}")
print(f"  de esas nuevas, cuantas se parecen a una tarea real (cos >= {UMBRAL_UTIL}): {100*np.mean(nuevas_utiles):.1f} %")
print(f"  tareas exclusivas de cada uno por historia: modelo 1 {np.mean(solo1):.2f}, modelo 2 {np.mean(solo2):.2f}")

# --------------------------------------------------- who wins where
s1 = ev.puntuar(sb, g1, refs)["mat"][:, 0]
s2 = ev.puntuar(sb, g2, refs)["mat"][:, 0]
n_ref = np.array([len(r) for r in refs])
n1 = np.array([len(x) for x in g1])
n2 = np.array([len(x) for x in g2])
print(f"\n=== quien gana donde (n={len(s1)})")
print(f"  gana el modelo 1 en {100*(s1 > s2 + 0.01).mean():.1f} % de las historias, "
      f"el 2 en {100*(s2 > s1 + 0.01).mean():.1f} %, empate en {100*(abs(s1-s2) <= 0.01).mean():.1f} %")
for lo, hi, etq in [(2, 3, "2-3 tareas reales"), (4, 6, "4-6"), (7, 99, "7 o mas")]:
    m = (n_ref >= lo) & (n_ref <= hi)
    if m.sum() > 5:
        print(f"  {etq:20s} n={m.sum():3d}  m1 {s1[m].mean():.4f} ({n1[m].mean():.1f} tareas)  "
              f"m2 {s2[m].mean():.4f} ({n2[m].mean():.1f} tareas)  dif {s2[m].mean()-s1[m].mean():+.4f}")
rng = np.random.default_rng(ev.SEED)
d = s2 - s1
bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
print(f"  modelo 2 menos modelo 1: {d.mean():+.4f} IC95 [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}]")

# --------------------------------------------- near-duplicates left by the pipeline
B = [json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")]
B = [b for b in B if b["sistema"] == "agix"]
tot, rep = 0, 0
for b in B:
    for lista in b["tareas"]:
        if len(lista) < 2:
            tot += len(lista)
            continue
        E = sb.encode(lista, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        S = (E @ E.T).float().cpu().numpy()
        np.fill_diagonal(S, 0)
        for i in range(1, len(lista)):
            if (S[i, :i] >= UMBRAL_DUP).any():
                rep += 1
        tot += len(lista)
print(f"\n=== casi-duplicados dentro de una misma historia en la salida real del pipeline")
print(f"  {rep} de {tot} tareas ({100*rep/tot:.1f} %) repiten otra de la MISMA historia con cos >= {UMBRAL_DUP}")
print("  (el pipeline colapsa duplicados ENTRE historias, pero dentro de una historia solo por texto exacto)")

R.to_csv(f"{BASE}/results/38_dos_modelos.csv", index=False)
print(f"\nescrito {BASE}/results/38_dos_modelos.csv")
