"""What the shipped pipeline would lose and gain by de-duplicating tasks INSIDE a story.

Today 19_pipeline.py collapses near-duplicate tasks ACROSS stories (cosine >= 0.90)
but skips pairs from the same story, so the two ensemble models leave near-copies
behind ("Create a multi mode" / "Create an multi mode"). The similarity matrix is
already computed for the cross-story pass, so the fix costs no extra time.

Two measurements, each on the population where it means something:

  A. task-level fidelity, on the test set where real tasks exist (n=300):
     current behaviour (exact-text dedup) vs the fix at 0.90 and at 0.95,
     with paired bootstrap CIs over the same examples.

  B. the shipped bench output (64 real projects): how many tasks per project
     disappear. No reference tasks exist there, so no fidelity is claimed.

out: results/39_dedup.csv
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

M1 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1"
M2 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1_bs8_lr9.2e-4"
N = 300
CFG = ev.CFGS["beam4_actual"]

T = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_test.parquet")
T = T[T.mitad == "tasks"].sample(N, random_state=ev.SEED).reset_index(drop=True)
refs = [list(x) for x in T.tareas]
sb = SentenceTransformer(ev.SBERT, device="cuda")


def generar(ruta):
    tok, m = ev.cargar_modelo(ruta)
    g = ev.generar(tok, m, T, CFG, bs=16)
    del m
    torch.cuda.empty_cache()
    return g


g1, g2 = generar(M1), generar(M2)


def exacto(lista):
    """what the pipeline does today: same text, case-insensitive"""
    out, vistos = [], set()
    for t in lista:
        k = t.strip().lower()
        if len(t.strip()) >= 5 and k not in vistos:
            vistos.add(k)
            out.append(t.strip())
    return out


def semantico(lista, umbral):
    if not lista:
        return []
    E = sb.encode(lista, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    S = (E @ E.T).float().cpu().numpy()
    vistos, out = np.zeros(len(lista), bool), []
    for i in range(len(lista)):
        if vistos[i]:
            continue
        vistos |= (S[i] >= umbral)
        out.append(lista[i])
    return out


crudo = [list(a) + list(b) for a, b in zip(g1, g2)]
variantes = {
    "actual (texto exacto)": [exacto(x) for x in crudo],
    "arreglo, umbral 0.95": [semantico(exacto(x), 0.95) for x in crudo],
    "arreglo, umbral 0.90": [semantico(exacto(x), 0.90) for x in crudo],
    "arreglo, umbral 0.85": [semantico(exacto(x), 0.85) for x in crudo],
}

print(f"=== A. fidelidad a nivel de tarea, {N} ejemplos de prueba (los mismos en las cuatro filas)")
punt, filas = {}, []
for nombre, gen in variantes.items():
    s = ev.puntuar(sb, gen, refs)
    punt[nombre] = s["mat"][:, 0]
    filas.append(dict(variante=nombre, sF1=round(s["sF1"], 4), precision=round(s["sPrec"], 4),
                      cobertura=round(s["sCob"], 4), tareas=round(np.mean([len(x) for x in gen]), 2)))
A = pd.DataFrame(filas)

base = punt["actual (texto exacto)"]
rng = np.random.default_rng(ev.SEED)
dif = []
for nombre in list(variantes)[1:]:
    d = punt[nombre] - base
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    dif.append(dict(variante=nombre, delta_sF1=round(d.mean(), 4),
                    ic95_lo=round(np.quantile(bs, .025), 4), ic95_hi=round(np.quantile(bs, .975), 4)))
A = A.merge(pd.DataFrame(dif), on="variante", how="left")
A["tareas_%"] = (100 * A.tareas / A.tareas.iloc[0] - 100).round(1)
print(A.to_string(index=False))

print("\n=== B. la salida real del banco: cuantas tareas por proyecto desaparecen")
B = [json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")]
B = [b for b in B if b["sistema"] == "agix"]
filas = []
for umbral in (0.95, 0.90, 0.85):
    antes = np.array([sum(len(l) for l in b["tareas"]) for b in B])
    despues = np.array([sum(len(semantico(l, umbral)) for l in b["tareas"]) for b in B])
    filas.append(dict(umbral=umbral, proyectos=len(B), tareas_antes=round(antes.mean(), 1),
                      tareas_despues=round(despues.mean(), 1),
                      quitadas_pct=round(100 * (1 - despues.sum() / antes.sum()), 1)))
Bt = pd.DataFrame(filas)
print(Bt.to_string(index=False))

A.to_csv(f"{BASE}/results/39_dedup.csv", index=False)
Bt.to_csv(f"{BASE}/results/39_dedup_banco.csv", index=False)
print(f"\nescrito results/39_dedup.csv y results/39_dedup_banco.csv")
print("nota: el coste en tiempo es cero -- el pipeline ya calcula esa matriz de similitud "
      "para deduplicar entre historias; el arreglo es quitar la condicion que salta los pares "
      "de la misma historia.")
