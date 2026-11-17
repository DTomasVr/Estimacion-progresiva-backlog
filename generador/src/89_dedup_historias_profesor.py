"""Curacion del profesor: quita las historias CASI-DUPLICADAS dentro de cada backlog.

Motivo medido el 28/09/2026 (scratchpad/solape12.py), comparando a igual tamano de
backlog (12 historias, que es lo que produce el producto):

  historias con al menos un casi-duplicado (SBERT propio, coseno > 0,85)
    humanas de Dalpiaz .................  4,8 %
    profesor v3 (alimenta v8ff) ........ 14,5 %   <- 3x las personas
    nuestro modelo ..................... 10,7 %

Es decir que el defecto viene con los datos, no lo inventa el modelo. Y es exactamente
el que la literatura de RE senala como el fallo persistente de las historias generadas
por LLM: «the models continue to struggle with generating independent and unique user
stories, often producing semantically overlapping outputs» (Can LLMs Generate User
Stories and Assess Their Quality?, 2025). La independencia es tambien uno de los
criterios del framework QUS (Lucassen et al.), que esta tesis ya usa.

Encaja en el eje de CURACION del repaso de Long et al., «On LLMs-Driven Synthetic Data
Generation, Curation, and Evaluation» (Findings ACL 2024): filtrar lo generado antes de
entrenar, en vez de generar mas.

  python src/89_dedup_historias_profesor.py [UMBRAL] [MINIMO]

out: data/synth/backlogs_v9dd_<modelo>.jsonl  y  results/89_dedup.csv
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = r"F:\Software Task generation Model"
ENTRADA = os.environ.get("AGIX_SYNTH_IN", "backlogs_v3_qwen2.5_7b.jsonl")
SALIDA = os.environ.get("AGIX_SYNTH_OUT", "backlogs_v9dd_qwen2.5_7b.jsonl")
UM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
MIN_H = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SBERT = f"{BASE}/models/sbert_agix_v1"


def texto(h):
    return h["historia"] if isinstance(h, dict) else str(h)


def main():
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer(SBERT if os.path.isdir(SBERT) else "all-MiniLM-L6-v2",
                             device="cpu")
    sb.max_seq_length = 128
    filas = [json.loads(l) for l in open(f"{BASE}/data/synth/{ENTRADA}", encoding="utf-8")]
    print(f"{ENTRADA}: {len(filas)} backlogs")
    n_ant = n_desp = 0
    reg = []
    for x in filas:
        clave = "historias" if "historias" in x else "backlog"
        hs = x.get(clave) or []
        n_ant += len(hs)
        if len(hs) <= MIN_H:
            n_desp += len(hs)
            reg.append((x.get("sector", ""), len(hs), len(hs)))
            continue
        E = sb.encode([texto(h) for h in hs], normalize_embeddings=True,
                      show_progress_bar=False, batch_size=64)
        S = E @ E.T
        # voraz: se recorre en orden y se descarta la que repite una ya aceptada,
        # asi se conserva siempre la primera formulacion y el orden del profesor
        keep = []
        for i in range(len(hs)):
            if all(S[i, j] <= UM for j in keep):
                keep.append(i)
        if len(keep) < MIN_H:                      # nunca dejar el backlog corto
            resto = [i for i in range(len(hs)) if i not in keep]
            keep = sorted(keep + resto[:MIN_H - len(keep)])
        x[clave] = [hs[i] for i in keep]
        n_desp += len(keep)
        reg.append((x.get("sector", ""), len(hs), len(keep)))
    with open(f"{BASE}/data/synth/{SALIDA}", "w", encoding="utf-8") as f:
        for x in filas:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    pd.DataFrame(reg, columns=["sector", "antes", "despues"]).to_csv(
        f"{BASE}/results/89_dedup.csv", index=False, encoding="utf-8")
    print(f"historias: {n_ant} -> {n_desp} ({100 * (n_ant - n_desp) / n_ant:.1f} % fuera)")
    print(f"por backlog: {n_ant / len(filas):.1f} -> {n_desp / len(filas):.1f}")
    print(f"-> data/synth/{SALIDA}")


if __name__ == "__main__":
    main()
