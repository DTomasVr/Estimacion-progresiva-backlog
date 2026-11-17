"""Aumento de datos por RETROTRADUCCION de la entrada, del lado del input solamente.

Motivo medido, no elegido por gusto. El paso 4c destapo que el pipeline se entrena con
descripciones en ingles ESCRITAS A MANO, pero en produccion la entrada es espanol que
pasa por Marian (opus-mt-es-en), y esa traduccion no habla igual: rompe actores
(«apoderados» -> «agents»), cambia el registro y reordena. O sea que hay un desajuste
entre entrenamiento e inferencia en la propia distribucion del texto de entrada.

Se arregla del lado de los datos: se anaden copias de los mismos ejemplos con la
descripcion pasada por un viaje de ida y vuelta EN -> ES -> EN con los MISMOS modelos
Marian que el producto usa en inferencia. El objetivo (la historia) no se toca, asi que
esto es aumento de la entrada, no datos sinteticos nuevos.

Respaldo en la literatura 2023+:
  - Ding et al., «Data Augmentation using LLMs» (Findings ACL 2024) y Long et al., «On
    LLMs-Driven Synthetic Data Generation, Curation and Evaluation» (Findings ACL 2024)
    recogen la parafrasis de la entrada como una de las familias de aumento vigentes.
  - Gerstgrasser et al. 2024 («Is Model Collapse Inevitable?»): los datos se ACUMULAN,
    no se reemplazan. Por eso aqui se anaden filas y no se sustituyen.
  - Yu et al., NeurIPS 2023 (AttrPrompt): la diversidad de la entrada es lo que mueve el
    rendimiento del modelo entrenado con datos generados.

  python src/88_retrotraducir_datos.py [FRACCION]     (por omision 0.5)

out: data/processed/s2s_v9bt_{train,val}.parquet  y  results/88_retrotraduccion.csv
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from transformers import MarianMTModel, MarianTokenizer

BASE = r"F:\Software Task generation Model"
CACHE = f"{BASE}/models/hf_cache"
SEED = 20260922
MAX_IN, EOS = 256, 1
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)
sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")


def entrada_ids(src):
    return sp.encode(src)[:MAX_IN - 1] + [EOS]


# la descripcion es el ultimo campo antes de « | examples:», y en el formato con fichas
# va detras de «| funcion: Y |». Se conservan las fichas y los ejemplos intactos.
_CON_FICHA = re.compile(r"^(feature -> user story \| role: [^|]*\| funcion: [^|]*\| )"
                        r"(.*?)( \| examples:.*)$", re.S)
_SIN_FICHA = re.compile(r"^(feature -> user story(?: \| role: [^|]*\|)?:? ?)"
                        r"(.*?)( \| examples:.*)$", re.S)


def partir(src):
    for pat in (_CON_FICHA, _SIN_FICHA):
        m = pat.match(src)
        if m and len(m.group(2).strip()) >= 10:
            return m.group(1), m.group(2), m.group(3)
    return None


def cargar_marian(nombre):
    tk = MarianTokenizer.from_pretrained(nombre, cache_dir=CACHE)
    md = MarianMTModel.from_pretrained(nombre, cache_dir=CACHE).to(DEV).eval()
    return tk, md


def traducir(tk, md, textos, lote=48):
    fuera = []
    for i in range(0, len(textos), lote):
        x = tk(textos[i:i + lote], return_tensors="pt", padding=True,
               truncation=True, max_length=256).to(DEV)
        with torch.no_grad():
            y = md.generate(**x, num_beams=1, max_new_tokens=220)
        fuera += tk.batch_decode(y, skip_special_tokens=True)
        if (i // lote) % 20 == 0:
            print(f"    {i + len(textos[i:i + lote])}/{len(textos)}", flush=True)
    return fuera


def main():
    t0 = time.time()
    partes = {}
    for split in ("train", "val"):
        d = pd.read_parquet(f"{BASE}/data/processed/s2s_v8ff_{split}.parquet")
        partes[split] = d
        print(f"v8ff_{split}: {len(d)} filas, estilos {d.estilo.value_counts().to_dict()}")

    # las descripciones UNICAS de las filas que se van a aumentar
    sel = {}
    for split, d in partes.items():
        ok = np.array([partir(s) is not None for s in d.src])
        # se aumentan las filas con ficha de rol, que son las del dominio de producto
        cand = np.where(ok & d.estilo.eq("sintetico").to_numpy())[0]
        n = int(round(FRAC * len(cand)))
        sel[split] = rng.choice(cand, size=n, replace=False) if n else np.array([], int)
        print(f"{split}: {len(cand)} candidatas con ficha, se aumentan {len(sel[split])}")

    textos = []
    for split, idx in sel.items():
        for i in idx:
            textos.append(partir(partes[split].src.iat[int(i)])[1])
    unicos = sorted(set(textos))
    print(f"\ntextos a retrotraducir: {len(textos)} filas, {len(unicos)} unicos")

    print("EN -> ES ...", flush=True)
    tk1, md1 = cargar_marian("Helsinki-NLP/opus-mt-en-es")
    es = traducir(tk1, md1, unicos)
    del md1
    torch.cuda.empty_cache()
    print("ES -> EN ...", flush=True)
    tk2, md2 = cargar_marian("Helsinki-NLP/opus-mt-es-en")
    en = traducir(tk2, md2, es)
    del md2
    torch.cuda.empty_cache()
    mapa = dict(zip(unicos, en))

    # cuanto cambia el texto: es la medida del desajuste que se quiere cubrir
    def jac(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / max(1, len(A | B))
    j = np.array([jac(u, mapa[u]) for u in unicos])
    ig = sum(1 for u in unicos if u.strip() == mapa[u].strip())
    print(f"\nsolapamiento de palabras tras el viaje: mediana {np.median(j):.3f}, "
          f"media {j.mean():.3f}, identicos {ig}/{len(unicos)}")
    os.makedirs(f"{BASE}/results", exist_ok=True)
    pd.DataFrame(dict(original=unicos, retro=[mapa[u] for u in unicos],
                      jaccard=j)).to_csv(
        f"{BASE}/results/88_retrotraduccion.csv", index=False, encoding="utf-8")

    for split, idx in sel.items():
        d = partes[split]
        if not len(idx):
            fuera = d
        else:
            aum = d.iloc[idx].copy()
            nuevos = []
            for s in aum.src:
                a, desc, c = partir(s)
                nuevos.append(a + mapa.get(desc, desc) + c)
            aum["src"] = nuevos
            aum["input_ids"] = [entrada_ids(s) for s in aum.src]
            aum["estilo"] = "sintetico_bt"
            fuera = pd.concat([d, aum], ignore_index=True)
        p = f"{BASE}/data/processed/s2s_v9bt_{split}.parquet"
        fuera.to_parquet(p, index=False)
        print(f"-> {p}: {len(fuera)} filas "
              f"({len(fuera) / max(1, len(d)):.2f}x)  estilos {fuera.estilo.value_counts().to_dict()}")
    print(f"\nTOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
