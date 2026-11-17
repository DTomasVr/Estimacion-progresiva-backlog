"""Variante v3b: la mitad de arriba SIN la duplicacion despublicada.

Por que. El metodo A (v3a) arreglo los roles y el relleno pero perdio en las 328
historias humanas: 0,6491 -> 0,6215, IC95 [-0,047; -0,007], gana en 1 de 6. El
sospechoso es la variante sin publico, que hizo dos cosas a la vez:

  1. duplico las filas de arriba con una entrada MENOS informativa (sin decir
     para quien es el producto), y
  2. subio la proporcion de sintetico en el conjunto de arriba del 37 % al 59 %,
     porque los reales no se duplicaron.

v3b aisla eso: mismo conjunto v3a pero quitando las filas despublicadas. No hace
falta regenerar ni volver a mezclar, se filtra por la columna `estilo`.

  python src/62_datos_v3b.py

out: data/processed/s2s_v3b_{train,val,test}.parquet
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for split in ["train", "val", "test"]:
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_v3a_{split}.parquet")
    n = len(d)
    fuera = d.estilo.astype(str) == "sintetico_sin_publico"
    d = d[~fuera].reset_index(drop=True)
    d.to_parquet(f"{BASE}/data/processed/s2s_v3b_{split}.parquet", index=False)
    sint = (d.estilo.astype(str) == "sintetico").sum()
    print(f"s2s_v3b_{split}: {n} -> {len(d)}  (quitadas {int(fuera.sum())} despublicadas) | "
          f"sintetico {sint} de {len(d)} = {100*sint/max(1,len(d)):.0f} %")
