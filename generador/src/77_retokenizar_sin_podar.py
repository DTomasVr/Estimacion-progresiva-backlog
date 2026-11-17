"""Re-tokeniza los datos de la mitad de abajo con el tokenizador COMPLETO de mT5.

Para comparar podado contra sin podar hay que entrenar con el mismo TEXTO, no con
los mismos identificadores: los del corpus podado no significan nada para el modelo
completo. El texto se recupera decodificando lo que se guardo (asi la comparacion
usa exactamente las mismas cadenas, con el mismo truncado, que vio el modelo
podado) y se vuelve a codificar con el tokenizador de 250.100 piezas.

  python src/77_retokenizar_sin_podar.py

out: data/processed/s2s_v5nb_{train,val}.parquet   (para DATOS=v5nb)
     data/processed/s2s_nb_test.parquet            (para AGIX_TEST_PARQUET)
"""
import os
import time

import pandas as pd
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f"{BASE}/models/hf_cache"
MAX_IN, MAX_OUT = 256, 192

tk_pod = AutoTokenizer.from_pretrained(f"{BASE}/models/mt5-small-pruned")
tk_full = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=CACHE)
print(f"podado {len(tk_pod):,} -> completo {len(tk_full):,}", flush=True)


def rehacer(entrada, salida):
    t0 = time.time()
    d = pd.read_parquet(f"{BASE}/data/processed/{entrada}.parquet").reset_index(drop=True)
    src = tk_pod.batch_decode(list(d.input_ids), skip_special_tokens=True)
    tgt = tk_pod.batch_decode(list(d.labels), skip_special_tokens=True)
    xi = tk_full(src, add_special_tokens=True, truncation=True, max_length=MAX_IN).input_ids
    yi = tk_full(tgt, add_special_tokens=True, truncation=True, max_length=MAX_OUT).input_ids
    d["input_ids"] = [list(map(int, x)) for x in xi]
    d["labels"] = [list(map(int, y)) for y in yi]
    d["src"] = src                      # el texto exacto que se usa, por si se audita
    d.to_parquet(f"{BASE}/data/processed/{salida}.parquet", index=False)
    print(f"  {entrada} -> {salida}: {len(d):,} filas, "
          f"piezas entrada {sum(map(len, xi)):,} objetivo {sum(map(len, yi)):,}, "
          f"{time.time()-t0:.0f}s", flush=True)


rehacer("s2s_v5_train", "s2s_v5nb_train")
rehacer("s2s_v5_val", "s2s_v5nb_val")
rehacer("s2s_pruned_test", "s2s_nb_test")
print("listo")
