"""Cuanto cuesta la poda de vocabulario, medido sobre el texto de prueba.

La poda (src/10_prune_vocab.py) bajo mT5-small de 250.100 piezas a 75.260 y se
verifico que los logits de las piezas conservadas son EXACTOS. Pero eso no dice
nada sobre el coste real, que es otro: con menos piezas la misma palabra se parte
en mas trozos (fertilidad), y una palabra cuyas piezas se fueron se reconstruye
peor. Eso si puede afectar a la calidad, porque cambia la longitud de la secuencia
que el modelo tiene que producir y el numero de decisiones que tiene que acertar.

Aqui se mide sobre el texto de prueba, sin entrenar nada:
  - fertilidad: piezas por palabra con cada tokenizador
  - <unk>: cuantas piezas quedan sin representar
  - ida y vuelta: cuantos textos NO se recuperan exactos al decodificar

  python src/76_coste_poda.py

out: results/76_coste_poda.csv
"""
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = f"{BASE}/models/hf_cache"

tk_full = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=CACHE)
tk_pod = AutoTokenizer.from_pretrained(f"{BASE}/models/mt5-small-pruned")
print(f"vocabulario: completo {len(tk_full):,}  podado {len(tk_pod):,}", flush=True)

d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_test.parquet")
# el texto que el modelo ve de verdad: la fuente tal cual, y el objetivo como se
# guardo (se recupera decodificando las etiquetas con el tokenizador podado)
fuentes = [str(s) for s in d.src]
objetivos = [tk_pod.decode(list(l), skip_special_tokens=True) for l in d.labels]
# y el espanol, que es donde la poda puede doler mas (el corpus de poda es 90 % del es)
es = [str(s) for s in d.titulo_es.dropna()][:2000]

filas = []
for etq, textos in [("fuente (en)", fuentes), ("objetivo (en)", objetivos),
                    ("titulo (es)", es)]:
    r = {}
    for nom, tk in [("completo", tk_full), ("podado", tk_pod)]:
        ids = tk(textos, add_special_tokens=False).input_ids
        pal = sum(len(t.split()) for t in textos)
        piezas = sum(len(x) for x in ids)
        unk = sum(x.count(tk.unk_token_id) for x in ids)
        vuelta = sum(1 for t, x in zip(textos, ids) if tk.decode(x).strip() != t.strip())
        r[nom] = dict(piezas=piezas, fertilidad=piezas / max(1, pal), unk=unk,
                      no_exactos=vuelta)
    filas.append(dict(
        texto=etq, n=len(textos), palabras=sum(len(t.split()) for t in textos),
        piezas_completo=r["completo"]["piezas"], piezas_podado=r["podado"]["piezas"],
        fert_completo=round(r["completo"]["fertilidad"], 3),
        fert_podado=round(r["podado"]["fertilidad"], 3),
        sobrecoste_pct=round(100 * (r["podado"]["piezas"] / r["completo"]["piezas"] - 1), 2),
        unk_completo=r["completo"]["unk"], unk_podado=r["podado"]["unk"],
        inexactos_completo=r["completo"]["no_exactos"],
        inexactos_podado=r["podado"]["no_exactos"]))
    print(f"{etq:<14} n={filas[-1]['n']:>5}  fertilidad {filas[-1]['fert_completo']:.3f} -> "
          f"{filas[-1]['fert_podado']:.3f}  ({filas[-1]['sobrecoste_pct']:+.2f} % piezas)  "
          f"unk {filas[-1]['unk_completo']} -> {filas[-1]['unk_podado']}  "
          f"ida y vuelta inexacta {filas[-1]['inexactos_completo']} -> "
          f"{filas[-1]['inexactos_podado']}", flush=True)

R = pd.DataFrame(filas)
R.to_csv(f"{BASE}/results/76_coste_poda.csv", index=False)
print("\n" + R.to_string(index=False))
print("-> results/76_coste_poda.csv")
