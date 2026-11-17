"""Aggregates the human study once the rating files come back.

    python src/37_analisis_humano.py entrega/respuestas_*.json

Unblinds with results/30_estudio_clave.json (never shown to the raters), then
reports per system: mean plausibility, share kept, and how often each backlog
was picked as the starting point. Also:
  * agreement between raters (Krippendorff-style alpha for ordinal ratings,
    plus plain pairwise correlation, so a low n is visible)
  * correlation between the human rating and our automatic metric -- the point
    of the study is to learn whether the metric predicts judgement
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = r"F:\Software Task generation Model"
clave = json.load(open(f"{BASE}/results/30_estudio_clave.json", encoding="utf-8"))
items = {x["id"]: x for x in json.load(open(f"{BASE}/results/30_estudio_items.json", encoding="utf-8"))}

rutas = []
for a in sys.argv[1:]:
    rutas += glob.glob(a)
if not rutas:
    rutas = glob.glob(f"{BASE}/entrega/respuestas_*.json")
if not rutas:
    raise SystemExit("no hay archivos de respuestas (entrega/respuestas_*.json)")
print(f"{len(rutas)} evaluador(es): {[r.split('respuestas_')[-1] for r in rutas]}")

filas, conj = [], []
for r in rutas:
    d = json.load(open(r, encoding="utf-8"))
    quien = d.get("quien") or r.split("respuestas_")[-1].replace(".json", "")
    for k, v in d.items():
        if k.startswith("p_") or k.startswith("k_"):
            iid = k[2:]
            filas.append(dict(evaluador=quien, item=iid, sistema=clave["items"].get(iid, "?"),
                              campo="plausible" if k[0] == "p" else "conservar", valor=v))
        elif k.startswith("mejor_") or k.startswith("repite_"):
            caso = k.split("_", 1)[1]
            letra = v
            sistema = clave["conjuntos"].get(caso, {}).get(letra, letra)
            conj.append(dict(evaluador=quien, caso=caso, campo=k.split("_")[0],
                             letra=letra, sistema=sistema))

D = pd.DataFrame(filas)
if D.empty:
    raise SystemExit("las respuestas no traen valoraciones")
P = D[D.campo == "plausible"].copy()
P["valor"] = P.valor.astype(float)
K = D[D.campo == "conservar"]

print("\n=== por sistema (a ciegas hasta ahora)")
res = P.groupby("sistema").valor.agg(["mean", "std", "count"]).round(3)
res["conservaria_%"] = (100 * K[K.valor == "si"].groupby("sistema").size() /
                        K.groupby("sistema").size()).round(1)
res["conservaria_o_editaria_%"] = (100 * K[K.valor.isin(["si", "editar"])].groupby("sistema").size() /
                                   K.groupby("sistema").size()).round(1)
print(res.to_string())

if len(rutas) > 1:
    piv = P.pivot_table(index="item", columns="evaluador", values="valor")
    piv = piv.dropna()
    print(f"\n=== acuerdo entre evaluadores ({len(piv)} items valorados por todos)")
    if len(piv.columns) > 1 and len(piv) > 2:
        print(piv.corr(method="spearman").round(3).to_string())
        obs = piv.var(axis=1, ddof=0).mean()
        esp = piv.values.flatten().var()
        print(f"alfa aproximado (1 - varianza interna / total): {1 - obs / esp:.3f}")

C = pd.DataFrame(conj)
if not C.empty:
    print("\n=== backlog elegido como punto de partida")
    print(C[C.campo == "mejor"].groupby("sistema").size().rename("veces").to_string())
    print("\n=== percibido como el mas repetitivo")
    print(C[C.campo == "repite"].groupby("sistema").size().rename("veces").to_string())

# does the automatic metric track the human judgement?
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from sentence_transformers import SentenceTransformer
    O = pd.concat([pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet"),
                   pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")])
    refs = O[O.estilo == "historia"].groupby("parent_uid").objetivo.apply(list).to_dict()
    med = P.groupby("item").valor.mean()
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    gen, ref, hum = [], [], []
    for iid, v in med.items():
        it = items.get(iid)
        if it and it["caso"] in refs:
            gen.append([it["historia"]])
            ref.append(refs[it["caso"]])
            hum.append(v)
    if len(hum) > 8:
        s = ev.puntuar(sb, gen, ref)
        auto = s["mat"][:, 0]
        r = np.corrcoef(auto, hum)[0, 1]
        print(f"\n=== la metrica automatica contra el juicio humano (n={len(hum)})")
        print(f"correlacion de Pearson: {r:+.3f}")
        print("  >0,4 respalda la metrica; cerca de 0 la desacredita y habria que decirlo")
except Exception as e:
    print(f"\n(no se pudo calcular la correlacion con la metrica: {e})")

salida = f"{BASE}/results/37_estudio_humano.csv"
res.to_csv(salida)
print(f"\nescrito {salida}")
