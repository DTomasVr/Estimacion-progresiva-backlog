"""Datos para el reordenador de beneficios: candidatas del PROPIO modelo,
etiquetadas por el juez local.

Por que del propio modelo y no del profesor. Las 7.423 historias del profesor
salen 89 % coherentes (results/65_curacion.csv), que es un conjunto malo para
entrenar un discriminador: casi no hay negativos. La salida del modelo esta al
34 % de coherencia, o sea dos de cada tres candidatas son negativas. Es muestreo
por rechazo: se generan candidatas, se etiquetan, y el clasificador aprende a
separar las que el modelo produce bien de las que produce mal.

Se generan SOLO las candidatas de la mitad de arriba (no las tareas), que es
mucho mas rapido, y se usan descripciones del split de ENTRENAMIENTO, para que
las reservadas sigan sirviendo de test.

  python src/73_datos_beneficio.py [N_DESCRIPCIONES]

out: data/synth/beneficio_etiquetado.jsonl
"""
import importlib.util
import json
import os
import re
import sys
import time

import pandas as pd
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(next((a for a in sys.argv[1:] if a.isdigit()), 120))
os.environ.setdefault("PBE_DEVICE", "cuda")
os.environ.setdefault("PBE_SEED", "20260926")

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
spec2 = importlib.util.spec_from_file_location("cur", f"{BASE}/src/65_curar_datos.py")
cur = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cur)          # reutiliza el juez y el partidor de historias

D = json.load(open(f"{BASE}/data/synth/descripciones_v3.json", encoding="utf-8"))
t = pd.read_parquet(f"{BASE}/data/processed/s2s_v3a_test.parquet", columns=["parent_uid"])
reservadas = {str(x).split("#")[0] for x in t.parent_uid.dropna()}
casos = [x for x in D if x["id"] not in reservadas][:N]
print(f"{len(casos)} descripciones de entrenamiento (las reservadas quedan fuera)", flush=True)

P = pl.Pipeline(traducir=False)


def candidatas(desc):
    """solo la mitad de arriba: el banco de candidatas condicionadas por actor"""
    if pl.SEMILLA is not None:
        torch.manual_seed(int(pl.SEMILLA))
        if pl.DISPOSITIVO == "cuda":
            torch.cuda.manual_seed_all(int(pl.SEMILLA))
    roles = P.planificador(desc) if P.planificador is not None else []
    gen = dict(max_new_tokens=64, do_sample=True, top_p=0.95, temperature=1.0)
    fuera = []
    if roles:
        for r in roles:
            fuera += P._generar(P.tok_a, P.m_a,
                                [f"feature -> user story | role: {r} | {desc} | examples: "],
                                num_return_sequences=12, **gen)
    else:
        fuera = P._generar(P.tok_a, P.m_a, [f"feature -> user story: {desc} | examples: "],
                           num_return_sequences=36, **gen)
    return [c.strip() for c in fuera if pl.PAT_HIST.match(c.strip())]


trip, meta = [], []
t0 = time.time()
for i, caso in enumerate(casos):
    for h in candidatas(caso["descripcion"]):
        p = cur.partir(h)
        if p:
            trip.append(p)
            meta.append((caso["id"], caso["descripcion"], h))
    if (i + 1) % 20 == 0:
        print(f"  generadas {len(trip)} candidatas de {i+1} descripciones "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

print(f"{len(trip)} candidatas a etiquetar; juez por lotes de {cur.LOTE}", flush=True)
etiquetas, t0 = [], time.time()
for k in range(0, len(trip), cur.LOTE):
    etiquetas += cur.juzgar(trip[k:k + cur.LOTE])
    if k and k % (cur.LOTE * 20) == 0:
        print(f"  {k}/{len(trip)}  faltan ~"
              f"{(len(trip)-k)*(time.time()-t0)/k/60:.0f} min", flush=True)

salida = f"{BASE}/data/synth/beneficio_etiquetado.jsonl"
with open(salida, "w", encoding="utf-8") as fh:
    for (rol, cap, ben), (cid, desc, h), ok in zip(trip, meta, etiquetas):
        fh.write(json.dumps(dict(id=cid, descripcion=desc, historia=h, rol=rol,
                                 capacidad=cap, beneficio=ben, coherente=bool(ok)),
                            ensure_ascii=False) + "\n")
pos = sum(etiquetas)
print(f"\n{len(etiquetas)} etiquetadas: {pos} coherentes ({100*pos/len(etiquetas):.1f} %), "
      f"{len(etiquetas)-pos} no")
print(f"-> {os.path.basename(salida)}")
