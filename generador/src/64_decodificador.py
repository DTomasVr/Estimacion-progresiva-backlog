"""Barrido del decodificador de la mitad de abajo, sobre los MISMOS pesos y las
mismas entradas. No reentrena nada.

Por que. Medido en esta tanda:
  - con HAZ (lo que hay hoy) la plantilla colapsa: el 82-87 % de las tareas
    empieza por uno de cinco arranques de dos palabras, cuando las tareas
    escritas por personas en el corpus de Jira estan en el 2,6 %.
  - con MUESTREO (top_p 0,92, temp 0,9) la plantilla baja al 39 % pero la
    pertenencia se rompe: la atribucion tarea->historia cae de 78,6 % a 53,5 %.

El haz maximiza verosimilitud y por eso amplifica la formulacion mas probable; el
muestreo la rompe pero pierde el hilo con la historia. Aqui se prueba el punto
medio: HAZ DIVERSO (num_beam_groups con diversity_penalty), que conserva la
busqueda por haz forzando diversidad entre grupos, y un muestreo mas suave.

Se busca la configuracion que baje la concentracion SIN perder atribucion ni
anclaje.

  python src/64_decodificador.py [n_descripciones]

out: results/64_decodificador.csv
"""
import collections
import importlib.util
import os
import re
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PBE_DEVICE", "cuda")
os.environ.setdefault("PBE_CONJUNTO", "no")
os.environ.setdefault("PBE_SEED", "20260926")
os.environ.setdefault("PBE_ARRIBA",
                      f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3r_pbe_v1")
os.environ.setdefault("PBE_ABAJO",
                      f"{BASE}/models/mt5-small-pruned_hist_pbe_v1_v3_pbe_v1")

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
pl.M_ARRIBA, pl.M_ABAJO = os.environ["PBE_ARRIBA"], os.environ["PBE_ABAJO"]

PARADA = {"a", "an", "the", "to", "for", "of", "in", "on", "and", "with", "by",
          "from", "at", "as", "their", "its", "my", "all", "new", "each", "per",
          "be", "is", "are", "should", "able", "that", "this", "it"}

# Las mismas 12 descripciones de consumo con las que se midio todo lo demas
CASOS = [
    "A supermarket app where customers browse the catalogue, build a basket and book a delivery slot.",
    "An online store for selling audiobooks and ebooks, with a library and offline listening.",
    "A gym management app with memberships, class booking and a check-in kiosk.",
    "A veterinary clinic system for appointments, vaccination records and reminders.",
    "A hotel booking platform with room availability, seasonal rates and guest reviews.",
    "A food delivery app connecting restaurants, couriers and customers with live tracking.",
    "A school platform where teachers publish assignments and parents follow grades.",
    "A car workshop system for service appointments, repair quotes and spare part stock.",
    "A recycling collection service with household pickup scheduling and weight reports.",
    "A microloan platform where borrowers apply and analysts review credit risk.",
    "An app to find and pay for street parking, with a timer and monthly invoices.",
    "A wedding planning tool to manage the guest list, seating chart and vendor payments.",
]

# El HAZ DIVERSO (num_beam_groups) ya no esta en el nucleo de transformers de
# esta version: exige trust_remote_code=True y descargar codigo de
# transformers-community/group-beam-search. No se anade esa dependencia (es
# ejecutar codigo de terceros y necesita red), asi que se barren las opciones que
# SI estan en el nucleo, incluida la busqueda por haz CON muestreo
# (do_sample=True junto a num_beams), que es un punto medio real entre las dos.
CONFIGS = [
    ("haz 4 (lo actual)", dict(num_beams=4)),
    ("haz 4 + muestreo t0,7", dict(num_beams=4, do_sample=True, top_p=0.90, temperature=0.7)),
    ("haz 4 + muestreo t0,9", dict(num_beams=4, do_sample=True, top_p=0.92, temperature=0.9)),
    ("muestreo p0,95 t0,5", dict(do_sample=True, top_p=0.95, temperature=0.5)),
    ("muestreo p0,90 t0,7", dict(do_sample=True, top_p=0.90, temperature=0.7)),
    ("muestreo p0,85 t0,8", dict(do_sample=True, top_p=0.85, temperature=0.8)),
    ("muestreo p0,92 t0,9", dict(do_sample=True, top_p=0.92, temperature=0.9)),
]


def metricas(sb, backlogs):
    """concentracion de arranques, anclaje y atribucion, con el mismo criterio
    que src/52_coherencia.py para que las cifras sean comparables"""
    arr = collections.Counter()
    n_t = anclada = 0
    aciertos = tot = 0
    for bl in backlogs:
        hs = [h for h, ts in bl if ts]
        for h, ts in bl:
            hp = set(re.findall(r"[a-z]+", h.lower())) - PARADA
            for t in ts:
                n_t += 1
                w = re.findall(r"[a-z']+", t.lower())[:2]
                if len(w) == 2:
                    arr[" ".join(w)] += 1
                p = [x for x in re.findall(r"[a-z]+", t.lower()) if x not in PARADA]
                if p and (set(p) & hp) and len(set(p) - hp) / len(set(p)) >= 0.3:
                    anclada += 1
        if len(hs) >= 4:
            idx = [i for i, (_, ts) in enumerate([(h, t) for h, t in bl if t]) for _ in ts]
            tareas = [t for _, ts in bl for t in ts]
            if tareas:
                Eh = sb.encode(hs, normalize_embeddings=True, show_progress_bar=False)
                Et = sb.encode(tareas, normalize_embeddings=True, show_progress_bar=False)
                S = Et @ Eh.T
                for k, i in enumerate(idx):
                    if S[k, i] >= np.delete(S[k], i).max():
                        aciertos += 1
                    tot += 1
    t_arr = sum(arr.values())
    return dict(tareas=n_t,
                conc=sum(c for _, c in arr.most_common(5)) / max(1, t_arr),
                top=" | ".join(k for k, _ in arr.most_common(3)),
                anclada=anclada / max(1, n_t),
                atrib=aciertos / max(1, tot))


N = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASOS)
P = pl.Pipeline(traducir=False, conjunto="no")
filas = []
for etq, cfg in CONFIGS:
    pl.GEN_ABAJO = cfg
    t0 = time.time()
    backlogs = []
    for d in CASOS[:N]:
        r = P(d, k_historias=12, n_muestras=36)
        backlogs.append([(h["historia"], list(h["tareas"])) for h in r["historias"]])
    m = metricas(P.sb, backlogs)
    m["config"] = etq
    m["seg_por_backlog"] = round((time.time() - t0) / max(1, N), 2)
    filas.append(m)
    print(f"{etq:28s} tareas {m['tareas']:4d}  concentracion {100*m['conc']:5.1f} %  "
          f"anclaje {100*m['anclada']:5.1f} %  atribucion {100*m['atrib']:5.1f} %  "
          f"{m['seg_por_backlog']:5.2f} s", flush=True)
    print(f"{'':28s} arranques: {m['top']}", flush=True)

D = pd.DataFrame(filas)[["config", "tareas", "conc", "anclada", "atrib", "seg_por_backlog", "top"]]
D.to_csv(f"{BASE}/results/64_decodificador.csv", index=False)
print("\n" + D.round(3).to_string(index=False))
print("\nreferencias: tareas humanas del corpus concentracion 2,6 %; "
      "profesor LLM v3 28,1 %; atribucion del profesor 71,4 %")
print("-> results/64_decodificador.csv")
