"""Evalua la ficha de funcion (paso 2): obediencia y reparto de clases.

Dos medidas, y hacen falta las dos:

  obediencia   de las historias generadas con `| funcion: Y |`, cuantas piden de
               verdad una funcion de la clase Y. Es el equivalente de la obediencia
               a la ficha de rol, que con el modelo v3r es del 100 %.
  entropia     reparto de las seis clases en el backlog SELECCIONADO. Es el defecto
               que motiva el paso 2: el modelo viejo se atasca en la clase view
               (42 % de las historias, contra 22 % de las humanas y del profesor) y
               la entropia sale 2,22 contra 2,45 humana, sobre un maximo de 2,58.

  python src/85_eval_ficha_funcion.py <ruta_arriba_nueva>

out: results/85_ficha_funcion.csv
"""
import collections
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGIX_DEVICE", "cuda")
os.environ.setdefault("AGIX_SEED", "20260926")
NUEVO = sys.argv[1] if len(sys.argv) > 1 else (
    f"{BASE}/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v8ff_agix_v1")

DESCRIPCIONES = [
    "A digital bank app where customers open an account with identity verification, "
    "schedule transfers, freeze their card and receive fraud alerts.",
    "A platform for organising weddings where the couple hires vendors, builds the guest "
    "list with their food restrictions and tracks the budget.",
    "An online supermarket where customers browse the catalogue, build a basket and book "
    "a delivery slot, and staff confirm the deliveries.",
    "A pharmacy app where patients upload their prescription, search for medicines with "
    "equivalents and request home delivery, and the pharmacist validates controlled drugs.",
    "A school management system with accounts for students, teachers, guardians and "
    "administration, with timetables, tutor messaging and absence reports.",
]


def entropia(cuenta, clases):
    n = sum(cuenta.get(c, 0) for c in clases)
    if not n:
        return 0.0
    p = np.array([cuenta.get(c, 0) / n for c in clases], dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cargar(ruta_arriba):
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    if ruta_arriba:
        pl.M_ARRIBA = ruta_arriba
    P = pl.Pipeline(traducir=False)
    return pl, P


filas = []
for etq, ruta in [("v3r (ficha de rol)", ""), ("v8ff (rol + funcion)", NUEVO)]:
    pl, P = cargar(ruta)
    clases = pl._func()["clases"]
    print(f"\n=== {etq}: arriba = {os.path.basename(pl.M_ARRIBA)}  "
          f"ficha de funcion = {P.usa_funcion}", flush=True)

    # 1. obediencia, midiendo cada par por separado
    ob_ok = ob_tot = 0
    if P.usa_funcion:
        for d in DESCRIPCIONES:
            roles = P.planificador(d) if P.planificador else []
            for r, cl in pl.plan_funciones(roles, 12):
                gen = P._generar(P.tok_a, P.m_a,
                                 [f"feature -> user story | role: {r} | funcion: {cl} | "
                                  f"{d} | examples: "],
                                 num_return_sequences=8, max_new_tokens=64,
                                 do_sample=True, top_p=0.95, temperature=1.0)
                for h in gen:
                    c = pl.clase_funcion(h)
                    if c:
                        ob_tot += 1
                        ob_ok += (c == cl)
        print(f"  obediencia a la ficha de funcion: {ob_ok}/{ob_tot} = "
              f"{100*ob_ok/max(1,ob_tot):.1f} %", flush=True)

    # 2. reparto de clases en el backlog seleccionado
    cnt = collections.Counter()
    n_hist = 0
    for d in DESCRIPCIONES:
        r = P(d, k_historias=12, n_muestras=36)
        for h in r["historias"]:
            c = pl.clase_funcion(h["historia"])
            n_hist += 1
            if c:
                cnt[c] += 1
    H = entropia(cnt, clases)
    rep = ", ".join(f"{c} {100*cnt[c]/max(1,sum(cnt.values())):.0f}%"
                    for c, _ in cnt.most_common())
    print(f"  {n_hist} historias; entropia de clases {H:.2f} / {np.log2(len(clases)):.2f}")
    print(f"  reparto: {rep}", flush=True)
    filas.append(dict(version=etq, n=n_hist, entropia=round(H, 3),
                      obediencia=round(100 * ob_ok / max(1, ob_tot), 1) if ob_tot else None,
                      **{c: cnt.get(c, 0) for c in clases}))
    del P

R = pd.DataFrame(filas)
R.to_csv(f"{BASE}/results/85_ficha_funcion.csv", index=False)
print("\n" + R.to_string(index=False))
print("\nreferencias: humanas 2,45  profesor 2,51  maximo 2,58")
print("-> results/85_ficha_funcion.csv")
