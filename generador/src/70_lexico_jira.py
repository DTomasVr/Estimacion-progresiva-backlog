"""Lexico de FUGAS del corpus de Jira, derivado de los datos y no a mano.

Las salidas traen tareas que no pertenecen al dominio de producto: "Ver los
detalles de las cuentas de los inquilinos" (de *tenant*, software multi-inquilino),
"Crear un constructor", "Crear una infraestructura para los coleccionistas",
"Anadir documentacion", "revise y vote sobre los nuevos invitados". Todas vienen
de que la mitad de abajo entrena con 22.871 filas de Jira contra 6.106 sinteticas.

Para medir cuanto se fuga hace falta un lexico, y ponerlo a mano seria hacer
trampa (uno lista justo lo que ya vio). Se deriva comparando frecuencias: se
queda con las palabras cuya frecuencia relativa en las tareas de Jira es al menos
RATIO veces la que tienen en las tareas sinteticas.

out: data/processed/lexico_jira.json
"""
import collections
import json
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATIO = 12.0        # cuantas veces mas frecuente en Jira que en sintetico
MIN_N = 40          # apariciones minimas en Jira, para no coger rarezas
TOPE = 300


def cuenta(tareas):
    c = collections.Counter()
    n = 0
    for t in tareas:
        for w in re.findall(r"[a-z]{4,}", str(t).lower()):
            c[w] += 1
            n += 1
    return c, n


d = pd.read_parquet(f"{BASE}/data/processed/s2s_v5_train.parquet",
                    columns=["tareas", "estilo"])
jira = [t for lst in d[d.estilo == "jira"].tareas for t in list(lst)]
sint = [t for lst in d[d.estilo == "sintetico"].tareas for t in list(lst)]
cj, nj = cuenta(jira)
cs, ns = cuenta(sint)
print(f"tareas: {len(jira)} de Jira ({nj} palabras), {len(sint)} sinteticas ({ns})")

filas = []
for w, n in cj.items():
    if n < MIN_N:
        continue
    fj = n / nj
    fs = cs.get(w, 0) / ns
    r = fj / fs if fs else float("inf")
    if r >= RATIO:
        filas.append((w, n, round(min(r, 999), 1)))
filas.sort(key=lambda x: -x[1])
filas = filas[:TOPE]

json.dump([w for w, _, _ in filas],
          open(f"{BASE}/data/processed/lexico_jira.json", "w", encoding="utf-8"),
          indent=1)
print(f"{len(filas)} palabras propias de Jira (ratio >= {RATIO}, n >= {MIN_N})")
print("las 40 mas frecuentes:")
for i in range(0, min(40, len(filas)), 8):
    print("  " + ", ".join(f"{w}" for w, _, _ in filas[i:i + 8]))
# cuantas tareas sinteticas contienen alguna: deberia ser bajo, es el control
lex = {w for w, _, _ in filas}
con = sum(1 for t in sint if lex & set(re.findall(r"[a-z]{4,}", str(t).lower())))
conj = sum(1 for t in jira if lex & set(re.findall(r"[a-z]{4,}", str(t).lower())))
print(f"\ncontrol: tareas con alguna de esas palabras -> "
      f"Jira {100*conj/len(jira):.1f} %, sintetico {100*con/len(sint):.1f} %")
print("-> data/processed/lexico_jira.json")
