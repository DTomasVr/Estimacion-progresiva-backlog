"""Tercer intento: el arbitro es el corpus HUMANO, que es independiente del profesor.

Por que hacia falta cambiar de arbitro. El filtro anterior quitaba la palabra si el
profesor la usaba en otro sector, y eso elimina justo las que quiero cazar: «garment»
cae porque el profesor YA la usa fuera de la lavanderia, que es la fuga misma. El
criterio confundia «palabra legitima en varios dominios» con «palabra que se filtra».

Arbitro nuevo: las 1.680 historias humanas de Dalpiaz, 22 proyectos de dominios
distintos. Una palabra curada se veta solo si las personas NO la usan de forma
extendida: si aparece en mas de MAX_PROY proyectos humanos, es vocabulario general y no
se toca.
"""
import collections
import json
import re

import pandas as pd

BASE = "F:/Software Task generation Model"
MAX_PROY = 2
LEX = json.load(open(f"{BASE}/data/processed/lexico_sector.json", encoding="utf-8"))
PROPIO = LEX["propio"]

d = pd.read_parquet(f"{BASE}/data/processed/s2s_dalpiaz.parquet")
por_proy = collections.defaultdict(set)
for proy, lst in zip(d.project, d.tareas):
    for h in list(lst):
        for w in re.findall(r"[a-z]{4,}", str(h).lower()):
            por_proy[w].add(proy)
n_proy = d.project.nunique()
print(f"arbitro: {n_proy} proyectos humanos, {len(por_proy)} palabras distintas")

estricto, caidas = {}, {}
for sec, ws in PROPIO.items():
    quedan = []
    for w in ws:
        n = len(por_proy.get(w, ()))
        if n <= MAX_PROY:
            quedan.append(w)
        else:
            caidas[w] = (sec, n)
    estricto[sec] = quedan

json.dump(dict(fuente="rasgos curados, filtrados por uso humano extendido",
               max_proyectos_humanos=MAX_PROY, propio=estricto,
               sectores=sorted(estricto)),
          open(f"{BASE}/data/processed/lexico_sector_estricto.json", "w",
               encoding="utf-8"), ensure_ascii=False, indent=1)

n_antes = sum(len(v) for v in PROPIO.values())
n_desp = sum(len(v) for v in estricto.values())
tam = sorted(len(v) for v in estricto.values())
print(f"palabras: {n_antes} curadas -> {n_desp} tras el filtro humano "
      f"({100*n_desp/max(1,n_antes):.0f} %)")
print(f"por sector: mediana {tam[len(tam)//2]}, min {tam[0]}, max {tam[-1]}")
print(f"\ncaen por ser de uso humano extendido ({len(caidas)}), las 14 mas extendidas:")
for w, (sec, n) in sorted(caidas.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"  {w:<14} de {sec:<14} en {n}/{n_proy} proyectos humanos")
print("\nlo que SOBREVIVE donde importa:")
for s in ["lavanderia", "cuidado", "eventos", "banca", "farmacia", "restaurantes",
          "colegio", "inmuebles alquiler", "movilidad"]:
    if s in estricto:
        print(f"  {s:<20} {len(estricto[s]):>2}: {', '.join(estricto[s])}")
print("\n-> data/processed/lexico_sector_estricto.json")
