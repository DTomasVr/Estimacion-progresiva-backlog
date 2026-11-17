"""Subconjunto SEGURO del lexico de Jira: palabras que el dominio de producto no usa nunca.

Por que hace falta. El filtro de tareas degeneradas del pipeline lleva una docena
de patrones escritos a mano (`snapshot`, `maven`, `branch`...), y por eso siguen
saliendo fugas como "Crear una distribucion de origen para la version" o
"Prepararse para una lista de correo". Pero el lexico de 300 palabras de
src/70_lexico_jira.py NO se puede usar tal cual como filtro: contiene "remove",
"should", "make", "build", "error", "default", "read", "handle", "security", que
son legitimas en un backlog de producto.

El criterio de aqui es automatico y comprobable: se queda la palabra solo si
aparece CERO veces en TODO el texto del dominio de producto. El control tiene que
ser amplio, no solo las tareas: con solo las tareas sinteticas pasaban el corte
"should", "where", "better", "sure" e "instead", que son inglés corriente y solo
faltaban porque las tareas del profesor tienen un estilo muy fijo. Se anaden las
HISTORIAS sinteticas y las 622 descripciones v3, donde ese vocabulario si aparece.
Si el dominio nunca usa la palabra en ninguna parte, vetarla no puede costar una
tarea legitima.

  python src/81_lexico_jira_estricto.py

out: data/processed/lexico_jira_estricto.json
"""
import collections
import json
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_JIRA = 40          # igual que en el 70: no coger rarezas

d = pd.read_parquet(f"{BASE}/data/processed/s2s_v5_train.parquet",
                    columns=["tareas", "estilo", "src"])
jira = [t for lst in d[d.estilo == "jira"].tareas for t in list(lst)]
sint = [t for lst in d[d.estilo == "sintetico"].tareas for t in list(lst)]
# control amplio: las tareas de producto MAS las historias que las generaron y las
# descripciones de la taxonomia
sint += [str(s) for s in d[d.estilo == "sintetico"].src]
V3 = f"{BASE}/data/synth/descripciones_v3.json"
if os.path.exists(V3):
    D3 = json.load(open(V3, encoding="utf-8"))
    sint += [x["descripcion"] for x in D3] + [x.get("descripcion_sin", "") for x in D3]


def cuenta(tareas):
    c = collections.Counter()
    for t in tareas:
        for w in re.findall(r"[a-z]{4,}", str(t).lower()):
            c[w] += 1
    return c


cj, cs = cuenta(jira), cuenta(sint)
print(f"tareas: {len(jira)} de Jira, {len(sint)} textos de producto", flush=True)

amplio = set(json.load(open(f"{BASE}/data/processed/lexico_jira.json", encoding="utf-8")))
estricto = sorted(w for w in amplio if cj.get(w, 0) >= MIN_JIRA and cs.get(w, 0) == 0)
descartadas = sorted(w for w in amplio if cs.get(w, 0) > 0)

print(f"lexico amplio {len(amplio)} -> estricto {len(estricto)}")
print(f"\nse QUEDAN (cero apariciones en producto):")
for i in range(0, len(estricto), 10):
    print("  " + ", ".join(estricto[i:i + 10]))
print(f"\nse DESCARTAN por aparecer tambien en producto ({len(descartadas)}), las 30 mas usadas:")
top = sorted(descartadas, key=lambda w: -cs[w])[:30]
print("  " + ", ".join(f"{w}({cs[w]})" for w in top))

json.dump(estricto, open(f"{BASE}/data/processed/lexico_jira_estricto.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n-> data/processed/lexico_jira_estricto.json ({len(estricto)} palabras)")
