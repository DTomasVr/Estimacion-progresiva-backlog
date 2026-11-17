"""Clases de funcion, agrupadas desde los datos y no escritas a mano.

Para el paso 2 hace falta una segunda ficha de control `| funcion: Y |` al estilo de
CTRL. La pregunta es de donde sale Y. Escribirla a mano seria volver al problema del
paso 1, donde mi agrupamiento era la hipotesis; aqui los verbos del medio de las 8.367
historias con rol se agrupan por significado con el SBERT propio (k-medias), y cada
grupo se nombra por su miembro mas frecuente.

  python src/84_clases_funcion.py [K]

out: data/processed/clases_funcion.json
"""
import ast
import collections
import io
import json
import os
import re
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MIN_N = 8
os.environ.setdefault("AGIX_DEVICE", "cpu")

src82 = io.open(f"{BASE}/src/82_rol_funcion.py", encoding="utf-8").read()
trozos = [n for n in ast.parse(src82).body
          if isinstance(n, (ast.FunctionDef, ast.Assign, ast.Import, ast.ImportFrom))]
ns = {"__file__": f"{BASE}/src/82_rol_funcion.py"}
exec(compile(ast.Module(body=trozos, type_ignores=[]), "82", "exec"), ns)

pares = ns["pares_dalpiaz"]() + ns["pares_profesor"]()
datos = [(ns["familia"](r), ns["verbo"](m)) for r, m in pares]
datos = [(f, v) for f, v in datos if v]
cuenta = collections.Counter(v for _, v in datos)
verbos = sorted([v for v, n in cuenta.items() if n >= MIN_N], key=lambda v: -cuenta[v])
print(f"{len(pares)} historias, {len(cuenta)} verbos distintos, "
      f"{len(verbos)} con al menos {MIN_N} apariciones", flush=True)

from sentence_transformers import SentenceTransformer  # noqa: E402
sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1", device=os.environ["AGIX_DEVICE"])
# se codifica el verbo dentro de una frase para que el embedding tenga contexto verbal
E = sb.encode([f"I want to {v} it" for v in verbos], normalize_embeddings=True,
              show_progress_bar=False)

from sklearn.cluster import KMeans  # noqa: E402
km = KMeans(n_clusters=K, n_init=20, random_state=20260922).fit(E)
grupos = collections.defaultdict(list)
for v, c in zip(verbos, km.labels_):
    grupos[int(c)].append(v)

# nombre del grupo: su verbo mas frecuente
nombre = {c: max(vs, key=lambda v: cuenta[v]) for c, vs in grupos.items()}
mapa = {v: nombre[int(c)] for v, c in zip(verbos, km.labels_)}

print(f"\n{K} clases (nombradas por su verbo mas frecuente):")
for c, vs in sorted(grupos.items(), key=lambda kv: -sum(cuenta[v] for v in kv[1])):
    n = sum(cuenta[v] for v in vs)
    ordenados = sorted(vs, key=lambda v: -cuenta[v])
    print(f"  {nombre[c]:<10} n={n:<5} ({len(vs)} verbos)  {', '.join(ordenados[:12])}")

# reparto de clases por familia de rol: es lo que usara el planificador
fam_clase = collections.Counter((f, mapa[v]) for f, v in datos if f and v in mapa)
fams = sorted({f for f, _ in datos if f})
print("\nP(clase | familia de rol), que es la matriz que usara el planificador:")
cond = {}
for f in fams:
    tot = sum(n for (ff, _), n in fam_clase.items() if ff == f)
    fila = {nombre[c]: round(fam_clase.get((f, nombre[c]), 0) / max(1, tot), 3)
            for c in sorted(grupos)}
    cond[f] = fila
    top = sorted(fila.items(), key=lambda kv: -kv[1])[:4]
    print(f"  {f:<11} " + ", ".join(f"{k} {100*v:.0f}%" for k, v in top))

json.dump(dict(k=K, min_n=MIN_N, clases=sorted(nombre.values()), verbo_clase=mapa,
               por_familia=cond, n=len(datos)),
          open(f"{BASE}/data/processed/clases_funcion.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n-> data/processed/clases_funcion.json ({len(mapa)} verbos en {K} clases)")
