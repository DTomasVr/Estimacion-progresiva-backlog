"""Matriz verbo -> familia de rol, sacada de los dos corpus con rol explicito.

Para que sirve. La historia «Como invitado, quiero ver la comida, para poder planificar
mi CARGA DE TRABAJO» esta mal porque un invitado no tiene carga de trabajo: el verbo y
el rol no pegan. Medido en src/82_rol_funcion.py, el rol predice el tipo de funcion con
V de Cramer 0,399 en las historias humanas y 0,382 en las del profesor, asi que la
senal existe y se puede usar para castigar candidatas en la seleccion.

Lo que se guarda NO es mi opinion sobre que puede pedir cada rol, es la distribucion
observada: para cada verbo, en que proporcion aparece con cada familia de rol. La
familia sigue siendo un agrupamiento por nucleo del rol (eso si es una decision), pero
las probabilidades salen de 5.887 historias reales.

  python src/83_matriz_rol_verbo.py

out: data/processed/matriz_rol_verbo.json
"""
import collections
import importlib.util
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_VERBO = 5          # un verbo con menos apariciones no dice nada

spec = importlib.util.spec_from_file_location("m82", f"{BASE}/src/82_rol_funcion.py")
# 82 imprime y escribe al importarse; se cargan solo las piezas que hacen falta
import ast  # noqa: E402
import io   # noqa: E402
import re   # noqa: E402

src82 = io.open(f"{BASE}/src/82_rol_funcion.py", encoding="utf-8").read()
mod = ast.parse(src82)
trozos = [n for n in mod.body
          if isinstance(n, (ast.FunctionDef, ast.Assign, ast.Import, ast.ImportFrom))]
ns = {"__file__": f"{BASE}/src/82_rol_funcion.py"}
exec(compile(ast.Module(body=trozos, type_ignores=[]), "82", "exec"), ns)

pares = ns["pares_dalpiaz"]() + ns["pares_profesor"]()
datos = [(ns["familia"](r), ns["verbo"](m)) for r, m in pares]
datos = [(f, v) for f, v in datos if f and v]
print(f"{len(pares)} historias con rol, {len(datos)} con familia y verbo")

cuenta = collections.Counter(datos)
por_verbo = collections.Counter(v for _, v in datos)
familias = sorted({f for f, _ in datos})

tabla = {}
for v, n in por_verbo.items():
    if n < MIN_VERBO:
        continue
    fila = {f: cuenta[(f, v)] / n for f in familias}
    tabla[v] = {f: round(p, 4) for f, p in fila.items()}

# base: cuanta masa tiene cada familia en total, para no castigar a la familia rara
base = collections.Counter(f for f, _ in datos)
tot = sum(base.values())
prior = {f: round(base[f] / tot, 4) for f in familias}

json.dump(dict(familias=familias, prior=prior, verbos=tabla,
               nucleos=ns["NUCLEO"], min_verbo=MIN_VERBO, n=len(datos)),
          open(f"{BASE}/data/processed/matriz_rol_verbo.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"familias: {prior}")
print(f"verbos con al menos {MIN_VERBO} apariciones: {len(tabla)}")
print("\nlos 12 verbos mas desequilibrados (los que mejor discriminan):")
orden = sorted(tabla.items(), key=lambda kv: -max(kv[1].values()))
for v, fila in orden[:12]:
    f = max(fila, key=fila.get)
    print(f"  {v:<14} {100*fila[f]:5.1f} % {f:<11} (n={por_verbo[v]})")
print("\nlos 8 mas repartidos (los que no discriminan y no deberian castigar):")
for v, fila in orden[-8:]:
    print(f"  {v:<14} " + "  ".join(f"{f[:6]} {100*p:.0f}%" for f, p in sorted(fila.items())))
print("\n-> data/processed/matriz_rol_verbo.json")
