"""Vocabulario propio de cada sector, para el paso 3 (restriccion dependiente de la entrada).

Para que sirve. Quedan fugas que ninguna regla anterior atrapa, porque no comparten
vocabulario con Jira ni se parecen poco a su historia: en un backlog de BODAS sale «la
hora estimada de llegada de mi PRENDA» (del sector de lavanderia) y en uno de LIMPIEZA
«el progreso de las citas de mi SER QUERIDO» (del sector de cuidados). El modelo importa
el vocabulario del sector entrenado mas parecido.

El paso 3 es una restriccion en la decodificacion que depende de la ENTRADA -- la
gramatica dependiente de la entrada de Geng et al. (EMNLP 2023) -- : para una descripcion
dada se vetan las palabras propias de OTROS sectores.

PRIMER INTENTO, DESCARTADO: sacar el lexico del texto generado por el profesor con un
criterio de razon de frecuencia (>= 8x mas frecuente en su sector). Metia palabras
genericas en las listas: «eventos» se quedaba con people, early, window, code, social,
media, someone, lists. Vetar eso en otros sectores seria peor que la fuga.

ESTE: el lexico sale de los RASGOS CURADOS de la taxonomia (src/41_dominios_sinteticos.py
y los 10 sectores nuevos de src/53_dominios_v3.py), que son los terminos de dominio
escritos a proposito («garment counting», «stain treatment notes», «caregiver matching»).
Y se queda solo con las palabras que aparecen en UN sector, que es lo que las hace
distintivas; asi caen booking, payments, notifications y compania.

  python src/86_lexico_sector.py

out: data/processed/lexico_sector.json
"""
import ast
import collections
import importlib.util
import io
import json
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SECTORES = 1        # una palabra vale si es propia de UN solo sector
PARADA = {"the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "with", "by",
          "from", "at", "as", "per", "that", "this", "it", "is", "are", "be", "can",
          "user", "users", "app", "data", "new", "system", "platform", "service",
          "management", "online", "web", "mobile", "digital"}


def sectores_curados():
    """(sector -> texto de sus rasgos y actores), de los dos ficheros de taxonomia"""
    out = collections.defaultdict(list)
    for fichero in ("41_dominios_sinteticos.py", "53_dominios_v3.py"):
        p = f"{BASE}/src/{fichero}"
        if not os.path.exists(p):
            continue
        arbol = ast.parse(io.open(p, encoding="utf-8").read())
        for n in arbol.body:
            if not isinstance(n, ast.Assign):
                continue
            nom = getattr(n.targets[0], "id", "")
            if nom not in ("SECTORES", "NUEVOS", "ROLES"):
                continue
            try:
                d = ast.literal_eval(n.value)
            except Exception:
                continue
            for sec, val in (d or {}).items():
                if isinstance(val, (list, tuple)) and val and isinstance(val[0], str) \
                        and nom == "ROLES":
                    out[sec] += list(val)
                elif isinstance(val, (list, tuple)) and len(val) == 3:
                    _desc, roles, rasgos = val
                    out[sec] += list(roles) + list(rasgos)
    return out


crudo = sectores_curados()
print(f"{len(crudo)} sectores con rasgos curados", flush=True)

palabras = {}
for sec, textos in crudo.items():
    ws = set()
    for t in textos:
        for w in re.findall(r"[a-z]{4,}", str(t).lower()):
            if w not in PARADA:
                ws.add(w)
    palabras[sec] = ws

cuantos = collections.Counter()
for ws in palabras.values():
    for w in ws:
        cuantos[w] += 1

propio = {s: sorted(w for w in ws if cuantos[w] <= MAX_SECTORES)
          for s, ws in palabras.items()}
descartadas = sorted((w for w, n in cuantos.items() if n > MAX_SECTORES),
                     key=lambda w: -cuantos[w])

json.dump(dict(max_sectores=MAX_SECTORES, sectores=sorted(propio), propio=propio),
          open(f"{BASE}/data/processed/lexico_sector.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

tam = sorted(len(v) for v in propio.values())
print(f"palabras distintivas por sector: mediana {tam[len(tam)//2]}, min {tam[0]}, max {tam[-1]}")
print(f"descartadas por aparecer en varios sectores ({len(descartadas)}), las 16 mas comunes:")
print("  " + ", ".join(f"{w} ({cuantos[w]})" for w in descartadas[:16]))
for s in ["lavanderia", "cuidado", "eventos", "banca", "farmacia", "restaurantes"]:
    if s in propio:
        print(f"  {s:<12} {len(propio[s]):>3}: {', '.join(propio[s][:14])}")
print("\n-> data/processed/lexico_sector.json")
