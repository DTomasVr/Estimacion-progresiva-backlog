"""Lexico de roles validos para el hueco "As a <rol>" de la historia.

El defecto que resuelve: la mitad de arriba aprendio la plantilla pero rellena
el hueco del rol con cualquier sustantivo de la descripcion, y salen historias
como "As a service appointment, I want to schedule my first day of repairs".
El sustantivo es del dominio, pero no es una PERSONA.

El lexico se construye de las tres fuentes de entrenamiento, no a mano:
  - el plantel de actores de la taxonomia v3 (src/53_dominios_v3.py)
  - los roles que aparecen en las historias reales del corpus (Jira + Dalpiaz)
  - los roles de la tanda v1 del profesor
Se acepta un rol generado si esta en el lexico o si su NUCLEO (la ultima
palabra) es un nucleo de persona visto >= 2 veces en el lexico. Asi
"membership manager" pasa aunque no este listado, y "service appointment" no.

La lista de nucleos se revisa a mano: es la unica parte curada y hay que poder
defenderla, asi que se imprime entera al final.

out: data/processed/roles_lexico.json
"""
import ast
import collections
import json
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_ROL = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,", re.I)


def norm(r):
    r = (r or "").strip().lower().strip(".")
    r = re.sub(r"^(a|an|the)\s+", "", r)
    r = re.sub(r"[^a-z\s'-]", " ", r)
    return re.sub(r"\s+", " ", r).strip()


roles = collections.Counter()
fuentes = collections.Counter()

# 1. plantel de la taxonomia v3
_src = open(f"{BASE}/src/53_dominios_v3.py", encoding="utf-8").read()
ROLES_V3 = next(ast.literal_eval(n.value) for n in ast.parse(_src).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ROLES")
for v in ROLES_V3.values():
    for r in v:
        roles[norm(r)] += 5          # curados a mano: pesan mas
        fuentes["taxonomia"] += 1

# 2. historias reales del corpus (objetivos en forma de historia) + Dalpiaz
for f, col in [("s2s_o2ofmt_train", "objetivo"), ("s2s_o2ofmt_val", "objetivo"),
               ("s2s_o2ofmt_test", "objetivo"), ("s2s_o2o_dalpiaz_test", None)]:
    p = f"{BASE}/data/processed/{f}.parquet"
    if not os.path.exists(p):
        continue
    d = pd.read_parquet(p)
    c = col if col in d.columns else next((x for x in ("objetivo", "titulo", "historia")
                                           if x in d.columns), None)
    if c is None:
        continue
    for t in d[c].astype(str):
        m = RE_ROL.match(t)
        if m:
            roles[norm(m.group(1))] += 1
            fuentes["corpus_real"] += 1

# 3. tanda v1 del profesor
p = f"{BASE}/data/synth/backlogs_qwen2.5_7b.jsonl"
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        for h in json.loads(l)["historias"]:
            m = RE_ROL.match(h["historia"])
            if m:
                roles[norm(m.group(1))] += 1
                fuentes["profesor_v1"] += 1

roles.pop("", None)

# --- nucleos de persona -----------------------------------------------------
# candidatos = ultima palabra de cada rol, con su frecuencia
cab = collections.Counter()
for r, n in roles.items():
    cab[r.split()[-1]] += n

# Nucleos que NO son personas y aparecen porque el corpus de Jira trae ruido
# ("as a result", "as a consequence") o porque el profesor se equivoco.
NO_PERSONA = {
    "result", "consequence", "rule", "matter", "part", "service", "system", "app",
    "platform", "tool", "site", "page", "process", "feature", "test", "example",
    "note", "reference", "side", "means", "whole", "workaround", "bonus", "aside",
    "consumer1", "etc", "value", "type", "case", "step", "way", "thing", "task",
    "story", "appointment", "booking", "order", "request", "report", "record",
    "account", "profile", "dashboard", "workflow", "pipeline", "library", "module",
    "api", "ui", "endpoint", "database", "server", "client-side", "backend",
    "frontend", "plugin", "package", "project", "product", "company", "business",
    "organisation", "organization", "team", "group", "department", "clinic",
    "restaurant", "hotel", "gym", "store", "shop", "school", "bank", "pharmacy",
    "salon", "cinema", "warehouse", "office", "site-wide", "vehicle", "device",
    # ruido del corpus de Jira que se cuela como nucleo
    "instance", "node", "course", "simple", "simulation", "integrity", "archive",
    "notifications", "osgi", "qstringbuilder", "participate", "peers", "means",
    "agency", "univitservice", "recyclingfacility", "mike", "indy", "fundingbody",
    "csm", "cst", "dpc", "sme", "pi", "launcher", "optimiser", "optimizer",
    "deployer", "tracker", "scheduler", "selector", "processor", "responder",
}
MIN_CAB = 2
cabezas = sorted(c for c, n in cab.items()
                 if n >= MIN_CAB and c not in NO_PERSONA and len(c) >= 3)

# un rol con nucleo descartado sale tambien del lexico literal
lexico = sorted(r for r in roles if r.split()[-1] in set(cabezas))

json.dump(dict(roles=lexico, cabezas=cabezas, no_persona=sorted(NO_PERSONA)),
          open(f"{BASE}/data/processed/roles_lexico.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

print(f"roles recogidos {len(roles)} de {sum(fuentes.values())} menciones  {dict(fuentes)}")
print(f"lexico final: {len(lexico)} roles, {len(cabezas)} nucleos de persona")
print(f"descartados por nucleo: {len(roles)-len(lexico)} roles")
print("\nnucleos (revisar a mano):")
for i in range(0, len(cabezas), 10):
    print("  " + ", ".join(cabezas[i:i + 10]))
print("\nejemplos de roles DESCARTADOS:")
desc = [r for r in roles if r.split()[-1] not in set(cabezas)]
for r in sorted(desc, key=lambda r: -roles[r])[:25]:
    print(f"  {roles[r]:5d}  {r}")
