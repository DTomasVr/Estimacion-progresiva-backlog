"""Busca en GitHub issues que SEAN una historia de usuario y tengan subtareas.

Por que se reintenta. El primer intento (src/50_scraper_github.py y
src/51_filtrar_github.py) rastreaba repositorios y filtraba despues: sobrevivio el
1,4 % y cero items de producto, porque los repos publicos descomponen bugs en
commits, no producto en tareas. Los sub-issues eran el 0,2 %.

Que cambia aqui. En lugar de filtrar por repositorio, se filtra por FORMA: se
piden directamente issues cuyo titulo empiece por "As a ..., I want ...", que es
la forma de la historia de usuario, y se exige que el cuerpo traiga una lista de
tareas ("- [ ]"). Si la cosecha sigue siendo residual, la via queda cerrada con
numero y no por intuicion.

Se usa la API oficial con token personal, respetando los limites que devuelve la
propia API. Nada de evadir limites.

  python src/72_issues_historia.py

out: data/raw/github_historias.jsonl  y  results/72_issues.csv
"""
import json
import os
import re
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = open(f"{BASE}/.github_token", encoding="utf-8").read().strip()
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "X-GitHub-Api-Version": "2022-11-28"}
FORMA = re.compile(r"^\s*as\s+(?:an?|the)\s+[^,]{2,40},\s*i\s+(?:want|would like|need)\b", re.I)
EMPIEZA = re.compile(r"^\s*as\s+(?:an?|the)\s+\w", re.I)
TAREA = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*(.+)$", re.M)

CONSULTAS = [
    '"As a" "I want" in:title is:issue',
    '"As a user I want" in:title is:issue',
    '"As a" in:title is:issue label:"user story"',
    '"As a" in:title is:issue label:story',
    '"As a" in:title is:issue label:epic',
    '"As a" "so that" in:title is:issue',
    '"As a" in:title is:issue language:javascript',
    '"As a" in:title is:issue language:python',
    '"As a" in:title is:issue language:java',
    '"As a" in:title is:issue is:closed',
]


def buscar(q, paginas=3):
    out = []
    for p in range(1, paginas + 1):
        r = requests.get("https://api.github.com/search/issues", headers=H,
                         params={"q": q, "per_page": 100, "page": p,
                                 "advanced_search": "true"}, timeout=60)
        if r.status_code == 403:
            espera = max(5, int(r.headers.get("x-ratelimit-reset", 0)) - int(time.time()))
            print(f"  limite alcanzado, espero {espera}s", flush=True)
            time.sleep(min(espera + 2, 90))
            continue
        if r.status_code != 200:
            print(f"  {r.status_code} en '{q[:40]}'", flush=True)
            break
        d = r.json()
        out += d.get("items", [])
        if len(d.get("items", [])) < 100:
            break
        time.sleep(2.2)                    # 30 busquedas por minuto
    return out


vistos, filas = set(), []
for q in CONSULTAS:
    items = buscar(q)
    nuevos = 0
    for it in items:
        u = it.get("html_url")
        if not u or u in vistos or "pull" in it:
            continue
        vistos.add(u)
        nuevos += 1
        titulo = (it.get("title") or "").strip()
        cuerpo = it.get("body") or ""
        tareas = [t.strip() for t in TAREA.findall(cuerpo)]
        filas.append(dict(url=u, titulo=titulo, empieza=bool(EMPIEZA.match(titulo)),
                          forma=bool(FORMA.match(titulo)), n_tareas=len(tareas),
                          tareas=tareas[:12], repo=u.split("/")[3] + "/" + u.split("/")[4]))
    print(f"{len(items):4d} resultados, {nuevos:4d} nuevos | {q[:56]}", flush=True)

os.makedirs(f"{BASE}/data/raw", exist_ok=True)
with open(f"{BASE}/data/raw/github_historias.jsonl", "w", encoding="utf-8") as fh:
    for f in filas:
        fh.write(json.dumps(f, ensure_ascii=False) + "\n")

import pandas as pd  # noqa: E402
D = pd.DataFrame(filas)
D.drop(columns=["tareas"]).to_csv(f"{BASE}/results/72_issues.csv", index=False)
n = len(D)
print("\n" + "=" * 68)
print(f"issues distintos            : {n}")
if n:
    print(f"empiezan por 'As a ...'     : {D.empieza.sum()} ({100*D.empieza.mean():.1f} %)")
    print(f"forma completa de historia  : {D.forma.sum()} ({100*D.forma.mean():.1f} %)")
    con = D[(D.forma) & (D.n_tareas >= 2)]
    print(f"forma + >=2 subtareas       : {len(con)} ({100*len(con)/n:.1f} %)")
    print(f"repositorios distintos      : {D.repo.nunique()}")
    if len(con):
        print(f"tareas por historia utiles  : mediana {con.n_tareas.median():.0f}, "
              f"total {int(con.n_tareas.sum())}")
        print("\nejemplos utiles:")
        for _, x in con.head(5).iterrows():
            print(f"\n  [{x.repo}] {x.titulo[:110]}")
            for t in x.tareas[:4]:
                print(f"      - {t[:96]}")
print("-> data/raw/github_historias.jsonl y results/72_issues.csv")
