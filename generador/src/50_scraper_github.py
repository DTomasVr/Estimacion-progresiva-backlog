"""Mine REAL parent -> task decompositions from consumer-product open source repos.

Why GitHub: the literature survey (Motger et al. 2025) finds user stories are only
4.8 % of requirements-engineering datasets and none carry sub-tasks, so the data we
need does not exist as a published corpus. But it exists in the open: a GitHub issue
whose body holds a task list is exactly a parent with its decomposition, written by
people, in product domains our Jira corpus lacks.

Conventions followed (GitHub REST API docs, "Best practices for using the REST API"):
  * a descriptive User-Agent, as the API requires
  * authenticated when GITHUB_TOKEN is set (5000 req/h), otherwise 60 req/h
  * the rate limit is READ from the response headers and waited out, never guessed
  * secondary-rate-limit and 403/429 responses honour Retry-After, then back off
  * one request at a time, a pause between them, no concurrency
  * ETag conditional requests, so a re-run costs nothing for unchanged pages
No attempt is made to evade limits or blocks: the token is the supported way to go
faster, and evading would breach the terms of service.

  python src/50_scraper_github.py [MAX_REPOS]

out: data/reales/github_issues.jsonl   (one line per issue that has a task list)
     data/reales/github_ritmo.csv      (throughput measurements)
"""
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/data/reales", exist_ok=True)
SALIDA = f"{BASE}/data/reales/github_issues.jsonl"
RITMO = f"{BASE}/data/reales/github_ritmo.csv"
ETAGS = f"{BASE}/data/reales/github_etags.json"
ARCHIVO_TOKEN = f"{BASE}/.github_token"


def leer_token():
    """GITHUB_TOKEN, or a one-line file the user can drop in WHILE this runs."""
    t = os.environ.get("GITHUB_TOKEN", "").strip()
    if not t and os.path.exists(ARCHIVO_TOKEN):
        t = open(ARCHIVO_TOKEN, encoding="utf-8").read().strip()
    return t


TOKEN = leer_token()
PAUSA = 1.0 if TOKEN else 2.0          # cortesia entre peticiones
PAGINAS = int(os.environ.get("PBE_PAGINAS", 5))   # 100 incidencias por pagina

# consumer/product repos: the domains the Jira corpus does not have.
REPOS = [
    # e-commerce y marketplace
    "saleor/saleor", "medusajs/medusa", "vendure-ecommerce/vendure", "spree/spree",
    "sylius/Sylius", "woocommerce/woocommerce", "shopware/shopware", "bagisto/bagisto",
    "PrestaShop/PrestaShop", "solidusio/solidus",
    # libros, audio y media
    "advplyr/audiobookshelf", "Kareadita/Kavita", "janeczku/calibre-web", "gotson/komga",
    "jellyfin/jellyfin", "Lidarr/Lidarr", "Readarr/Readarr", "navidrome/navidrome",
    # reservas, citas y eventos
    "calcom/cal.com", "alextselegidis/easyappointments", "pretix/pretix", "hi-ventures/booking",
    # salud, educacion y comunidad
    "openemr/openemr", "moodle/moodle", "chatwoot/chatwoot", "mattermost/mattermost",
    "nextcloud/server", "mastodon/mastodon", "discourse/discourse", "rocketchat/Rocket.Chat",
    # finanzas personales y negocio
    "firefly-iii/firefly-iii", "actualbudget/actual", "invoiceninja/invoiceninja",
    "frappe/erpnext", "odoo/odoo", "dolibarr/dolibarr", "akaunting/akaunting",
    # logistica, inmuebles, viajes
    "grocy/grocy", "dividab/tsconfig-paths", "opentripplanner/OpenTripPlanner",
    "traccar/traccar", "photoprism/photoprism", "immich-app/immich",
    "documenso/documenso", "twentyhq/twenty", "formbricks/formbricks",
]

# "- [ ] do the thing"  /  "* [x] do the thing"
TAREA = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s+(.{6,200})$", re.M)
REF_ISSUE = re.compile(r"^#\d+\s*$|^https?://github\.com/\S+$")
RUIDO = re.compile(r"^\s*(todo|tbd|n/?a|\.\.\.|etc\.?)\s*$", re.I)

ses = requests.Session()
ses.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "pbe-thesis-backlog-research/1.0 (academic use; contact via repo)",
})
if TOKEN:
    ses.headers["Authorization"] = f"Bearer {TOKEN}"
    print("autenticado: 5000 peticiones/hora")
else:
    print("SIN token: 60 peticiones/hora. Define GITHUB_TOKEN para multiplicar por 80.")

etags = json.load(open(ETAGS, encoding="utf-8")) if os.path.exists(ETAGS) else {}
peticiones = {"n": 0, "espera_s": 0.0, "no_modificado": 0}


def pedir(url, params=None):
    """one request, honouring the documented rate limits"""
    for intento in range(6):
        cab = {}
        clave = url + json.dumps(params or {}, sort_keys=True)
        if clave in etags:
            cab["If-None-Match"] = etags[clave]
        r = ses.get(url, params=params, headers=cab, timeout=60)
        peticiones["n"] += 1
        if r.status_code == 304:
            peticiones["no_modificado"] += 1
            return None
        if r.status_code == 200:
            if "ETag" in r.headers:
                etags[clave] = r.headers["ETag"]
            queda = int(r.headers.get("X-RateLimit-Remaining", 1))
            if queda <= 1:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                espera = max(5, reset - time.time() + 3)
                print(f"    limite agotado, esperando {espera/60:.1f} min", flush=True)
                peticiones["espera_s"] += espera
                time.sleep(espera)
            return r.json()
        if r.status_code in (403, 429):
            ra = r.headers.get("Retry-After")
            if ra:
                espera = int(ra) + 2
            else:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                espera = max(60, reset - time.time() + 3)
            print(f"    {r.status_code}: esperando {espera/60:.1f} min", flush=True)
            peticiones["espera_s"] += espera
            time.sleep(espera)
            continue
        if r.status_code == 404:
            return []
        print(f"    HTTP {r.status_code} en {url}", flush=True)
        time.sleep(5 * (intento + 1))
    return []


def tareas_de(cuerpo):
    if not cuerpo:
        return []
    out = []
    for m in TAREA.finditer(cuerpo):
        t = re.sub(r"\s+", " ", m.group(1)).strip(" .")
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)        # [texto](url) -> texto
        t = re.sub(r"[`*_~]", "", t).strip()
        if len(t) >= 8 and not REF_ISSUE.match(t) and not RUIDO.match(t):
            out.append(t)
    return out


hechos = set()
if os.path.exists(SALIDA):
    hechos = {json.loads(l)["uid"] for l in open(SALIDA, encoding="utf-8")}
print(f"{len(hechos)} incidencias ya guardadas")

max_repos = int(sys.argv[1]) if len(sys.argv) > 1 else len(REPOS)
t0 = time.time()
nuevos = 0
filas_ritmo = []
with open(SALIDA, "a", encoding="utf-8") as fh:
    for repo in REPOS[:max_repos]:
        # el token puede aparecer a mitad de la corrida: se recoge sin reiniciar
        if not TOKEN:
            t_nuevo = leer_token()
            if t_nuevo:
                TOKEN = t_nuevo
                ses.headers["Authorization"] = f"Bearer {TOKEN}"
                PAUSA = 1.0
                print("    token detectado: 5000 peticiones/hora desde ahora", flush=True)
        t_repo = time.time()
        n_repo = 0
        for pagina in range(1, PAGINAS + 1):             # PAGINAS x 100 incidencias por repo
            datos = pedir(f"https://api.github.com/repos/{repo}/issues",
                          {"state": "all", "per_page": 100, "page": pagina,
                           "sort": "created", "direction": "desc"})
            time.sleep(PAUSA)
            if datos is None:                            # 304: sin cambios
                break
            if not datos:
                break
            for it in datos:
                if "pull_request" in it:                 # los PR no son descomposiciones
                    continue
                uid = f"{repo}#{it['number']}"
                if uid in hechos:
                    continue
                tareas = tareas_de(it.get("body") or "")
                if len(tareas) < 2:
                    continue
                fh.write(json.dumps(dict(
                    uid=uid, repo=repo, numero=it["number"], titulo=it["title"],
                    cuerpo=(it.get("body") or "")[:4000], tareas=tareas,
                    etiquetas=[l["name"] for l in it.get("labels", [])],
                    creado=it.get("created_at")), ensure_ascii=False) + "\n")
                fh.flush()
                hechos.add(uid)
                nuevos += 1
                n_repo += 1
            if len(datos) < 100:
                break
        dt = time.time() - t_repo
        filas_ritmo.append(dict(repo=repo, incidencias_utiles=n_repo, segundos=round(dt, 1),
                                peticiones=peticiones["n"], espera_min=round(peticiones["espera_s"]/60, 1)))
        print(f"  {repo:<42} {n_repo:4d} con lista de tareas  {dt/60:5.1f} min  "
              f"({peticiones['n']} peticiones, {peticiones['espera_s']/60:.0f} min esperando)", flush=True)
        json.dump(etags, open(ETAGS, "w", encoding="utf-8"))
        import pandas as pd
        pd.DataFrame(filas_ritmo).to_csv(RITMO, index=False)

dt = time.time() - t0
tot_tareas = sum(len(json.loads(l)["tareas"]) for l in open(SALIDA, encoding="utf-8"))
print(f"\n{nuevos} incidencias nuevas con descomposicion en {dt/60:.1f} min")
print(f"total acumulado: {len(hechos)} incidencias, {tot_tareas} tareas")
print(f"peticiones: {peticiones['n']} ({peticiones['no_modificado']} sin cambios), "
      f"{peticiones['espera_s']/60:.0f} min esperando el limite")
if nuevos:
    print(f"ritmo: {nuevos/(dt/3600):.0f} incidencias utiles por hora"
          f"{' (con token seria ~80x mas)' if not TOKEN else ''}")
