"""Pre-etiquetado de los backlogs de calibracion con la rubrica (data/calibracion/RUBRICA.md).

Es una APROXIMACION para que una persona revise, no la etiqueta final. Cada criterio se
calcula como sigue, de mas fiable a menos:

  C1 funciones   cada funcion nombrada se busca entre las historias con el SBERT propio;
                 cubierta si alguna historia supera UMBRAL_FUNCION. La principal es la
                 primera de la lista.
  C2 actores     NO se puede automatizar bien, y no se finge. Medido el 07/11/2026 sobre
                 30 pares rol-descripcion ya juzgados: los roles ajenos que importan son
                 VECINOS del dominio («instructor» en reserva de salas se parece 0,69 a
                 «alumno») y un legitimo puede parecerse menos («reclutador», 0,44). El
                 mejor umbral acierta el 83 % pero deja escapar 5 de 9 ajenos. Por eso
                 solo se marca ajeno lo inequivoco (umbrales sin falsos ajenos en esos 30
                 pares) y la lista de roles va a la hoja de revision para juicio humano.
  C3 deriva      historia cuya relevancia a la descripcion queda bajo UMBRAL_DERIVA. Es
                 la aproximacion mas gruesa: una historia legitima que la descripcion no
                 nombra puede caer aqui.
  C4 tareas      tareas que el filtro del pipeline marcaria y restos sin traducir. El
                 pipeline ya quita las obvias, asi que esto subestima; lo afina la revision.

Los grupos vaga e invalida no se etiquetan con la rubrica (su respuesta correcta es no
generar), salvo que el sistema haya generado: entonces son «mal».

  python src/95_preetiquetar.py data/calibracion/salidas_v4.json data/calibracion/etiquetas_pre_v4.csv

out: el csv que se le pase, con una fila por backlog
"""
import csv
import importlib.util
import json
import os
import re
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
UMBRAL_FUNCION = float(os.environ.get("AGIX_U_FUNCION", 0.55))
UMBRAL_ACTOR = float(os.environ.get("AGIX_U_ACTOR", 0.60))      # calibrados el 07/11
UMBRAL_ROL_DESC_SEGURO = float(os.environ.get("AGIX_U_ROL_DESC", 0.22))
# 0,22 y no 0,30: leidas 20 historias marcadas al azar en backlogs con buen plan, con 0,30
# solo la mitad eran deriva real («scooter» en una app de taxis); por debajo de 0,22, 5 de 6
UMBRAL_DERIVA = float(os.environ.get("AGIX_U_DERIVA", 0.22))
GENERICOS = ["administrador", "usuario", "gerente", "agente de soporte", "desarrollador"]
# Roles de gestion: encajan sin forzar en casi cualquier negocio, asi que no son «ajenos»
# aunque la descripcion no los nombre (rubrica, C2). Sin esto, «dueno de salon» en una
# cadena de peluquerias salia como rol ajeno.
GESTION = ("dueño", "dueña", "propietario", "propietaria", "gerente", "administrador",
           "administradora", "coordinador", "coordinadora", "supervisor", "supervisora",
           "jefe", "jefa", "encargado", "encargada", "director", "directora")


spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(int(os.environ.get("AGIX_HILOS", 4)))
sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1",
                         device=os.environ.get("AGIX_DEVICE", "cpu"))
enc = lambda xs: sb.encode(xs, normalize_embeddings=True, show_progress_bar=False)


def capacidad_es(h):
    """«Como X, quiero Y, para Z» -> Y; si no tiene esa forma, la historia entera"""
    m = re.search("quiero (.+?)(?:,? para |$)", h or "", re.I)
    return m.group(1) if m else (h or "")


def rol_es(h):
    m = re.match("como (?:un |una |el |la )?(.+?),", h or "", re.I)
    return m.group(1).strip() if m else ""


def etiqueta(c1p, c1k, c1n, c2, c3, c4m, c4t):
    """gana el peor criterio (rubrica)"""
    frac = c1k / c1n if c1n else 1.0
    malas = c4m / c4t if c4t else 0.0
    if not c1p or frac < 0.5 or c2 > 3 or c3 > 4 or malas > 0.20:
        return "mal"
    if frac < 0.8 or c2 > 1 or c3 > 2 or malas > 0.10:
        return "regular"
    return "bien"


J = json.load(open(sys.argv[1], encoding="utf-8"))
filas = []
for i, (k, v) in enumerate(J.items()):
    H = v["historias"]
    hist = [h["es"] or h["en"] for h in H]
    if v["grupo"] in ("vaga", "invalida"):
        filas.append(dict(id=k, grupo=v["grupo"], c1_principal="", c1_cubiertas="", c1_total="",
                          c2_ajenos="", c3_deriva="", c4_malas="", c4_total="",
                          etiqueta="mal" if H else "correcto_no_genera",
                          nota=f"genero {len(H)} historias" if H else ""))
        continue
    caps = [capacidad_es(h) for h in hist]
    Ec = enc(caps)
    # C1
    fun = v["funciones"]
    Ef = enc(fun)
    S = Ef @ Ec.T
    cub = (S.max(1) >= UMBRAL_FUNCION)
    c1p, c1k, c1n = bool(cub[0]), int(cub.sum()), len(fun)
    # C2
    roles = [rol_es(h) for h in hist]
    Er = enc([r or "-" for r in roles])
    Ea = enc(v["actores"] + GENERICOS)
    Ed = enc([v["texto"]])[0]
    ajeno = [((Er[j] @ Ea.T).max() < UMBRAL_ACTOR
              and float(Er[j] @ Ed) < UMBRAL_ROL_DESC_SEGURO
              and not roles[j].lower().startswith(GESTION))
             for j in range(len(roles))]
    c2 = int(sum(ajeno))
    # C3
    rel = enc(hist) @ Ed
    c3 = int((rel < UMBRAL_DERIVA).sum())
    # C4
    tareas = [(te, ts) for h in H for te, ts in zip(h["tareas_en"], h["tareas_es"] or h["tareas_en"])]
    malas = [ts for te, ts in tareas
             if pl.tarea_degenerada(te) or re.search("[a-z]+-[a-z]+ing|-[a-z]+ ?$", ts or "")]
    no_cub = [f for f, c in zip(fun, cub) if not c]
    nota = ["roles: " + ", ".join(sorted({r for r in roles if r}))]
    if no_cub:
        nota.append("sin cubrir: " + "; ".join(no_cub))
    if c2:
        ajenos = sorted({r for r, a_ in zip(roles, ajeno) if a_})
        nota.append("roles ajenos: " + ", ".join(ajenos))
    filas.append(dict(id=k, grupo=v["grupo"], c1_principal="si" if c1p else "no",
                      c1_cubiertas=c1k, c1_total=c1n, c2_ajenos=c2, c3_deriva=c3,
                      c4_malas=len(malas), c4_total=len(tareas),
                      etiqueta=etiqueta(c1p, c1k, c1n, c2, c3, len(malas), len(tareas)),
                      nota=" | ".join(nota)))

with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0]))
    w.writeheader()
    w.writerows(filas)
import collections  # noqa: E402
for g in ("taxonomia", "obligatorio", "fuera", "vaga", "invalida"):
    c = collections.Counter(x["etiqueta"] for x in filas if x["grupo"] == g)
    if c:
        print(f"  {g:<11} {dict(c)}")
print(f"escrito {sys.argv[2]}")
