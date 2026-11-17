"""Widen the sector taxonomy, because the reading of the first pilot showed WHERE
the gain comes from: sectors that were trained improved a lot, sectors that were
not fell back to the nearest trained neighbour (parking -> delivery, weddings ->
hotels). So breadth beats depth: many sectors with few descriptions each.

The 30 hand-written sectors stay; the rest are asked from the local LLM, parsed,
and de-duplicated against each other with the project's SBERT.

  python src/47_taxonomia_amplia.py [N_SECTORES] [POR_SECTOR]

out: data/synth/descripciones.json  (overwrites; the previous one is backed up)
"""
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import sys

import numpy as np
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260922
random.seed(SEED)
N_SECTORES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
POR_SECTOR = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MODELO = "qwen2.5:7b"
URL = "http://127.0.0.1:11434/api/generate"

# no ejecutamos 41 (tiene efectos secundarios): solo queremos su diccionario
texto41 = open(f"{BASE}/src/41_dominios_sinteticos.py", encoding="utf-8").read()
ns = {}
exec(texto41[texto41.index("SECTORES = {"):texto41.index("PLANTILLAS = [")], ns)
SECTORES = dict(ns["SECTORES"])
print(f"{len(SECTORES)} sectores escritos a mano")

PROMPT = """List {n} DIFFERENT kinds of software product, each in a different industry or niche.
Avoid developer tools, infrastructure and anything a software engineer builds for other engineers:
these must be products for ordinary people or for a specific trade.
Already covered, do not repeat: {ya}.

One per line, in this exact format, nothing else:
NAME | what the product is | who uses it (2-3 roles, comma separated) | 6 things it must do (comma separated)

Example:
dry cleaning | a laundry pickup service | customers, drivers, shop staff | order pickup, garment tagging, stain notes, delivery slots, subscription plans, damage claims
"""


def pedir(n, ya):
    r = requests.post(URL, json={"model": MODELO, "prompt": PROMPT.format(n=n, ya=", ".join(ya)),
                                 "stream": False,
                                 "options": {"temperature": 0.9, "num_predict": 2400}}, timeout=600)
    r.raise_for_status()
    return r.json()["response"]


def parsear(txt):
    out = {}
    for linea in txt.split("\n"):
        partes = [p.strip() for p in linea.split("|")]
        if len(partes) != 4:
            continue
        nombre, que, quien, feats = partes
        nombre = re.sub(r"^[\d\.\-\*\s]+", "", nombre).lower()
        roles = [x.strip() for x in quien.split(",") if len(x.strip()) > 2]
        fs = [x.strip() for x in feats.split(",") if len(x.strip()) > 4]
        if len(nombre) > 2 and len(roles) >= 2 and len(fs) >= 4 and not que.lower().startswith("a "):
            que = "a " + que if not que.lower().startswith(("a ", "an ")) else que
        if len(nombre) > 2 and len(roles) >= 2 and len(fs) >= 4:
            out[nombre] = (que, roles, fs)
    return out


faltan = N_SECTORES - len(SECTORES)
intentos = 0
while len(SECTORES) < N_SECTORES and intentos < 30:
    intentos += 1
    ya = random.sample(list(SECTORES), min(40, len(SECTORES)))
    try:
        nuevos = parsear(pedir(25, ya))
    except Exception as e:
        print(f"  reintento ({type(e).__name__})", flush=True)
        continue
    antes = len(SECTORES)
    for k, v in nuevos.items():
        if k not in SECTORES:
            SECTORES[k] = v
    print(f"  lote {intentos}: +{len(SECTORES)-antes} sectores (total {len(SECTORES)})", flush=True)

print(f"{len(SECTORES)} sectores tras {intentos} lotes")

PLANTILLAS = [
    "{articulo} for {publico}, with {f1} and {f2}.",
    "{articulo} where {publico} can {verbo} {f1}, plus {f2} and {f3}.",
    "A platform that offers {f1}, {f2} and {f3}, used by {publico}.",
    "{articulo} focused on {f1}, with {f2} for {publico}.",
    "{articulo}. The main needs are {f1}, {f2} and {f3}.",
    "We are building {articulo_min}. It must support {f1} and {f2}, mainly for {publico}.",
    "{articulo} that lets {publico} {verbo} {f1}. It also needs {f2}.",
    "A mobile app for {publico}: {f1}, {f2} and {f3}.",
    "A web system for {publico} to {verbo} {f1}, including {f2}.",
    "{articulo} with {f1}. {publico} also need {f2} and {f3}.",
    "An internal tool for {publico}, covering {f1} and {f2}.",
]
VERBOS = ["manage", "handle", "set up", "keep track of", "organise"]

casos = []
for sector, (articulo, publicos, feats) in SECTORES.items():
    combos = list(itertools.combinations(feats, 3))
    random.shuffle(combos)
    art = articulo[0].upper() + articulo[1:]
    for i in range(POR_SECTOR):
        f1, f2, f3 = combos[i % len(combos)]
        txt = PLANTILLAS[(i * 3 + len(sector)) % len(PLANTILLAS)].format(
            articulo=art, articulo_min=articulo, publico=random.choice(publicos),
            verbo=random.choice(VERBOS), f1=f1, f2=f2, f3=f3)
        # id por contenido: si la descripcion cambia, el backlog viejo no se reusa por error
        h = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:8]
        casos.append(dict(id=f"{sector.replace(' ', '_')}-{h}", sector=sector, descripcion=txt))

from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(f"{BASE}/models/sbert_agix_v1", device="cpu")
E = sb.encode([c["descripcion"] for c in casos], convert_to_tensor=True, batch_size=128,
              normalize_embeddings=True, show_progress_bar=False)
S = (E @ E.T).float().cpu().numpy()
np.fill_diagonal(S, 0)
vistos, quedan = np.zeros(len(casos), bool), []
for i in range(len(casos)):
    if vistos[i]:
        continue
    vistos |= (S[i] >= 0.93)
    quedan.append(casos[i])

ruta = f"{BASE}/data/synth/descripciones.json"
if os.path.exists(ruta):
    shutil.copy(ruta, f"{BASE}/data/synth/descripciones_v1.json")
json.dump(quedan, open(ruta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump({k: list(v) for k, v in SECTORES.items()},
          open(f"{BASE}/data/synth/sectores.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"{len(casos)} descripciones, {len(quedan)} tras deduplicar, {len(SECTORES)} sectores")
print(f"similitud media entre distintas: {S[S > 0].mean():.3f}")
for c in random.sample(quedan, 5):
    print(f"  [{c['sector']}] {c['descripcion']}")
