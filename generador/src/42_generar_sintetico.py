"""Phase 2 of the distillation pilot: ask a local LLM for a full backlog for each
product description, to teach our model the consumer/product domains the Jira
corpus does not cover.

Licence note: only Apache-2.0 models are used (mistral:7b, qwen2.5:7b), so the
resulting fine-tuned model carries no naming or usage conditions. llama is
excluded on purpose -- its licence would force "Llama" into the model name.

  python src/42_generar_sintetico.py --probe          compare models on 2 cases
  python src/42_generar_sintetico.py MODELO [N]       generate, resumable

out: data/synth/backlogs_<modelo>.jsonl  (one line per description)
"""
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:11434/api/generate"
CASOS = json.load(open(f"{BASE}/data/synth/descripciones.json", encoding="utf-8"))

PROMPT = """You are a senior product owner writing the initial backlog for a new software product.

Product: {desc}

Write 12 user stories. Each story must:
- follow exactly the form "As a <role>, I want <capability>, so that <benefit>"
- be specific to THIS product, never generic ("As a user, I want a good system" is wrong)
- cover a different capability from the others

Under each story write 3 to 5 implementation tasks. Each task must:
- be one short imperative line a developer could pick up (e.g. "Add coupon code field to checkout form")
- be concrete and different from the other tasks

Answer ONLY with this format, nothing else, no numbering, no commentary:
S| As a ..., I want ..., so that ...
T| task
T| task
T| task
S| As a ..., I want ..., so that ...
T| task
"""

FORMA = re.compile(r"^as\s+(?:an?|the)\s+[^,]{2,60},?\s+i\s+want\b.*", re.I)


def pedir(modelo, desc, temp=0.8, timeout=300):
    r = requests.post(URL, json={"model": modelo, "prompt": PROMPT.format(desc=desc),
                                 "stream": False, "options": {"temperature": temp, "num_predict": 1400}},
                      timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parsear(txt):
    historias, actual = [], None
    for linea in txt.split("\n"):
        linea = linea.strip()
        if linea.startswith("S|"):
            if actual and actual["tareas"]:
                historias.append(actual)
            actual = dict(historia=linea[2:].strip(" -*"), tareas=[])
        elif linea.startswith("T|") and actual is not None:
            t = linea[2:].strip(" -*")
            if len(t) >= 5:
                actual["tareas"].append(t)
    if actual and actual["tareas"]:
        historias.append(actual)
    return historias


def calidad(historias):
    """cheap, readable quality signals -- the judgement is done by reading"""
    if not historias:
        return dict(n=0, formato=0, tareas=0, largo=0)
    fmt = sum(bool(FORMA.match(h["historia"])) for h in historias) / len(historias)
    tareas = [t for h in historias for t in h["tareas"]]
    return dict(n=len(historias), formato=round(fmt, 2),
                tareas=round(len(tareas) / len(historias), 1),
                largo=round(sum(len(t.split()) for t in tareas) / max(1, len(tareas)), 1))


def descargar(modelo):
    try:
        requests.post("http://127.0.0.1:11434/api/generate",
                      json={"model": modelo, "keep_alive": 0}, timeout=60)
    except Exception:
        pass


if "--probe" in sys.argv:
    for modelo in ["mistral:7b", "qwen2.5:7b"]:
        for caso in CASOS[:1] + CASOS[100:101]:
            t = time.time()
            txt = pedir(modelo, caso["descripcion"])
            H = parsear(txt)
            print(f"\n{'='*100}\n### {modelo}  [{caso['sector']}]  {time.time()-t:.1f}s  {calidad(H)}")
            print(f"  {caso['descripcion']}")
            for h in H[:4]:
                print(f"   * {h['historia'][:120]}")
                for x in h["tareas"][:3]:
                    print(f"       - {x[:95]}")
        descargar(modelo)
    sys.exit()

MODELO = sys.argv[1] if len(sys.argv) > 1 else "mistral:7b"
N = int(sys.argv[2]) if len(sys.argv) > 2 else len(CASOS)
salida = f"{BASE}/data/synth/backlogs_{MODELO.replace(':', '_')}.jsonl"
hechos = set()
if os.path.exists(salida):
    hechos = {json.loads(l)["id"] for l in open(salida, encoding="utf-8")}
print(f"{MODELO}: {len(hechos)} ya hechos, faltan {min(N, len(CASOS)) - len(hechos)}", flush=True)

t0 = time.time()
n_ok = 0
with open(salida, "a", encoding="utf-8") as fh:
    for i, caso in enumerate(CASOS[:N]):
        if caso["id"] in hechos:
            continue
        for intento in range(3):
            try:
                t = time.time()
                H = parsear(pedir(MODELO, caso["descripcion"]))
                if len(H) >= 5:
                    break
            except Exception as e:
                print(f"  reintento {intento+1} en {caso['id']}: {type(e).__name__}", flush=True)
                time.sleep(5)
                H = []
        if len(H) < 5:
            print(f"  descartado {caso['id']} ({len(H)} historias)", flush=True)
            continue
        fh.write(json.dumps(dict(id=caso["id"], sector=caso["sector"],
                                 descripcion=caso["descripcion"], historias=H,
                                 segundos=round(time.time() - t, 1), modelo=MODELO),
                            ensure_ascii=False) + "\n")
        fh.flush()
        n_ok += 1
        if n_ok % 10 == 0:
            hechas = len(hechos) + n_ok
            falta = (min(N, len(CASOS)) - hechas) * (time.time() - t0) / n_ok
            print(f"  {hechas}/{min(N, len(CASOS))}  {calidad(H)}  "
                  f"faltan ~{falta/60:.0f} min", flush=True)
print(f"listo: {n_ok} nuevos en {(time.time()-t0)/60:.1f} min -> {salida}")
