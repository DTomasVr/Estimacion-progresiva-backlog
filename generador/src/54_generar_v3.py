"""Generacion v3 del profesor. Cambia el prompt y, sobre todo, VALIDA lo que
devuelve en lugar de escribirlo tal cual.

Los tres defectos que corrige, todos medidos con src/52_coherencia.py sobre la
tanda v1:

  ROLES        63,5 % de los backlogs v1 usaban un solo rol. Ahora el prompt
               lleva el plantel de actores del sector y se EXIGE >= 3 roles
               distintos, todos del plantel; si no, se reintenta.

  ECO          la ultima tarea de cada historia degeneraba en un eco
               ("Create a store for the store", "Add catalog view to the
               Catalog catalog"). Se detectan y se borran.

  RELLENO      las tareas se habian vuelto una plantilla de tres lineas
               ("Implement a feature to X" / "Create a dashboard to display X" /
               "Allow users to filter by date range"). Estan prohibidas en el
               prompt y se cuentan; si pasan del 20 % se reintenta.

  python src/54_generar_v3.py --probe [N]     genera N=3 y los imprime para leer
  python src/54_generar_v3.py qwen2.5:7b      genera todo, reanudable

out: data/synth/backlogs_v3_<modelo>.jsonl
"""
import collections
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:11434/api/generate"
CASOS = json.load(open(f"{BASE}/data/synth/descripciones_v3.json", encoding="utf-8"))

PROMPT = """You are a senior product owner writing the initial backlog for a new software product.

Product: {desc}

The people who use this product are, and these are the ONLY roles you may write:
{roles}

Write 12 user stories. Rules for the stories:
- use exactly the form "As a <role>, I want <capability>, so that <benefit>"
- <role> must be copied from the list above, spelled exactly as listed. Never write "As a user".
- use AT LEAST FOUR different roles from the list across the 12 stories. The staff and operator roles get real stories too, not only the customer-facing one.
- the <benefit> must be a true consequence of that capability FOR THAT ROLE
- every story covers a different capability, and all of them are specific to THIS product

Under each story write 3 or 4 implementation tasks. Rules for the tasks:
- each task must name the concrete thing its own story is about, so a developer reading only the task knows which story it belongs to
- the tasks of one story must cover DIFFERENT layers of the work: what data has to be stored, what the screen or endpoint does, what rule or validation applies, and what happens when it fails or when someone has to be notified
- one short imperative line each, the way it would be written on a card
- FORBIDDEN: "Implement a feature to ...", "Create a dashboard to display ...", "Allow users to filter ... by date range", "Develop a ... tracking feature", and any line that repeats the same word twice ("create a store for the store")

Example of the exact output format, for a different product:
S| As a pharmacist, I want to reject a prescription that is missing the doctor's signature, so that I do not dispense medicine illegally.
T| Add a rejection reason and timestamp to the prescription record
T| Show the missing-signature warning on the prescription review screen
T| Block dispensing while the prescription is in rejected state
T| Notify the patient by push that the prescription needs a new signature
S| As a courier, I want to see which orders are cold-chain, so that I load them in the insulated bag first.
T| Flag cold-chain items in the order payload sent to the courier app
T| Sort the pickup list so cold-chain orders appear first
T| Warn the courier if a cold-chain order has been in the van over 30 minutes

Answer ONLY in that format, nothing else, no numbering, no commentary, no blank lines.

The example above is only two stories long so you can see the format. Your answer must contain TWELVE stories, each followed by its 3 or 4 tasks, spread over at least four of the roles listed. Do not stop early.
"""

FORMA = re.compile(r"^as\s+(?:an?|the)\s+[^,]{2,60},?\s+i\s+want\b.*", re.I)
RE_ROL = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,", re.I)
RELLENO = [
    re.compile(r"^implement a feature to", re.I),
    re.compile(r"^create a dashboard to (display|show|view)", re.I),
    re.compile(r"^develop a .{0,30}tracking feature", re.I),
    re.compile(r"filter .{0,30}by date range", re.I),
    re.compile(r"^(add|create|implement) (a )?(feature|functionality)\b", re.I),
]
PARADA = {"a", "an", "the", "to", "for", "of", "in", "on", "and", "with", "by",
          "from", "at", "as", "their", "its", "my", "all", "new", "each", "per"}


def normalizar(r):
    r = r.strip().lower().strip(".")
    r = re.sub(r"^(a|an|the)\s+", "", r)
    return re.sub(r"\s+", " ", r)


def eco(tarea, historia):
    """tarea degenerada: repite un sustantivo, o es la historia otra vez"""
    p = [w for w in re.findall(r"[a-z]+", tarea.lower()) if w not in PARADA]
    if len(p) < 2:
        return True
    # "create a store for the store", "add inventory process to the inventory process"
    if len(p) != len(set(p)):
        c = collections.Counter(p).most_common(1)[0][1]
        if c >= 2 and len(p) <= 8:
            return True
    # la tarea es la historia recortada
    h = set(re.findall(r"[a-z]+", historia.lower())) - PARADA
    if p and len(set(p) - h) / len(set(p)) < 0.2:
        return True
    return False


def pedir(modelo, caso, temp=0.85, timeout=420):
    roles = "\n".join(f"  - {r}" for r in caso["roles"])
    r = requests.post(URL, json={
        "model": modelo,
        "prompt": PROMPT.format(desc=caso["descripcion"], roles=roles),
        "stream": False,
        "options": {"temperature": temp, "num_predict": 3000}}, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parsear(txt, caso):
    """parsea y limpia; devuelve (historias, diagnostico)"""
    historias, actual = [], None
    n_eco = n_relleno = n_tareas = 0
    for linea in txt.split("\n"):
        linea = linea.strip()
        if linea.startswith("S|"):
            if actual and len(actual["tareas"]) >= 2:
                historias.append(actual)
            actual = dict(historia=linea[2:].strip(" -*"), tareas=[])
        elif linea.startswith("T|") and actual is not None:
            t = linea[2:].strip(" -*")
            if len(t) < 8:
                continue
            n_tareas += 1
            if eco(t, actual["historia"]):
                n_eco += 1
                continue
            if any(p.search(t) for p in RELLENO):
                n_relleno += 1
                continue
            if t.lower() in {x.lower() for x in actual["tareas"]}:
                continue
            actual["tareas"].append(t)
    if actual and len(actual["tareas"]) >= 2:
        historias.append(actual)

    historias = [h for h in historias if FORMA.match(h["historia"])]
    plantel = {normalizar(r) for r in caso["roles"]}
    roles = [normalizar(m.group(1)) for h in historias
             if (m := RE_ROL.match(h["historia"]))]
    dentro = sum(1 for r in roles if r in plantel)
    diag = dict(n=len(historias), roles=len(set(roles)),
                en_plantel=dentro / len(roles) if roles else 0,
                eco=n_eco / n_tareas if n_tareas else 0,
                relleno=n_relleno / n_tareas if n_tareas else 0,
                tareas=round(sum(len(h["tareas"]) for h in historias) / max(1, len(historias)), 1))
    return historias, diag


MIN_HISTORIAS = 8
MIN_ROLES = 3
MIN_PLANTEL = 0.75
MAX_RELLENO = 0.25


def acepta(d):
    return (d["n"] >= MIN_HISTORIAS and d["roles"] >= MIN_ROLES
            and d["en_plantel"] >= MIN_PLANTEL and d["relleno"] <= MAX_RELLENO)


def porque(d):
    r = []
    if d["n"] < MIN_HISTORIAS:
        r.append(f"solo {d['n']} historias")
    if d["roles"] < MIN_ROLES:
        r.append(f"solo {d['roles']} roles")
    if d["en_plantel"] < MIN_PLANTEL:
        r.append(f"roles fuera del plantel {100*(1-d['en_plantel']):.0f} %")
    if d["relleno"] > MAX_RELLENO:
        r.append(f"relleno {100*d['relleno']:.0f} %")
    return ", ".join(r)


if __name__ == "__main__":
    MODELO = "qwen2.5:7b"
    if "--probe" in sys.argv:
        n = int(sys.argv[sys.argv.index("--probe") + 1]) if len(sys.argv) > sys.argv.index("--probe") + 1 else 3
        paso = max(1, len(CASOS) // n)
        for caso in CASOS[::paso][:n]:
            t = time.time()
            H, d = parsear(pedir(MODELO, caso), caso)
            print(f"\n{'='*100}\n### [{caso['sector']}] {time.time()-t:.1f}s  {d}"
                  f"{'' if acepta(d) else '  <-- RECHAZADO: ' + porque(d)}")
            print(f"  {caso['descripcion']}")
            print(f"  plantel: {', '.join(caso['roles'])}")
            for h in H:
                print(f"\n   * {h['historia']}")
                for x in h["tareas"]:
                    print(f"       - {x}")
        sys.exit()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        MODELO = sys.argv[1]
    salida = f"{BASE}/data/synth/backlogs_v3_{MODELO.replace(':', '_')}.jsonl"
    hechos = set()
    if os.path.exists(salida):
        hechos = {json.loads(l)["id"] for l in open(salida, encoding="utf-8")}
    pend = [c for c in CASOS if c["id"] not in hechos]
    print(f"{MODELO}: {len(hechos)} ya hechos, faltan {len(pend)}", flush=True)

    t0 = time.time()
    n_ok = n_desc = 0
    acum = collections.Counter()
    with open(salida, "a", encoding="utf-8") as fh:
        for caso in pend:
            mejor, mejor_d = None, None
            for intento in range(3):
                try:
                    t = time.time()
                    H, d = parsear(pedir(MODELO, caso), caso)
                except Exception as e:
                    print(f"  reintento {intento+1} en {caso['id']}: {type(e).__name__}", flush=True)
                    time.sleep(5)
                    continue
                if mejor is None or d["n"] * d["roles"] > mejor_d["n"] * mejor_d["roles"]:
                    mejor, mejor_d = H, d
                if acepta(d):
                    break
            if mejor is None or not acepta(mejor_d):
                n_desc += 1
                print(f"  descartado {caso['id']}: {porque(mejor_d) if mejor_d else 'sin respuesta'}",
                      flush=True)
                continue
            fh.write(json.dumps(dict(id=caso["id"], sector=caso["sector"],
                                     descripcion=caso["descripcion"],
                                     roles=caso["roles"], historias=mejor,
                                     diag=mejor_d, segundos=round(time.time() - t, 1),
                                     modelo=MODELO), ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += 1
            acum["roles"] += mejor_d["roles"]
            acum["hist"] += mejor_d["n"]
            acum["tareas"] += sum(len(h["tareas"]) for h in mejor)
            if n_ok % 10 == 0:
                falta = (len(pend) - n_ok - n_desc) * (time.time() - t0) / (n_ok + n_desc)
                print(f"  {n_ok}/{len(pend)}  descartados {n_desc}  "
                      f"roles/backlog {acum['roles']/n_ok:.1f}  "
                      f"historias {acum['hist']/n_ok:.1f}  "
                      f"tareas/historia {acum['tareas']/acum['hist']:.1f}  "
                      f"faltan ~{falta/60:.0f} min", flush=True)
    print(f"listo: {n_ok} nuevos, {n_desc} descartados, "
          f"{(time.time()-t0)/60:.1f} min -> {salida}")
