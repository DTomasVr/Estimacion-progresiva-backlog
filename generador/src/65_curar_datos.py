"""Cura los datos del profesor SIN generar nada nuevo. Dos filtros:

  A. ANCLAJE (instantaneo, sin modelo). Una tarea se descarta si sus sustantivos
     de contenido no tienen soporte en su historia, en la descripcion del
     proyecto, ni en el vocabulario tecnico generico. Ataca las invenciones
     ("video receipt", "audiobook variants").

  B. BENEFICIO (juez local). ¿La clausula "so that" se deduce de la capacidad
     para ese rol? Es el defecto que peor se lee y que NUNCA se habia medido:
     "quiero registrar la hora de reserva, para que pueda monitorear la salud
     financiera de los pacientes" en una app de limpieza. Se usa el propio
     profesor (qwen2.5:7b, local) como juez, por lotes de 25.

Se mide lo mismo sobre la salida del MODELO (results/58_salidas_con.json), para
saber de donde partimos y no solo como esta el profesor.

  python src/65_curar_datos.py --probe      100 historias, imprime el juicio
  python src/65_curar_datos.py              todo; escribe el fichero curado

out: data/synth/backlogs_v6_qwen2.5_7b.jsonl  y  results/65_curacion.csv
"""
import json
import os
import re
import sys
import time

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://127.0.0.1:11434/api/generate"
MODELO = "qwen2.5:7b"
LOTE = 25
RE_H = re.compile(r"^\s*as\s+(?:an?|the)\s+([^,]{2,40}?)\s*,\s*i\s+want\s+(.*?)\s*,\s*so\s+that\s+(.*)$",
                  re.I | re.S)

PARADA = {
    "a", "an", "the", "to", "for", "of", "in", "on", "and", "or", "with", "by",
    "from", "at", "as", "that", "this", "it", "is", "are", "be", "can", "so",
    "my", "their", "its", "i", "want", "able", "all", "each", "when", "if",
    "add", "create", "store", "display", "show", "send", "implement", "allow",
    "update", "delete", "generate", "provide", "ensure", "validate", "notify",
    "track", "manage", "set", "view", "make", "apply", "check", "log", "save",
    "new", "when", "into", "them", "they", "have", "has", "will", "should",
}
# vocabulario tecnico que es legitimo aunque no este en la descripcion
TECNICO = {
    "database", "screen", "page", "form", "field", "button", "dashboard", "view",
    "endpoint", "api", "email", "sms", "push", "notification", "report", "list",
    "table", "filter", "search", "status", "record", "profile", "account",
    "threshold", "audit", "trail", "history", "log", "logs", "queue", "cache",
    "token", "session", "role", "permission", "admin", "panel", "app", "system",
    "backend", "frontend", "server", "client", "message", "alert", "reminder",
    "confirmation", "receipt", "invoice", "summary", "detail", "details", "data",
    "user", "users", "date", "time", "timestamp", "amount", "total", "price",
    "quantity", "code", "id", "name", "address", "phone", "photo", "image",
    "file", "document", "pdf", "export", "import", "csv", "rule", "rules",
}


def palabras(t):
    return [w for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if w not in PARADA]


def soportada(t, contexto):
    """proporcion de sustantivos de contenido con soporte en el contexto"""
    p = palabras(t)
    if not p:
        return 1.0
    ok = 0
    for w in p:
        if w in TECNICO:
            ok += 1
        elif any(w[:5] == c[:5] for c in contexto):
            ok += 1
    return ok / len(p)


PROMPT = """You check whether a user story is internally coherent.

For each numbered item, answer whether the BENEFIT is a true consequence of the CAPABILITY for that ROLE. Answer NO if the benefit is about something unrelated, belongs to a different role, or does not follow.

{items}

Answer with exactly one line per item, nothing else:
1: YES
2: NO
"""


def juzgar(triples, timeout=300):
    """triples: [(rol, capacidad, beneficio)] -> [bool]"""
    items = "\n".join(
        f"{i+1}. ROLE: {r} | CAPABILITY: {c} | BENEFIT: {b}"
        for i, (r, c, b) in enumerate(triples))
    try:
        resp = requests.post(URL, json={
            "model": MODELO, "prompt": PROMPT.format(items=items), "stream": False,
            "options": {"temperature": 0, "num_predict": 40 * len(triples)}},
            timeout=timeout).json()["response"]
    except Exception as e:
        print(f"  fallo el juez: {type(e).__name__}", flush=True)
        return [True] * len(triples)          # ante la duda no se descarta
    veredicto = [True] * len(triples)
    for linea in resp.split("\n"):
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(yes|no|si|sí)\b", linea.strip(), re.I)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < len(triples):
                veredicto[i] = m.group(2).lower() in ("yes", "si", "sí")
    return veredicto


def partir(h):
    m = RE_H.match(re.sub(r"\s+", " ", h).strip())
    return (m.group(1).strip(), m.group(2).strip(), m.group(3).strip(" .")) if m else None


def tasa_beneficio(historias, etq, limite=None):
    """% de historias cuyo beneficio se deduce de la capacidad"""
    trip, idx = [], []
    for i, h in enumerate(historias):
        p = partir(h)
        if p:
            trip.append(p)
            idx.append(i)
    if limite:
        trip, idx = trip[:limite], idx[:limite]
    ok = {}
    t0 = time.time()
    for k in range(0, len(trip), LOTE):
        lote = trip[k:k + LOTE]
        for j, v in enumerate(juzgar(lote)):
            ok[idx[k + j]] = v
        if k and k % (LOTE * 20) == 0:
            hecho = k + LOTE
            print(f"  {etq}: {hecho}/{len(trip)}  "
                  f"faltan ~{(len(trip)-hecho)*(time.time()-t0)/hecho/60:.0f} min", flush=True)
    n = len(ok)
    buenas = sum(ok.values())
    print(f"{etq}: {100*buenas/max(1,n):.1f} % de beneficios coherentes  "
          f"(n={n}, {n - buenas} descartables)", flush=True)
    return ok, buenas / max(1, n), n


if __name__ == "__main__":
    probe = "--probe" in sys.argv
    lim = 100 if probe else None
    filas = []

    # --- el MODELO: de donde partimos (salidas ya guardadas, no se genera nada)
    p = f"{BASE}/results/58_salidas_con.json"
    if os.path.exists(p):
        S = json.load(open(p, encoding="utf-8"))
        hs = [h["h"] for s in S for h in s["historias"]]
        _, tasa, n = tasa_beneficio(hs, "MODELO (salida actual)", lim)
        filas.append(dict(fuente="modelo", beneficio_ok=round(tasa, 4), n=n))

    # --- el PROFESOR v3
    L = [json.loads(l) for l in
         open(f"{BASE}/data/synth/backlogs_v3_qwen2.5_7b.jsonl", encoding="utf-8")]
    if probe:
        L = L[:12]
    plano = [(bi, hi, h["historia"]) for bi, b in enumerate(L)
             for hi, h in enumerate(b["historias"])]
    ok, tasa, n = tasa_beneficio([x[2] for x in plano], "PROFESOR v3", lim)
    filas.append(dict(fuente="profesor_v3", beneficio_ok=round(tasa, 4), n=n))

    # --- filtro A: anclaje de las tareas (instantaneo)
    n_t = n_fuera = 0
    for b in L:
        ctx = palabras(b["descripcion"])
        for h in b["historias"]:
            ctx_h = ctx + palabras(h["historia"])
            for t in h["tareas"]:
                n_t += 1
                if soportada(t, ctx_h) < 0.30:
                    n_fuera += 1
    print(f"ANCLAJE: {100*n_fuera/max(1,n_t):.1f} % de las tareas del profesor no estan "
          f"ancladas (n={n_t})")
    filas.append(dict(fuente="anclaje_tareas", beneficio_ok=round(1 - n_fuera / max(1, n_t), 4),
                      n=n_t))

    if probe:
        print("\n--- ejemplos juzgados NO coherentes")
        malas = [plano[i][2] for i in ok if not ok[i]][:8]
        for m in malas:
            print(f"  * {m[:150]}")
        print("\n--- ejemplos de tareas SIN anclaje")
        vistos = 0
        for b in L:
            ctx = palabras(b["descripcion"])
            for h in b["historias"]:
                for t in h["tareas"]:
                    if soportada(t, ctx + palabras(h["historia"])) < 0.30 and vistos < 8:
                        print(f"  * {t}   <- {h['historia'][:70]}")
                        vistos += 1
        sys.exit()

    # --- guardar las etiquetas del juez: el activo real de esta pasada
    # Sirven para destilar el juez en un reordenador pequeno. El fichero curado
    # importa menos: los datos del profesor ya son 91 % buenos.
    with open(f"{BASE}/data/synth/etiquetas_beneficio.jsonl", "w", encoding="utf-8") as fh:
        for i, v in ok.items():
            tr = partir(plano[i][2])
            if not tr:
                continue
            r, c, b = tr
            fh.write(json.dumps(dict(rol=r, capacidad=c, beneficio=b,
                                     coherente=bool(v), historia=plano[i][2]),
                                ensure_ascii=False) + chr(10))
    print(f"etiquetas -> data/synth/etiquetas_beneficio.jsonl ({len(ok)})")

    # --- escribir el fichero curado
    salida = f"{BASE}/data/synth/backlogs_v6_{MODELO.replace(':', '_')}.jsonl"
    n_h = n_h_out = n_t2 = n_t_out = 0
    with open(salida, "w", encoding="utf-8") as fh:
        for bi, b in enumerate(L):
            ctx = palabras(b["descripcion"])
            hs = []
            for hi, h in enumerate(b["historias"]):
                n_h += 1
                i_plano = next((k for k, x in enumerate(plano)
                                if x[0] == bi and x[1] == hi), None)
                if i_plano is not None and not ok.get(i_plano, True):
                    n_h_out += 1
                    continue
                ctx_h = ctx + palabras(h["historia"])
                ts = []
                for t in h["tareas"]:
                    n_t2 += 1
                    if soportada(t, ctx_h) < 0.30:
                        n_t_out += 1
                        continue
                    ts.append(t)
                if len(ts) >= 2:
                    hs.append(dict(historia=h["historia"], tareas=ts))
            if len(hs) >= 5:
                fh.write(json.dumps({**b, "historias": hs}, ensure_ascii=False) + "\n")
    print(f"\ncurado -> {os.path.basename(salida)}")
    print(f"  historias {n_h} -> {n_h - n_h_out} (fuera {n_h_out}, {100*n_h_out/max(1,n_h):.1f} %)")
    print(f"  tareas    {n_t2} -> {n_t2 - n_t_out} (fuera {n_t_out}, {100*n_t_out/max(1,n_t2):.1f} %)")
    import pandas as pd
    pd.DataFrame(filas).to_csv(f"{BASE}/results/65_curacion.csv", index=False)
    print("-> results/65_curacion.csv")
