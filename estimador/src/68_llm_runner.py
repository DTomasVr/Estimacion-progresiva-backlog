"""Run one LLM over the frozen benchmark sample, resumably.

    python src/68_llm_runner.py ollama llama3.1:8b A
    python src/68_llm_runner.py groq  llama-3.1-8b-instant C --conjunto calibracion

Every answer is appended to a JSONL file as soon as it arrives, and a rerun
skips the tasks already answered. Free tiers are rate limited and a laptop can
sleep, so a run that dies at task 412 must continue at 413, not start over.

Rate limits are read from the provider's response headers instead of being
hard-coded: they change, and a guessed number either wastes hours or trips the
limit.

Unparseable or refused answers are recorded as such, never dropped, so the
evaluation can report each model's failure rate next to its accuracy.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")

SISTEMA = (
    "Eres un desarrollador de software senior que estima esfuerzo. "
    "Estima cuántas HORAS DE TRABAJO EFECTIVO registrará UNA sola persona para "
    "completar la tarea: tiempo de reloj dedicado, no días de calendario ni "
    "puntos de historia. Responde únicamente con un objeto JSON de la forma "
    '{"horas": número, "minimo": número, "maximo": número}, donde minimo y '
    "maximo forman un intervalo en el que estás 80 % seguro de que cae el valor real."
)


def prompt(fila, condicion, ejemplos):
    partes = []
    if condicion == "B":
        partes.append("Ejemplos de tareas ya terminadas, con sus horas reales:")
        partes += [f"- [{e['tipo']}] {e['texto']} → {e['horas']} h" for e in ejemplos]
        partes.append("")
    if condicion == "C":
        hist = json.loads(fila.hist)
        partes.append("Tareas ya terminadas en este mismo proyecto, con sus horas reales:")
        partes += [f"- {e['texto']} → {e['horas']} h" for e in hist]
        partes.append("")
    if condicion == "P":                          # sonda de contaminacion
        return ("Esta es una incidencia de un Jira público. Si la reconoces, "
                'responde {"clave": "PROYECTO-123", "horas_registradas": número}; '
                'si no, {"clave": null, "horas_registradas": null}.\n\n'
                + fila.texto_llm)
    partes.append(f"Tarea a estimar (tipo: {fila.task_kind}):")
    partes.append(fila.texto_llm)
    return "\n".join(partes)


# ----------------------------------------------------------------- backends
def llamar_ollama(modelo, sistema, usuario):
    r = requests.post("http://localhost:11434/api/chat", timeout=600, json={
        "model": modelo, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "seed": 0},
        "messages": [{"role": "system", "content": sistema},
                     {"role": "user", "content": usuario}]})
    r.raise_for_status()
    j = r.json()
    return j["message"]["content"], {"entrada": j.get("prompt_eval_count"),
                                     "salida": j.get("eval_count")}


def llamar_groq(modelo, sistema, usuario):
    clave = os.environ.get("GROQ_API_KEY")
    if not clave:
        raise SystemExit("falta GROQ_API_KEY en el entorno")
    for intento in range(8):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", timeout=120,
            headers={"Authorization": f"Bearer {clave}"},
            json={"model": modelo, "temperature": 0, "seed": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": sistema},
                               {"role": "user", "content": usuario}]})
        if r.status_code == 429:
            espera = float(r.headers.get("retry-after") or 2 ** intento * 5)
            print(f"    limite de tasa, esperando {espera:.0f}s", flush=True)
            time.sleep(espera)
            continue
        r.raise_for_status()
        # frena ANTES de agotar la cuota, segun lo que diga el proveedor
        quedan = r.headers.get("x-ratelimit-remaining-requests")
        if quedan is not None and int(quedan) <= 1:
            reset = r.headers.get("x-ratelimit-reset-requests", "60s")
            time.sleep(_segundos(reset) + 1)
        j = r.json()
        u = j.get("usage", {})
        return (j["choices"][0]["message"]["content"],
                {"entrada": u.get("prompt_tokens"), "salida": u.get("completion_tokens")})
    raise RuntimeError("limite de tasa persistente")


def _segundos(s):
    total = 0.0
    for n, u in re.findall(r"([\d.]+)(ms|s|m|h)", s):
        total += float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[u]
    return total or 60


BACKENDS = {"ollama": llamar_ollama, "groq": llamar_groq}


def interpretar(texto, condicion):
    try:
        j = json.loads(re.search(r"\{.*\}", texto, re.S).group(0))
    except Exception:
        return {"valido": False}
    if condicion == "P":
        return {"valido": True, "clave": j.get("clave"),
                "horas_sonda": j.get("horas_registradas")}
    try:
        h = float(j["horas"])
        lo = float(j.get("minimo", "nan"))
        hi = float(j.get("maximo", "nan"))
    except (KeyError, TypeError, ValueError):
        return {"valido": False}
    return {"valido": h > 0, "horas": h, "minimo": lo, "maximo": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=BACKENDS)
    ap.add_argument("modelo")
    ap.add_argument("condicion", choices=["A", "B", "C", "P"])
    ap.add_argument("--conjunto", default="muestra", choices=["muestra", "calibracion"])
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet(BENCH / f"{a.conjunto}.parquet")
    if a.condicion == "C":
        d = d[d.n_hist >= 3]
    if a.condicion == "P":
        d = d[d.source != "SiP"]
    ejemplos = json.load(open(BENCH / "pocos_ejemplos.json", encoding="utf-8"))

    RAW.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]", "_", a.modelo)
    destino = RAW / f"{a.conjunto}__{a.backend}__{slug}__{a.condicion}.jsonl"
    hechas = set()
    if destino.exists():
        hechas = {json.loads(l)["task_id"] for l in open(destino, encoding="utf-8")}
    pendientes = d[~d.task_id.isin(hechas)]
    if a.limite:
        pendientes = pendientes.head(a.limite)
    print(f"{destino.name}: {len(hechas)} hechas, {len(pendientes)} pendientes")

    llamar = BACKENDS[a.backend]
    t0 = time.time()
    with open(destino, "a", encoding="utf-8") as fh:
        for i, fila in enumerate(pendientes.itertuples(), 1):
            usuario = prompt(fila, a.condicion, ejemplos)
            t = time.time()
            try:
                crudo, uso = llamar(a.modelo, SISTEMA, usuario)
                res = interpretar(crudo, a.condicion)
            except Exception as e:                    # se registra, no se oculta
                crudo, uso, res = f"ERROR: {e}", {}, {"valido": False}
            fh.write(json.dumps({"task_id": fila.task_id, "backend": a.backend,
                                 "modelo": a.modelo, "condicion": a.condicion,
                                 "segundos": round(time.time() - t, 2),
                                 **uso, **res, "crudo": crudo[:500]},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(pendientes)}  {(time.time()-t0)/i:.1f}s/tarea",
                      flush=True)
    print("listo")


if __name__ == "__main__":
    main()
