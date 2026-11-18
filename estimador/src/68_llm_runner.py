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

The JSONL field names (modelo, condicion, segundos, valido, horas, minimo, maximo,
entrada, salida, crudo) are kept as they were written: the published runs use them.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")
# The keys are read from .env.llm (ignored by git), one per line, KEY=value; whatever is
# already in the environment wins. They are never printed.
if Path(".env.llm").exists():
    for _l in open(".env.llm", encoding="utf-8-sig"):
        _l = _l.strip()
        if "=" in _l and not _l.startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# 127.0.0.1 and not localhost: on Windows localhost resolves first to IPv6 (::1),
# Ollama only listens on IPv4, and each new connection lost ~2,1 s in the
# fallback -- measured: 2,09 s against 0,02 s. The session reuses the connection.
SESION = requests.Session()

# The prompts are in Spanish on purpose: they are the bench's input, the same for every
# model, and translating them would change the experiment.

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
        hist = json.loads(fila["hist"])
        partes.append("Tareas ya terminadas en este mismo proyecto, con sus horas reales:")
        partes += [f"- {e['texto']} → {e['horas']} h" for e in hist]
        partes.append("")
    if condicion == "P":                          # contamination probe
        return ("Esta es una incidencia de un Jira público. Si la reconoces, "
                'responde {"clave": "PROYECTO-123", "horas_registradas": número}; '
                'si no, {"clave": null, "horas_registradas": null}.\n\n'
                + fila["texto_llm"])
    partes.append(f"Tarea a estimar (tipo: {fila['task_kind']}):")
    partes.append(fila["texto_llm"])
    return "\n".join(partes)


# ----------------------------------------------------------------- backends
def llamar_ollama(modelo, sistema, usuario):
    r = SESION.post("http://127.0.0.1:11434/api/chat", timeout=600, json={
        "model": modelo, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "seed": 0},
        "messages": [{"role": "system", "content": sistema},
                     {"role": "user", "content": usuario}]})
    r.raise_for_status()
    j = r.json()
    return j["message"]["content"], {"entrada": j.get("prompt_eval_count"),
                                     "salida": j.get("eval_count")}


def llamar_groq(modelo, sistema, usuario):
    return _compatible_openai("https://api.groq.com/openai/v1/chat/completions",
                              "GROQ_API_KEY", modelo, sistema, usuario,
                              {"seed": 0, "response_format": {"type": "json_object"}})


def llamar_gemini(modelo, sistema, usuario):
    # Google's OpenAI-compatible endpoint. No reasoning: with reasoning on, the recent
    # Flash models took over 40 s per task or did not answer (measured on 29/10/2026);
    # with reasoning_effort=none, ~9 s.
    return _compatible_openai(
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY", modelo, sistema, usuario, {"reasoning_effort": "none"})


def _compatible_openai(url, var, modelo, sistema, usuario, extra):
    clave = os.environ.get(var)
    if not clave:
        raise SystemExit(f"{var} is missing from the environment and from .env.llm")
    for intento in range(8):
        r = SESION.post(
            url, timeout=180,
            headers={"Authorization": f"Bearer {clave}"},
            json={"model": modelo, "temperature": 0, **extra,
                  "messages": [{"role": "system", "content": sistema},
                               {"role": "user", "content": usuario}]})
        if r.status_code in (429, 503):
            espera = float(r.headers.get("retry-after") or 2 ** intento * 5)
            if espera > 600:                     # the DAY's quota is used up: stop, do not wait
                raise CuotaAgotada(f"the provider asks to wait {espera / 3600:.1f} h")
            print(f"    rate limit, waiting {espera:.0f}s", flush=True)
            time.sleep(espera)
            continue
        r.raise_for_status()
        # slow down BEFORE the quota runs out, as the provider says
        quedan = r.headers.get("x-ratelimit-remaining-requests")
        if quedan is not None and int(quedan) <= 1:
            reset = r.headers.get("x-ratelimit-reset-requests", "60s")
            time.sleep(_segundos(reset) + 1)
        j = r.json()
        u = j.get("usage", {})
        return (j["choices"][0]["message"]["content"],
                {"entrada": u.get("prompt_tokens"), "salida": u.get("completion_tokens")})
    raise CuotaAgotada("rate limit persists after 8 attempts")


class CuotaAgotada(Exception):
    """No quota: the run stops WITHOUT recording the task, and resumes later."""


def _segundos(s):
    total = 0.0
    for n, u in re.findall(r"([\d.]+)(ms|s|m|h)", s):
        total += float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[u]
    return total or 60


BACKENDS = {"ollama": llamar_ollama, "groq": llamar_groq, "gemini": llamar_gemini}


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
    # shuffled and fixed order: a run stopped halfway is still a representative
    # sample, and every model goes through the tasks in the same order
    pendientes = d[~d.task_id.isin(hechas)].sample(frac=1, random_state=20260914)
    if a.limite:
        pendientes = pendientes.head(a.limite)
    print(f"{destino.name}: {len(hechas)} done, {len(pendientes)} pending")

    llamar = BACKENDS[a.backend]
    t0 = time.time()
    with open(destino, "a", encoding="utf-8") as fh:
        for i, fila in enumerate(pendientes.to_dict("records"), 1):
            usuario = prompt(fila, a.condicion, ejemplos)
            t = time.time()
            try:
                crudo, uso = llamar(a.modelo, SISTEMA, usuario)
                res = interpretar(crudo, a.condicion)
            except CuotaAgotada as e:
                print(f"stopped at {i - 1}/{len(pendientes)}: {e}; it resumes when relaunched", flush=True)
                sys.exit(3)  # 259_banco_api.py tells the quota stop apart
            except Exception as e:                    # recorded, not hidden
                crudo, uso, res = f"ERROR: {e}", {}, {"valido": False}
            fh.write(json.dumps({"task_id": fila["task_id"], "backend": a.backend,
                                 "modelo": a.modelo, "condicion": a.condicion,
                                 "segundos": round(time.time() - t, 2),
                                 **uso, **res, "crudo": crudo[:500]},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(pendientes)}  {(time.time()-t0)/i:.1f}s/task",
                      flush=True)
    print("done")


if __name__ == "__main__":
    main()
