"""Run the whole local benchmark: every model, every condition, resumably.

One model is loaded at a time -- the GPU has 8 GB and two 8B models do not fit
together -- and the order is chosen so a partial run is still useful: all three
conditions on the test sample first, then the calibration set each factor needs,
and the contamination probe last because it only qualifies the others.

Because the runner skips answered tasks, killing and restarting this script
loses at most the answer in flight.
"""
import os
import subprocess
import sys
import time
from datetime import datetime

MODELOS = ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b", "mistral:7b"]
TAREAS = ([("muestra", c) for c in ("A", "B", "C")]
          + [("calibracion", c) for c in ("A", "B", "C")]
          + [("muestra", "P")])
OLLAMA = os.environ.get("OLLAMA_EXE", r"F:\PBE\Ollama\ollama.exe")
PY = sys.executable
LOG = "results/llm_bench/log.txt"


def registrar(msg):
    linea = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(linea + "\n")


def main():
    os.makedirs("results/llm_bench", exist_ok=True)
    modelos = sys.argv[1:] or MODELOS
    for modelo in modelos:
        registrar(f"=== {modelo} ===")
        for conjunto, cond in TAREAS:
            t0 = time.time()
            registrar(f"  {conjunto} {cond} ...")
            r = subprocess.run([PY, "src/68_llm_runner.py", "ollama", modelo, cond,
                                "--conjunto", conjunto],
                               capture_output=True, text=True, encoding="utf-8")
            ultima = (r.stdout.strip().splitlines() or ["(sin salida)"])[-1]
            registrar(f"  {conjunto} {cond} -> {ultima}  "
                      f"({(time.time() - t0) / 60:.1f} min, exit {r.returncode})")
            if r.returncode:
                registrar("    " + r.stderr.strip()[-400:])
        # descarga el modelo de la GPU antes de cargar el siguiente
        subprocess.run([OLLAMA, "stop", modelo], capture_output=True)
    registrar("banco completo")


if __name__ == "__main__":
    main()
