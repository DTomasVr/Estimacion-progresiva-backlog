"""Corre el banco de estimacion contra LLM por API (Groq, Gemini), reanudable.

    python src/259_banco_api.py groq openai/gpt-oss-120b

Mismo banco que 70_correr_banco.py (601 tareas de muestra + 130 de calibracion,
mismos prompts), en el orden que deja una corrida parcial util: primero la
condicion A (solo el texto, la comparable con el producto) y su calibracion,
despues B (con ejemplos) y C (con historial del proyecto). Si el proveedor agota
la cuota del dia, el runner corta sin registrar nada y esto se relanza otro dia:
solo hace lo que falta.
"""
import subprocess
import sys
import time
from datetime import datetime

ORDEN = [("muestra", "A"), ("calibracion", "A"), ("muestra", "B"), ("calibracion", "B"),
         ("muestra", "C"), ("calibracion", "C")]
LOG = "results/llm_bench/log_api.txt"


def registrar(msg):
    linea = f"[{datetime.now():%d/%m %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(linea + "\n")


def main():
    backend, modelo = sys.argv[1], sys.argv[2]
    for conjunto, cond in ORDEN:
        t0 = time.time()
        r = subprocess.run([sys.executable, "src/68_llm_runner.py", backend, modelo, cond,
                            "--conjunto", conjunto],
                           capture_output=True, text=True, encoding="utf-8")
        ultima = (r.stdout.strip().splitlines() or ["(sin salida)"])[-1]
        registrar(f"{modelo} {conjunto} {cond}: {ultima} "
                  f"({(time.time() - t0) / 60:.1f} min, exit {r.returncode})")
        if "cortado" in ultima:
            registrar(f"{modelo}: sin cuota, se retoma otro dia")
            return
    registrar(f"{modelo}: banco completo")


if __name__ == "__main__":
    main()
