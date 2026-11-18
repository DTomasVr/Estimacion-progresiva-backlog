"""Runs the estimation bench against LLMs through an API (Groq, Gemini), resumable.

    python src/259_banco_api.py groq openai/gpt-oss-120b

Same bench as 70_correr_banco.py (601 sample tasks + 130 calibration tasks, same
prompts), in the order that leaves a useful partial run: first condition A (text
only, the one comparable with the product) and its calibration, then B (with
examples) and C (with the project history). If the provider runs out of daily quota,
the runner stops without recording anything and this is relaunched another day: it
only does what is missing.
"""
import subprocess
import sys
import time
from datetime import datetime

ORDER = [("muestra", "A"), ("calibracion", "A"), ("muestra", "B"), ("calibracion", "B"),
         ("muestra", "C"), ("calibracion", "C")]
LOG = "results/llm_bench/log_api.txt"


def log(msg):
    line = f"[{datetime.now():%d/%m %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    backend, model = sys.argv[1], sys.argv[2]
    for subset, cond in ORDER:
        t0 = time.time()
        r = subprocess.run([sys.executable, "src/68_llm_runner.py", backend, model, cond,
                            "--conjunto", subset],
                           capture_output=True, text=True, encoding="utf-8")
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        log(f"{model} {subset} {cond}: {last} "
            f"({(time.time() - t0) / 60:.1f} min, exit {r.returncode})")
        if r.returncode == 3 or "stopped" in r.stdout:
            log(f"{model}: no quota left, it resumes another day")
            return
    log(f"{model}: bench complete")


if __name__ == "__main__":
    main()
