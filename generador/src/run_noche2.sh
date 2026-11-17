#!/usr/bin/env bash
# Night chain 2: LLM bench (the thesis' speed claim) -> its summary -> HP search
# with whatever time is left. Each stage resumes if it was already partly done.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/noche2.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"

echo "[$(date '+%F %T')] banco: nuestro pipeline vs 3 LLMs (28 proyectos)" >> "$LOG"
"$PY" src/20_banco_llms.py 24 > logs/20_banco.log 2>&1
echo "[$(date '+%F %T')] banco terminado ($(wc -l < results/20_banco.jsonl) corridas)" >> "$LOG"

"$PY" src/22_resumen_banco.py > logs/22_resumen.log 2>&1
echo "[$(date '+%F %T')] resumen del banco escrito" >> "$LOG"

echo "[$(date '+%F %T')] busqueda de hiperparametros (hasta 5 h)" >> "$LOG"
"$PY" src/21_busqueda_hp.py 5 40 > logs/21_busqueda.log 2>&1
echo "[$(date '+%F %T')] busqueda terminada" >> "$LOG"
