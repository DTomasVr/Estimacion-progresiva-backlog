#!/usr/bin/env bash
# Night 3: widen the bench from 28 to 80 held-out projects so the speed/fidelity
# table in the slides rests on tighter intervals. Resumes: the 28 already measured
# are skipped.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/noche3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"
echo "[$(date '+%F %T')] banco ampliado a 64 proyectos" >> "$LOG"
"$PY" -u src/20_banco_llms.py 60 > logs/20_banco_80.log 2>&1
echo "[$(date '+%F %T')] banco terminado ($(wc -l < results/20_banco.jsonl) corridas)" >> "$LOG"
"$PY" src/22_resumen_banco.py > logs/22_resumen_80.log 2>&1
echo "[$(date '+%F %T')] resumen reescrito" >> "$LOG"
