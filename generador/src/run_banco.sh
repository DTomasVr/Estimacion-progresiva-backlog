#!/usr/bin/env bash
# Recalcula NUESTRO brazo del banco de 64 proyectos con la configuracion v3/v5,
# porque el 0,472 de fidelidad de la presentacion es de la version anterior.
# Los brazos de los LLM no se regeneran: sus salidas estaban guardadas.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/banco_v5.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 AGIX_DEVICE=cuda AGIX_SEED=20260926
cd "$BASE"
echo "[$(date '+%F %T')] recalculando el brazo agix (64 proyectos)" > "$LOG"
"$PY" -u src/20_banco_llms.py 64 >> "$LOG" 2>&1 || { echo "FALLO en 20" >> "$LOG"; exit 1; }
echo "[$(date '+%F %T')] resumen pareado" >> "$LOG"
"$PY" -u src/22_resumen_banco.py >> "$LOG" 2>&1 || echo "FALLO en 22" >> "$LOG"
echo "[$(date '+%F %T')] BANCO TERMINADO" >> "$LOG"
