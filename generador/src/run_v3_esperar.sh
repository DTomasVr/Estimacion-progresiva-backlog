#!/usr/bin/env bash
# Espera a que termine src/54_generar_v3.py y lanza la cadena.
# Se separa en otro fichero para no editar nunca run_v3.sh mientras corre:
# bash relee el script por desplazamiento de bytes y editarlo en caliente lo
# rompe a media ejecucion (paso una noche por eso).
set -u
BASE="/f/Software Task generation Model"
LOG="$BASE/logs/v3.log"
cd "$BASE"

echo "[$(date '+%F %T')] esperando a que termine la generacion v3" >> "$LOG"
while ! grep -qE "^listo:|Traceback" logs/54_v3.log; do sleep 60; done

N=$(wc -l < data/synth/backlogs_v3_qwen2.5_7b.jsonl)
echo "[$(date '+%F %T')] generacion terminada: $N backlogs" >> "$LOG"
if [ "$N" -lt 450 ]; then
  echo "[$(date '+%F %T')] ABORTO: solo $N backlogs, esperaba ~600" >> "$LOG"
  exit 1
fi
exec bash src/run_v3.sh
