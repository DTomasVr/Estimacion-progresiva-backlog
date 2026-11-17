#!/usr/bin/env bash
# Variante C: ficha de rol + generacion por HAZ en lugar de muestreo.
#
# De donde sale. Leyendo las salidas del metodo A se vio que las 12 historias
# salian con votos max 1, o sea sin ningun acuerdo entre las 30 muestras: mucha
# entropia y poca precision, y eso explica la perdida en el test humano. El
# muestreo estaba ahi para generar variedad; con la ficha de rol la variedad ya
# la da la lista de actores, asi que se puede generar de forma determinista.
#
# Efecto lateral util: sin muestreo el resultado es reproducible, y el ruido de
# 0,0122 entre corridas desaparece.
#
# Espera a que run_v3_semillas.sh termine. NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 AGIX_DEVICE=cuda AGIX_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A_CON="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"
ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v3_agix_v1"

di "esperando a que termine run_v3_semillas.sh"
while ! grep -qE "SEMILLAS TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== variante C: ficha de rol + haz (determinista) ==="

export AGIX_ARRIBA="$A_CON" AGIX_ABAJO="$ABAJO" AGIX_ARRIBA_HAZ=1 AGIX_SEED=20260926

di "--- coherencia"
"$PY" -u src/52_coherencia.py modelo > logs/52_v3haz.log 2>&1 \
  && grep -E "ROLES|TAREAS|ATRIB|^ +(cuota|un solo|sin |roles dist|top|concentracion|eco|ancladas|margen|acierto)" \
     logs/52_v3haz.log | tee -a "$LOG" || di "  fallo 52"

di "--- F1 de arista, referencias v1 y v3"
AGIX_SUFIJO="_v3haz" "$PY" -u src/57_f1_arista.py --modelo > logs/57_v3haz.log 2>&1 \
  && tail -7 logs/57_v3haz.log | tee -a "$LOG" || di "  fallo 57 v1"
AGIX_REF=v3 AGIX_SUFIJO="_v3haz_refv3" "$PY" -u src/57_f1_arista.py --modelo \
  > logs/57_v3haz_refv3.log 2>&1 \
  && tail -7 logs/57_v3haz_refv3.log | tee -a "$LOG" || di "  fallo 57 v3"

di "--- 328 historias humanas (determinista, una corrida basta)"
AGIX_SUFIJO="_v3haz" "$PY" -u src/49_eval_real.py ambos > logs/49_v3haz.log 2>&1 \
  && grep -E "^##### |nuevo - viejo|gana el nuevo" logs/49_v3haz.log | tee -a "$LOG" \
  || di "  fallo 49"

di "--- lectura de salidas"
"$PY" -u src/45_leer_salidas.py nuevo > logs/45_v3haz.log 2>&1

di "HAZ TERMINADO. Leer logs/45_v3haz.log contra logs/45_v3a.log y logs/45_nuevo.log"
