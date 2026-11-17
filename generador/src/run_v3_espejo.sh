#!/usr/bin/env bash
# Comparacion ESPEJO: el mismo F1 de arista pero contra las 40 descripciones
# reservadas de la tanda v3, cuyas referencias tienen varios roles y tareas por
# capas. Ninguno de los dos modelos se entreno con esas 40 descripciones.
#
# Ninguna de las dos referencias es neutral: las v1 favorecen al modelo de
# produccion y las v3 al nuevo. Por eso se reportan LAS DOS, y la prueba que
# decide es la de las 328 historias humanas, que no las escribio ningun modelo.
#
# Se lanza cuando run_v3.sh haya terminado.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 AGIX_DEVICE=cuda AGIX_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A_PROD="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
B_PROD="$BASE/models/mt5-small-pruned_hist_30sec"
A_SIN="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3a_agix_v1"
A_CON="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"
ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v3_agix_v1"

di "=== espejo: F1 de arista contra referencias v3 ==="
for trio in "base|$A_PROD|$B_PROD" "v3a|$A_SIN|$ABAJO" "v3r|$A_CON|$ABAJO"; do
  etq="${trio%%|*}"; resto="${trio#*|}"; arriba="${resto%%|*}"; abajo="${resto##*|}"
  di "--- $etq contra referencias v3"
  AGIX_ARRIBA="$arriba" AGIX_ABAJO="$abajo" AGIX_REF=v3 AGIX_SUFIJO="_${etq}_refv3" \
    "$PY" -u src/57_f1_arista.py --modelo > "logs/57_${etq}_refv3.log" 2>&1 \
    && tail -7 "logs/57_${etq}_refv3.log" | tee -a "$LOG" \
    || di "  fallo en $etq"
done
di "espejo TERMINADO"
