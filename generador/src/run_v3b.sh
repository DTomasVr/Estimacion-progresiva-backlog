#!/usr/bin/env bash
# Seguimiento de la tanda v3, tras el resultado negativo del metodo A en las 328
# historias humanas (0,6491 -> 0,6215, IC95 [-0,047; -0,007], gana 1 de 6).
#
#   fase 1  entrenar arriba SIN la duplicacion despublicada (v3b), para aislar
#           si la perdida viene de ahi. v3b baja el sintetico del 59 % al 42 %.
#   fase 2  medir v3b igual que las otras
#   fase 3  comparacion ESPEJO: el mismo F1 de arista contra las referencias v3,
#           que tienen varios roles y tareas por capas. Ninguna de las dos
#           referencias es neutral (las v1 favorecen a produccion, las v3 al
#           nuevo), por eso se reportan las dos y decide el test humano.
#   fase 4  rehacer la tabla
#
# Espera a que run_v3.sh termine. NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ARRIBA_BASE="models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1"
A_PROD="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
B_PROD="$BASE/models/mt5-small-pruned_hist_30sec"
A_SIN="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3a_agix_v1"
A_CON="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"
A_B="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3b_agix_v1"
ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v3_agix_v1"

di "esperando a que termine run_v3.sh"
while ! grep -qE "^\[.*\] TERMINADO\.|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== seguimiento v3b + espejo ==="

export AGIX_DEVICE=cuda AGIX_CONJUNTO=no

# ---------------------------------------------------------------- fase 1
di "fase 1: entrenando arriba sin la duplicacion despublicada (~4 min)"
DATOS=v3b "$PY" -u src/09_train_mt5.py "$ARRIBA_BASE" 32 48 5e-4 4 \
  > logs/44_train_arriba_v3b.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/44_train_arriba_v3b.log | tee -a "$LOG"
[ -d "$A_B" ] || { di "FALLO: no existe $A_B"; exit 1; }

# ---------------------------------------------------------------- fase 2
di "fase 2: midiendo v3b"
export AGIX_ARRIBA="$A_B" AGIX_ABAJO="$ABAJO" AGIX_SUFIJO="_v3b"
"$PY" -u src/52_coherencia.py modelo > logs/52_v3b.log 2>&1 \
  && grep -E "ROLES|TAREAS|ATRIB|^ +(cuota|un solo|sin |roles dist|top|eco|ancladas|margen|acierto)" \
     logs/52_v3b.log | tee -a "$LOG" || di "  fallo 52"
"$PY" -u src/57_f1_arista.py --modelo > logs/57_v3b.log 2>&1 \
  && tail -7 logs/57_v3b.log | tee -a "$LOG" || di "  fallo 57"
"$PY" -u src/49_eval_real.py ambos > logs/49_v3b.log 2>&1 \
  && grep -E "^##### |nuevo - viejo|gana el nuevo" logs/49_v3b.log | tee -a "$LOG" \
  || di "  fallo 49"
"$PY" -u src/45_leer_salidas.py nuevo > logs/45_v3b.log 2>&1
unset AGIX_SUFIJO

# ---------------------------------------------------------------- fase 3
di "fase 3: espejo, F1 de arista contra referencias v3"
for trio in "base|$A_PROD|$B_PROD" "v3a|$A_SIN|$ABAJO" "v3r|$A_CON|$ABAJO" "v3b|$A_B|$ABAJO"; do
  etq="${trio%%|*}"; resto="${trio#*|}"; arriba="${resto%%|*}"; abajo="${resto##*|}"
  di "--- $etq contra referencias v3"
  AGIX_ARRIBA="$arriba" AGIX_ABAJO="$abajo" AGIX_REF=v3 AGIX_SUFIJO="_${etq}_refv3" \
    "$PY" -u src/57_f1_arista.py --modelo > "logs/57_${etq}_refv3.log" 2>&1 \
    && tail -7 "logs/57_${etq}_refv3.log" | tee -a "$LOG" \
    || di "  fallo espejo en $etq"
done

# ---------------------------------------------------------------- fase 4
di "fase 4: tabla"
"$PY" -u src/59_tabla_v3.py > logs/59_tabla.log 2>&1 \
  && cat logs/59_tabla.log | tee -a "$LOG" || di "  fallo la tabla"

di "SEGUIMIENTO TERMINADO"
