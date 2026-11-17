#!/usr/bin/env bash
# Ultima variante: muestrear las TAREAS en lugar de generarlas con haz.
#
# De donde sale. La concentracion de arranques de dos palabras en las tareas es
#   2,6 %  las escritas por personas en el corpus de Jira (150.753 tareas)
#  28,1 %  el profesor LLM v3
#  83,6 %  nuestro alumno
# El haz de la mitad de abajo maximiza verosimilitud y por eso amplifica la
# formulacion mas probable. PBE_ABAJO_MUESTREO=1 lo cambia por muestreo. Es una
# prueba gratis: no toca datos ni pesos.
#
# Se mide sobre el metodo B, que es el que mejor leia. Espera a run_v3_conc.sh.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PBE_DEVICE=cuda PBE_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A_CON="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3r_pbe_v1"
A_PROD="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
B_PROD="$BASE/models/mt5-small-pruned_hist_30sec"
ABAJO="$BASE/models/mt5-small-pruned_hist_pbe_v1_v3_pbe_v1"

di "esperando a que termine run_v3_conc.sh"
while ! grep -qE "CONCENTRACION TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== muestrear las tareas en lugar de haz ==="

export PBE_ABAJO_MUESTREO=1 PBE_SEED=20260926

for par in "v3rmu|$A_CON|$ABAJO" "basemu|$A_PROD|$B_PROD"; do
  etq="${par%%|*}"; resto="${par#*|}"; arriba="${resto%%|*}"; abajo="${resto##*|}"
  di "--- $etq (tareas muestreadas)"
  PBE_ARRIBA="$arriba" PBE_ABAJO="$abajo" \
    "$PY" -u src/52_coherencia.py modelo > "logs/52_${etq}.log" 2>&1 \
    && grep -E "concentracion|ancladas|acierto|cuota|un solo" "logs/52_${etq}.log" | tee -a "$LOG" \
    || di "  fallo 52 en $etq"
  PBE_ARRIBA="$arriba" PBE_ABAJO="$abajo" PBE_SUFIJO="_${etq}" \
    "$PY" -u src/57_f1_arista.py --modelo > "logs/57_${etq}.log" 2>&1 \
    && tail -7 "logs/57_${etq}.log" | tee -a "$LOG" || di "  fallo 57 en $etq"
done

di "--- lectura de salidas del metodo B con tareas muestreadas"
PBE_ARRIBA="$A_CON" PBE_ABAJO="$ABAJO" \
  "$PY" -u src/45_leer_salidas.py nuevo > logs/45_v3rmu.log 2>&1

di "TAREAS TERMINADO. Leer logs/45_v3rmu.log contra logs/45_v3r.log"
