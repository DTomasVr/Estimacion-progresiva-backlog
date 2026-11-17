#!/usr/bin/env bash
# Vuelve a medir la coherencia de TODAS las variantes con la metrica de
# concentracion de arranques, que se anadio despues de medir base, v3a y v3r.
#
# Por que hacia falta: medir el relleno con una lista fija de frases fue un
# error. La tanda v3 no quito la plantilla, la cambio ("Implement a feature to X"
# -> "Store X in the database"), y la lista vieja daba 0 %. La concentracion mide
# que parte de las tareas empieza por uno de los cinco arranques de dos palabras
# mas frecuentes, asi que detecta cualquier plantilla sin tener que anticiparla.
#
# Primer dato: v3b sale al 83,6 % con solo cinco arranques, mientras su profesor
# esta al 28,1 %. El alumno amplifica la plantilla del profesor.
#
# Espera a que run_v3_haz.sh termine. NO EDITAR MIENTRAS CORRE.
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

di "esperando a que termine run_v3_haz.sh"
while ! grep -qE "HAZ TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== concentracion de arranques en todas las variantes ==="

for trio in "base|$A_PROD|$B_PROD" "v3a|$A_SIN|$ABAJO" "v3r|$A_CON|$ABAJO"; do
  etq="${trio%%|*}"; resto="${trio#*|}"; arriba="${resto%%|*}"; abajo="${resto##*|}"
  AGIX_ARRIBA="$arriba" AGIX_ABAJO="$abajo" AGIX_SEED=20260926 \
    "$PY" -u src/52_coherencia.py modelo > "logs/52_${etq}_conc.log" 2>&1 \
    && di "$etq: $(grep -E 'concentracion' "logs/52_${etq}_conc.log" | head -1)" \
    || di "  fallo en $etq"
done

di "y el profesor, para comparar alumno contra maestro"
AGIX_DEVICE=cpu "$PY" -u src/52_coherencia.py profesor > logs/52_prof_v1_conc.log 2>&1 \
  && di "profesor v1: $(grep -E 'concentracion' logs/52_prof_v1_conc.log | head -1)"

di "tabla final"
"$PY" -u src/59_tabla_v3.py > logs/59_tabla.log 2>&1 \
  && cat logs/59_tabla.log | tee -a "$LOG" || di "  fallo la tabla"

di "CONCENTRACION TERMINADO"
