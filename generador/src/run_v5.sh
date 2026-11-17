#!/usr/bin/env bash
# v5: misma tanda v3 pero con TOPE DE ARRANQUES en los datos de tarea.
#
# De donde sale. El colapso de plantilla es el defecto grave (87 % de las tareas
# con cinco arranques, contra 2,6 % de las personas) y el barrido de
# decodificadores demostro que NO se puede arreglar ahi: cada 10 puntos de
# plantilla que se quitan cuestan 5-6 de atribucion, sin codo.
#
# Y la culpa esta repartida: la mitad de abajo entrenada solo con datos reales de
# Jira sale al 41 %, y al destilar sube al 87 %. O sea que el haz y el tamano del
# modelo amplifican de 2,6 % a 41 %, y el profesor LLM anade el resto.
#
# v5 ataca esa segunda mitad sin regenerar nada: se limita cada arranque de dos
# palabras a DOS tareas por backlog en los datos del profesor. Eso baja el
# sintetico de 27,9 % a 18,8 % perdiendo un 22 % de filas de tarea.
#
# NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ABAJO_BASE="models/mt5-small-pruned_hist_agix_v1"
ABAJO_V5="$BASE/models/mt5-small-pruned_hist_agix_v1_v5_agix_v1"
ABAJO_V3="$BASE/models/mt5-small-pruned_hist_agix_v1_v3_agix_v1"
A_CON="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"

di "=== v5: tope de arranques en los datos de tarea ==="
di "fase 1: entrenando abajo con v5 (~35 min)"
DATOS=v5 "$PY" -u src/09_train_mt5.py "$ABAJO_BASE" 16 192 5e-4 4 \
  > logs/44_train_abajo_v5.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/44_train_abajo_v5.log | tee -a "$LOG"
di "VRAM maxima: $(grep -oE 'VRAM [0-9.]+ GB' logs/44_train_abajo_v5.log | sort -k2 -n | tail -1)"
[ -d "$ABAJO_V5" ] || { di "FALLO: no existe $ABAJO_V5"; exit 1; }

export AGIX_DEVICE=cuda AGIX_CONJUNTO=no AGIX_SEED=20260926 AGIX_ARRIBA="$A_CON"

di "fase 2: comparando abajo v5 contra abajo v3, con la misma mitad de arriba"
for par in "v3abajo|$ABAJO_V3" "v5abajo|$ABAJO_V5"; do
  etq="${par%%|*}"; abajo="${par#*|}"
  AGIX_ABAJO="$abajo" "$PY" -u src/52_coherencia.py modelo > "logs/52_$etq.log" 2>&1 \
    && grep -E "concentracion|ancladas|acierto|TAREAS" "logs/52_$etq.log" \
       | sed "s/^/  [$etq] /" | tee -a "$LOG" || di "  fallo 52 en $etq"
  AGIX_ABAJO="$abajo" AGIX_SUFIJO="_$etq" "$PY" -u src/57_f1_arista.py --modelo \
    > "logs/57_$etq.log" 2>&1 && tail -6 "logs/57_$etq.log" | sed "s/^/  [$etq] /" \
    | tee -a "$LOG" || di "  fallo 57 en $etq"
done

di "fase 3: lectura de salidas con abajo v5"
AGIX_ABAJO="$ABAJO_V5" "$PY" -u src/45_leer_salidas.py nuevo > logs/45_v5.log 2>&1

di "V5 TERMINADO"
