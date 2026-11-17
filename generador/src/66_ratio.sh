#!/usr/bin/env bash
# Barrido del punto de operacion para el NUMERO DE TAREAS, que es el defecto de
# primer orden que queda: generamos ~la mitad de las tareas de la referencia y el
# texto de cada tarea es la entrada de la cadena de estimacion, asi que faltar
# tareas es faltar trabajo.
#
# Se varian dos cosas, las dos en inferencia y sin reentrenar:
#   PBE_DUP_INTRA  umbral de deduplicacion DENTRO de una historia (hoy 0,95)
#   PBE_ABAJO3     un tercer generador en el conjunto de abajo
#
# Se mira que el numero suba SIN que se hundan la concentracion de plantilla ni
# la atribucion tarea->historia.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/66_ratio.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export PBE_DEVICE=cuda PBE_SEED=20260926 PBE_CONJUNTO=dos_modelos
export PBE_ARRIBA="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3r_pbe_v1"
export PBE_ABAJO="$BASE/models/mt5-small-pruned_hist_pbe_v1_v5_pbe_v1"
M3="$BASE/models/mt5-small-pruned_hist_pbe_v1_v3_pbe_v1"
cd "$BASE"
: > "$LOG"

corre() {   # etiqueta, dup_intra, tercer_modelo("" = ninguno)
  local etq="$1" dup="$2" m3="$3"
  echo "=== $etq  (dup_intra=$dup, tercer modelo: ${m3:-no})" | tee -a "$LOG"
  if [ -n "$m3" ]; then export PBE_ABAJO3="$m3"; else unset PBE_ABAJO3; fi
  PBE_DUP_INTRA="$dup" "$PY" -u src/52_coherencia.py modelo > "logs/52_$etq.log" 2>&1 \
    && grep -E "TAREAS  |concentracion|ancladas|acierto" "logs/52_$etq.log" \
       | sed "s/^/   /" | tee -a "$LOG" \
    || echo "   fallo" | tee -a "$LOG"
}

corre "r_base"      0.95 ""
corre "r_dup97"     0.97 ""
corre "r_dup99"     0.99 ""
corre "r_tres"      0.95 "$M3"
corre "r_tres_97"   0.97 "$M3"

echo "RATIO TERMINADO" | tee -a "$LOG"
