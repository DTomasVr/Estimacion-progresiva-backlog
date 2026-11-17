#!/usr/bin/env bash
# Dos barridos, en este orden:
#
#  A. SELECCION SUBMODULAR contra el "colapsar y ordenar por relevancia" actual.
#     La formulacion esta en src/19_pipeline.py (localizacion de instalaciones).
#     tau es la regla de parada: 0 = sin parada (tamano fijo k), >0 = el contenido
#     decide cuantas tareas lleva cada historia.
#
#  B. TERCER MIEMBRO del conjunto de abajo: el nuevo modelo entrenado con datos
#     v5 y otros hiperparametros, contra el de solo datos reales que hay puesto.
#     Hipotesis: el de datos reales trae ruido de Jira ("Add documentation for
#     the service representation") porque es el unico que no vio datos de
#     producto; uno con datos v5 deberia dar las tareas extra sin ese ruido.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/67_submodular.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export AGIX_DEVICE=cuda AGIX_SEED=20260926 AGIX_CONJUNTO=dos_modelos
export AGIX_ARRIBA="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"
export AGIX_ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v5_agix_v1"
REAL="$BASE/models/mt5-small-pruned_hist_agix_v1"
NUEVO="$BASE/models/mt5-small-pruned_hist_agix_v1_v5_agix_v1"   # placeholder, se fija abajo
cd "$BASE"
: > "$LOG"

mide() {   # etiqueta
  local etq="$1"
  "$PY" -u src/52_coherencia.py modelo > "logs/52_$etq.log" 2>&1 \
    && grep -E "TAREAS  |concentracion|ancladas|acierto|cuota" "logs/52_$etq.log" \
       | sed "s/^/   /" | tee -a "$LOG" \
    || echo "   fallo" | tee -a "$LOG"
}

echo "########## A. seleccion submodular" | tee -a "$LOG"
export AGIX_ABAJO3="$REAL"
for cfg in "actual|0|" "sub_k|0|1" "sub_t15|0.15|1" "sub_t25|0.25|1"; do
  etq="${cfg%%|*}"; resto="${cfg#*|}"; tau="${resto%%|*}"; sub="${resto##*|}"
  echo "=== $etq  (submodular=${sub:-no}, tau=$tau)" | tee -a "$LOG"
  if [ -n "$sub" ]; then export AGIX_SUBMODULAR=1; else unset AGIX_SUBMODULAR; fi
  AGIX_TAU="$tau" mide "$etq"
done
unset AGIX_SUBMODULAR

echo "########## B. tercer miembro del conjunto" | tee -a "$LOG"
# el nombre del modelo nuevo lo pone 09_train_mt5.py como <base>_<DATOS>_agix_v1,
# y al repetir DATOS=v5 sobre la misma base colisiona con el que ya existe, asi
# que se busca el mas reciente que contenga v5 y no sea el actual
for cand in $(ls -dt models/*v5* 2>/dev/null); do
  if [ "$BASE/$cand" != "$AGIX_ABAJO" ]; then NUEVO="$BASE/$cand"; break; fi
done
echo "   tercer miembro nuevo: $(basename "$NUEVO")" | tee -a "$LOG"
for par in "t_real|$REAL" "t_nuevo|$NUEVO"; do
  etq="${par%%|*}"; m3="${par#*|}"
  echo "=== $etq  ($(basename "$m3"))" | tee -a "$LOG"
  AGIX_ABAJO3="$m3" mide "$etq"
done

echo "SUBMODULAR TERMINADO" | tee -a "$LOG"
