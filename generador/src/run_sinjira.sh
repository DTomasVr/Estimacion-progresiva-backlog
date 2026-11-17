#!/usr/bin/env bash
# Entrena y mide las dos variantes que reducen el peso del corpus de Jira en la
# mitad de abajo, mas la actual como control. Las tres desde el MISMO checkpoint
# (el ya entrenado con Jira), asi que "quitar Jira" es dejar de reforzarlo.
#
#   v5 (control)  22.871 jira + 5.403 sinteticas   Jira 79 %
#   v8 (50/50)     5.403 jira + 5.403 sinteticas
#   v7 (sin jira)       0 jira + 5.403 sinteticas
#
# Se mide la FUGA de vocabulario de Jira con lexico derivado por frecuencias
# (src/70_lexico_jira.py): controles 72,7 % en tareas de Jira, 2,4 % en sinteticas.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/sinjira.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export PBE_DEVICE=cuda PBE_SEED=20260926 PBE_CONJUNTO=dos_modelos
export PBE_ARRIBA="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3r_pbe_v1"
cd "$BASE"
: > "$LOG"

for D in v8 v7; do
  echo "########## entrenando abajo con $D" | tee -a "$LOG"
  DATOS=$D "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_hist_pbe_v1 16 192 5e-4 4 \
    > "logs/train_abajo_$D.log" 2>&1
  grep -E "^  epoca|parada|TOTAL" "logs/train_abajo_$D.log" | tee -a "$LOG"
done

for par in "control_v5|mt5-small-pruned_hist_pbe_v1_v5_pbe_v1" \
           "mitad_v8|mt5-small-pruned_hist_pbe_v1_v8_pbe_v1" \
           "sinjira_v7|mt5-small-pruned_hist_pbe_v1_v7_pbe_v1"; do
  etq="${par%%|*}"; m="${par#*|}"
  echo "########## $etq" | tee -a "$LOG"
  if [ ! -d "models/$m" ]; then echo "   no existe models/$m" | tee -a "$LOG"; continue; fi
  export PBE_ABAJO="$BASE/models/$m"
  "$PY" -u src/52_coherencia.py modelo > "logs/52_$etq.log" 2>&1 \
    && grep -E "FUGAS|concentracion|ancladas|acierto|cuota" "logs/52_$etq.log" \
       | sed 's/^/   /' | tee -a "$LOG" || echo "   fallo 52" | tee -a "$LOG"
  "$PY" -u src/68_cobertura.py 30 > "logs/68_$etq.log" 2>&1 \
    && grep -E "cubiertos|completos|historias [0-9]" "logs/68_$etq.log" \
       | sed 's/^/   /' | tee -a "$LOG" || echo "   fallo 68" | tee -a "$LOG"
  "$PY" -u src/45_leer_salidas.py nuevo > "logs/45_$etq.log" 2>&1
done
echo "SINJIRA TERMINADO" | tee -a "$LOG"
