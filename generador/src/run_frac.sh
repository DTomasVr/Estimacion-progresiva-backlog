#!/usr/bin/env bash
# Barrido de FRACCION DE EPOCA para la mitad de arriba.
#
# Por que. Las tres variantes de arriba (v3a, v3r, v3b) pararon en la epoca 1, o
# sea que el minimo de validacion estaba ya al final de la primera pasada; el
# optimo puede estar ANTES. Si con media epoca se conservan los roles y se pierde
# menos fidelidad, no hay que elegir entre el banco y el producto.
#
# 09_train_mt5.py --sub entrena una fraccion y nombra la salida con la fraccion,
# asi que no colisiona con nada.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/frac.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export AGIX_DEVICE=cuda AGIX_SEED=20260926 AGIX_CONJUNTO=dos_modelos
export AGIX_ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v5_agix_v1"
ARRIBA_BASE="models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1"
cd "$BASE"
: > "$LOG"

for f in 0.15 0.35 0.6; do
  echo "########## fraccion $f" | tee -a "$LOG"
  DATOS=v3r "$PY" -u src/09_train_mt5.py --sub "$ARRIBA_BASE" 32 48 5e-4 "$f" \
    > "logs/frac_${f}_train.log" 2>&1
  M=$(ls -dt models/*v3r_sub${f}* 2>/dev/null | head -1)
  if [ -z "$M" ]; then echo "  no se creo el modelo" | tee -a "$LOG"; continue; fi
  echo "  modelo: $(basename "$M")" | tee -a "$LOG"
  grep -E "^  epoca|val loss" "logs/frac_${f}_train.log" | tail -2 | tee -a "$LOG"
  export AGIX_ARRIBA="$BASE/$M"
  "$PY" -u src/52_coherencia.py modelo > "logs/52_frac$f.log" 2>&1 \
    && grep -E "un solo|cuota|concentracion|ancladas|acierto" "logs/52_frac$f.log" \
       | sed 's/^/  /' | tee -a "$LOG"
  "$PY" -u src/68_cobertura.py 30 > "logs/68_frac$f.log" 2>&1 \
    && grep -E "cubiertos|completos|historias [0-9]" "logs/68_frac$f.log" \
       | sed 's/^/  /' | tee -a "$LOG"
  AGIX_SUFIJO="_frac$f" "$PY" -u src/49_eval_real.py ambos > "logs/49_frac$f.log" 2>&1 \
    && grep -E "^##### |nuevo - viejo" "logs/49_frac$f.log" | sed 's/^/  /' | tee -a "$LOG"
done
echo "FRAC TERMINADO" | tee -a "$LOG"
