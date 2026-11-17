#!/bin/sh
# v8ff desplegado contra v9bt sobre las 328 historias humanas, 3 semillas.
# La semilla 20260926 ya esta corrida en esta sesion; aqui van las otras dos.
BASE="F:/Software Task generation Model"
PY="F:/PBE/venv/Scripts/python.exe"
AB="$BASE/models/mt5-small-pruned_hist_pbe_v1_v5_pbe_v1"
V8="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v8ff_pbe_v1"
V9="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v9bt_pbe_v1"
for S in 777 31337; do
  for N in v8ff v9bt; do
    if [ "$N" = "v8ff" ]; then AR="$V8"; else AR="$V9"; fi
    echo "===== $N semilla $S ====="
    PBE_DEVICE=cuda PBE_SEED=$S PBE_ARRIBA="$AR" PBE_ABAJO="$AB" \
      PBE_FICHA_ROL=1 PBE_MODO_FUNCION=1 \
      "$PY" "$BASE/src/49_eval_real.py" nuevo 2>&1 | grep -E "sF1 conjunto"
    cp "$BASE/results/49_real.csv" "$BASE/results/49_real_${N}_s${S}.csv"
  done
done
echo "===== FIN ====="
