#!/bin/sh
# NU_CAP=0.25 contra el desplegado (v9bt sin castigo) sobre las 328 historias humanas.
# El brazo base ya esta corrido en esta sesion: 0.6361 / 0.6346 / 0.6382.
BASE="F:/Software Task generation Model"
PY="F:/PBE/venv/Scripts/python.exe"
for S in 20260926 777 31337; do
  echo "===== NU_CAP 0.15 semilla $S ====="
  PBE_DEVICE=cuda PBE_SEED=$S PBE_NU_CAP=0.15 \
    "$PY" "$BASE/src/49_eval_real.py" nuevo 2>&1 | grep -E "sF1 conjunto"
  cp "$BASE/results/49_real.csv" "$BASE/results/49_real_nucap015_s${S}.csv"
done
echo "===== FIN ====="
