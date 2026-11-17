#!/usr/bin/env bash
# La opcion mas barata: modelos de PRODUCCION + reparacion del rol en inferencia.
#
# De donde sale. El validador de rol se rendia cuando casi todas las candidatas
# traian un rol invalido (el taller de coches, con 7 de 8 diciendo "As a service
# appointment"), porque el respaldo que puse devolvia todas las candidatas.
# PBE_REPARAR_ROL=1 no se rinde: sustituye el hueco del rol por un actor del
# plantel del sector, conservando la capacidad de la historia.
#
# Por que importa. Las tres variantes reentrenadas cuestan ~0,014 de sF1 en las
# 328 historias humanas (media de tres semillas para el metodo B: -0,0144). Esta
# no reentrena nada, asi que el coste deberia ser mucho menor, y arregla el
# defecto que el usuario ve.
#
# Espera a run_v3_tareas.sh. NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 PBE_DEVICE=cuda PBE_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A_PROD="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
B_PROD="$BASE/models/mt5-small-pruned_hist_30sec"

di "esperando a que termine run_v3_tareas.sh"
while ! grep -qE "TAREAS TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== produccion + reparacion del rol (sin reentrenar) ==="

export PBE_REPARAR_ROL=1 PBE_ARRIBA="$A_PROD" PBE_ABAJO="$B_PROD"

di "--- coherencia"
PBE_SEED=20260926 "$PY" -u src/52_coherencia.py modelo > logs/52_rep.log 2>&1 \
  && grep -E "ROLES|HIST|TAREAS|ATRIB|^ +(cuota|un solo|roles dist|top|verbo|clausula|concentracion|eco|ancladas|acierto)" \
     logs/52_rep.log | tee -a "$LOG" || di "  fallo 52"

di "--- F1 de arista"
PBE_SEED=20260926 PBE_SUFIJO="_rep" "$PY" -u src/57_f1_arista.py --modelo \
  > logs/57_rep.log 2>&1 && tail -7 logs/57_rep.log | tee -a "$LOG" || di "  fallo 57"

di "--- 328 historias humanas, tres semillas (el brazo viejo es el MISMO modelo sin reparar)"
for s in 20260926 777 31337; do
  PBE_SEED="$s" PBE_SUFIJO="_rep_s$s" "$PY" -u src/49_eval_real.py ambos \
    > "logs/49_rep_s$s.log" 2>&1
  v=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_rep_s$s.log" | tail -1)
  n=$(grep -oE "nuevo: sF1 conjunto [0-9.]+" "logs/49_rep_s$s.log" | tail -1)
  d=$(grep -E "nuevo - viejo" "logs/49_rep_s$s.log" | head -1)
  di "semilla $s reparado | $v | $n | $d"
done

di "--- lectura de salidas"
"$PY" -u src/45_leer_salidas.py nuevo > logs/45_rep.log 2>&1

di "REPARAR TERMINADO"
