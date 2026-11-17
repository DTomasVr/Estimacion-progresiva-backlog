#!/usr/bin/env bash
# Segundo miembro entrenado con los datos v5 pero otros hiperparametros (bs8,
# lr 9,2e-4), replicando el par que ya existia. Razon: el conjunto de abajo
# funciona porque los miembros yerran distinto, y ahora DOS de los tres estan
# entrenados solo con datos reales, asi que estan correlacionados. Un segundo
# modelo con datos v5 deberia dar las tareas extra sin traer ruido de Jira
# ("Add documentation for the service representation").
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$BASE"
echo "[$(date '+%F %T')] entrenando v5 con bs8 lr9.2e-4" > logs/v5b.log
DATOS=v5 "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_hist_agix_v1 8 192 9.2e-4 4 \
  >> logs/v5b.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/v5b.log | tail -6
echo "[$(date '+%F %T')] V5B TERMINADO" >> logs/v5b.log
