#!/usr/bin/env bash
# Cierre definitivo con la configuracion final (cobertura de rasgos + cobertura
# de roles mu=0,15 + reordenador de beneficios lambda=0,15).
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/cierre.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export PBE_DEVICE=cuda PBE_CONJUNTO=dos_modelos
cd "$BASE"
: > "$LOG"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

di "1. 328 historias humanas: actual contra PRODUCCION, tres semillas"
export PBE_ARRIBA="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
export PBE_ABAJO="$BASE/models/mt5-small-pruned_hist_30sec"
for s in 20260926 777 31337; do
  PBE_SEED=$s PBE_SUFIJO="_cie_s$s" "$PY" -u src/49_eval_real.py ambos \
    > "logs/49_cie_s$s.log" 2>&1
  a=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_cie_s$s.log" | tail -1)
  b=$(grep -oE "nuevo: sF1 conjunto [0-9.]+" "logs/49_cie_s$s.log" | tail -1)
  di "   semilla $s | actual $a | produccion $b"
done
unset PBE_ARRIBA PBE_ABAJO

di "2. banco de 64 proyectos (~40 min)"
PBE_SEED=20260926 "$PY" -u src/20_banco_llms.py 64 > logs/20_banco_cie.log 2>&1 \
  && "$PY" -u src/22_resumen_banco.py > logs/22_cie.log 2>&1 \
  && cat results/22_banco_pareado.csv | tee -a "$LOG" || di "   fallo el banco"

di "3. ejemplos de la demo, con el glosario y la cobertura de roles"
PBE_SEED=20260926 "$PY" -u src/40_ejemplos_dominio.py > logs/40_cie.log 2>&1 \
  && grep -E "historias," logs/40_cie.log | tail -9 | tee -a "$LOG" || di "   fallo"

di "CIERRE TERMINADO"
