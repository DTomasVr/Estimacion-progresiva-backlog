#!/usr/bin/env bash
# Cierre: todo lo que quedo desactualizado al anadir el reordenador de beneficios
# (lambda 0,15), que cambia que historias se seleccionan.
#
#   1. las 328 historias humanas, tres semillas
#   2. el banco de 64 proyectos (solo nuestro brazo; los LLM estaban guardados)
#   3. ejemplos de la demo regenerados
#   4. coherencia y cobertura finales, para la tabla
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/final.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
export PBE_DEVICE=cuda PBE_CONJUNTO=dos_modelos
cd "$BASE"
: > "$LOG"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

di "1. las 328 historias humanas, tres semillas"
for s in 20260926 777 31337; do
  PBE_SEED=$s PBE_SUFIJO="_fin_s$s" "$PY" -u src/49_eval_real.py ambos \
    > "logs/49_fin_s$s.log" 2>&1
  v=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_fin_s$s.log" | tail -1)
  n=$(grep -oE "nuevo: sF1 conjunto [0-9.]+" "logs/49_fin_s$s.log" | tail -1)
  d=$(grep -E "nuevo - viejo" "logs/49_fin_s$s.log" | head -1)
  di "   semilla $s | $v | $n | $d"
done

di "2. banco de 64 proyectos (~40 min)"
PBE_SEED=20260926 "$PY" -u src/20_banco_llms.py 64 > logs/20_banco_fin.log 2>&1 \
  && "$PY" -u src/22_resumen_banco.py > logs/22_fin.log 2>&1 \
  && cat results/22_banco_pareado.csv | tee -a "$LOG" \
  || di "   fallo el banco"

di "3. ejemplos de la demo"
PBE_SEED=20260926 "$PY" -u src/40_ejemplos_dominio.py > logs/40_fin.log 2>&1 \
  && grep -E "historias," logs/40_fin.log | tail -9 | tee -a "$LOG" || di "   fallo"

di "4. coherencia y cobertura finales"
PBE_SEED=20260926 "$PY" -u src/52_coherencia.py modelo > logs/52_fin.log 2>&1 \
  && grep -E "un solo|cuota|FUGAS|verbo|clausula|concentracion|ancladas|acierto" logs/52_fin.log \
     | tee -a "$LOG"
PBE_SEED=20260926 "$PY" -u src/68_cobertura.py 30 > logs/68_fin.log 2>&1 \
  && grep -E "cobertura suave|cubiertos|completos|historias [0-9]" logs/68_fin.log | tee -a "$LOG"

di "FINAL TERMINADO"
