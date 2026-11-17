#!/usr/bin/env bash
# Segunda mitad del experimento de poda: espera a que el brazo PODADO acabe y
# sigue con el brazo SIN PODAR, las dos evaluaciones y el pareado.
#
# Va aparte de run_sin_podar.sh porque el brazo podado ya esta entrenando: este
# guion no vuelve a entrenarlo. El brazo sin podar necesita ~4,4 GB asignados, asi
# que no puede compartir la GPU con nada mas; por eso espera.
#
# NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/sin_podar.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

P_OUT="$BASE/models/mt5-small-pruned_v5_pbe_v1"
U_OUT="$BASE/models/mt5-small_v5nb_pbe_v1"

# se espera la linea "TOTAL <min> min" que 09_train_mt5 imprime al acabar: es mas
# fiable que buscar el proceso (wmic ya no viene en Windows 11)
di "esperando a que termine el brazo podado"
for _ in $(seq 1 120); do
  grep -qE "^TOTAL [0-9.]+ min" logs/09_poda_P.log && break
  sleep 60
done
grep -qE "^TOTAL [0-9.]+ min" logs/09_poda_P.log || { di "FALLO: el brazo P no termino en 2 h"; exit 1; }
sleep 20
grep -E "^  epoca|parada|TOTAL" logs/09_poda_P.log | tee -a "$LOG"
[ -d "$P_OUT" ] || { di "FALLO: no existe $P_OUT"; exit 1; }
di "VRAM maxima del brazo P: $(grep -oE 'VRAM [0-9.]+ GB' logs/09_poda_P.log | sort -k2 -n | tail -1)"

if [ -d "$U_OUT" ]; then
  di "ABORTA: $U_OUT ya existe"; exit 1
fi
di "brazo U (sin podar, 250.100 piezas), lote 4 x acumulacion 4, ~2 h 20"
DATOS=v5nb ACUM=4 "$PY" -u src/09_train_mt5.py google/mt5-small 4 192 5e-4 4 \
  > logs/09_poda_U.log 2>&1
grep -E "^  epoca|parada|TOTAL|val loss inicial" logs/09_poda_U.log | tee -a "$LOG"
[ -d "$U_OUT" ] || { di "FALLO: no se guardo $U_OUT"; exit 1; }
di "VRAM maxima del brazo U: $(grep -oE 'VRAM [0-9.]+ GB' logs/09_poda_U.log | sort -k2 -n | tail -1)"

di "evaluacion: cada brazo con SU parquet de test"
"$PY" -u src/11_eval_gen.py --test "$P_OUT" beam4_actual > logs/11_poda_P.log 2>&1
grep -E "sF1" logs/11_poda_P.log | tail -3 | tee -a "$LOG"
PBE_TEST_PARQUET="$BASE/data/processed/s2s_nb_test.parquet" \
  "$PY" -u src/11_eval_gen.py --test "$U_OUT" beam4_actual > logs/11_poda_U.log 2>&1
grep -E "sF1" logs/11_poda_U.log | tail -3 | tee -a "$LOG"

di "comparacion pareada"
"$PY" -u src/79_pareado_poda.py mt5-small-pruned_v5_pbe_v1 mt5-small_v5nb_pbe_v1 \
  2>&1 | tee -a "$LOG"
di "PODA TERMINADA"
