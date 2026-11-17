#!/usr/bin/env bash
# Podado contra SIN PODAR, misma receta y mismo lote EFECTIVO.
#
# De donde sale: la asesora pidio que el modelo funcione bien antes de optimizar
# tiempos, y la poda de vocabulario (250.100 -> 75.360 piezas) es precisamente una
# optimizacion. Habia que medir si cuesta calidad.
#
# Lo que ya se sabia antes de entrenar (results/76_coste_poda.csv): sobre el texto
# de prueba la poda cuesta +0,01 % a +0,05 % de piezas, cero <unk> y la ida y
# vuelta es identica. O sea que se espera un empate. Esto lo comprueba entrenando.
#
# Diseno. Los dos brazos salen del checkpoint CRUDO (no de la cadena hist) y
# entrenan una sola etapa con los datos v5, misma lr, mismas epocas y lote
# efectivo 16. Sin podar solo cabe a lote 4 (sondeo: bs 8 deja 6,46 GB asignados,
# que con la reserva del driver se va a 7,6 de 8,1 y arriesga derrame), asi que se
# usa 4 con acumulacion 4 en LOS DOS brazos: si un brazo llevara otro lote, la
# comparacion mediria el lote y no la poda.
#
# NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/sin_podar.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
cd "$BASE"
: > "$LOG"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

P_OUT="$BASE/models/mt5-small-pruned_v5_agix_v1"
U_OUT="$BASE/models/mt5-small_v5nb_agix_v1"
for d in "$P_OUT" "$U_OUT"; do
  [ -d "$d" ] && { di "ABORTA: $d ya existe, no se sobreescribe"; exit 1; }
done

di "=== poda: dos brazos, lote 4 x acumulacion 4 (efectivo 16), lr 5e-4, 4 epocas ==="

di "brazo P (podado, 75.360 piezas) ~85 min"
DATOS=v5 ACUM=4 "$PY" -u src/09_train_mt5.py models/mt5-small-pruned 4 192 5e-4 4 \
  > logs/09_poda_P.log 2>&1
grep -E "^  epoca|parada|TOTAL|val loss inicial" logs/09_poda_P.log | tee -a "$LOG"
[ -d "$P_OUT" ] || { di "FALLO: no se guardo $P_OUT"; exit 1; }
di "VRAM P: $(grep -oE 'VRAM [0-9.]+ GB' logs/09_poda_P.log | sort -k2 -n | tail -1)"

di "brazo U (sin podar, 250.100 piezas) ~170 min"
DATOS=v5nb ACUM=4 "$PY" -u src/09_train_mt5.py google/mt5-small 4 192 5e-4 4 \
  > logs/09_poda_U.log 2>&1
grep -E "^  epoca|parada|TOTAL|val loss inicial" logs/09_poda_U.log | tee -a "$LOG"
[ -d "$U_OUT" ] || { di "FALLO: no se guardo $U_OUT"; exit 1; }
di "VRAM U: $(grep -oE 'VRAM [0-9.]+ GB' logs/09_poda_U.log | sort -k2 -n | tail -1)"

di "evaluacion: cada brazo con SU parquet de test (~8 min cada uno)"
"$PY" -u src/11_eval_gen.py --test "$P_OUT" beam4_actual > logs/11_poda_P.log 2>&1
grep -E "sF1" logs/11_poda_P.log | tail -4 | tee -a "$LOG"
AGIX_TEST_PARQUET="$BASE/data/processed/s2s_nb_test.parquet" \
  "$PY" -u src/11_eval_gen.py --test "$U_OUT" beam4_actual > logs/11_poda_U.log 2>&1
grep -E "sF1" logs/11_poda_U.log | tail -4 | tee -a "$LOG"

di "comparacion pareada"
"$PY" -u src/79_pareado_poda.py mt5-small-pruned_v5_agix_v1 mt5-small_v5nb_agix_v1 \
  2>&1 | tee -a "$LOG"

di "PODA TERMINADA"
