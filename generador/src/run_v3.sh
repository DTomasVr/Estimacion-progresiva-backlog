#!/usr/bin/env bash
# Tanda v3: corregir los roles y la coherencia historia->tarea.
# Se lanza DESPUES de que termine src/54_generar_v3.py.
#
#   fase 1  mezclar real + sintetico v3. Escribe TRES conjuntos:
#             s2s_v3_*   abajo, historia -> tareas
#             s2s_v3a_*  arriba sin ficha, mas la variante sin publico
#             s2s_v3r_*  arriba con ficha de rol Y sin ficha
#   fase 2  adaptar las dos mitades desde los modelos de SOLO DATOS REALES, no
#           desde los destilados v1: para no arrastrar el habito de un solo rol
#           por backlog que traia la tanda v1
#   fase 3  linea base: el modelo de produccion, con las mismas medidas
#   fase 4  metodo A (filtrar): arriba sin ficha + validador de rol
#   fase 5  metodo B (construir): arriba con ficha + planificador de roles
#   fase 6  ablacion de los filtros
#
# Todas las pasadas usan UN SOLO generador abajo (PBE_CONJUNTO=no). Con el
# conjunto de dos se mezclaria el modelo nuevo con el viejo y la comparacion no
# seria limpia.
#
# Una mitad a la vez: dos entrenamientos en el mismo proceso desbordan los 8 GB
# de VRAM y Windows lo resuelve en silencio con RAM compartida, 10x mas lento.
# Referencia de las tandas anteriores: abajo bs16 = 27 min y 6,16 GB de pico;
# arriba bs32 = 4 min y 6,14 GB. El limite seguro es 6,5 GB.
#
# NO EDITAR MIENTRAS CORRE: bash relee el fichero por desplazamiento de bytes.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"

ARRIBA_BASE="models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1"
ABAJO_BASE="models/mt5-small-pruned_hist_pbe_v1"
A_PROD="$BASE/models/mt5-small-pruned_o2ofmt_30sec"
B_PROD="$BASE/models/mt5-small-pruned_hist_30sec"
A_SIN="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3a_pbe_v1"
A_CON="$BASE/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v3r_pbe_v1"
ABAJO="$BASE/models/mt5-small-pruned_hist_pbe_v1_v3_pbe_v1"

di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; }

di "=== tanda v3: roles y coherencia ==="

# descargar el LLM profesor de la VRAM: 5 GB ocupados harian desbordar el
# entrenamiento a RAM compartida sin dar error, 10x mas lento
for M in qwen2.5:7b mistral:7b gemma2:9b llama3.1:8b; do
  curl -s -m 30 http://127.0.0.1:11434/api/generate \
    -d "{\"model\": \"$M\", \"keep_alive\": 0}" > /dev/null 2>&1 || true
done
sleep 8
di "VRAM tras descargar Ollama: $(vram)"

# ---------------------------------------------------------------- fase 1
di "fase 1: mezclando datos v3"
PBE_SYNTH=backlogs_v3_qwen2.5_7b.jsonl PBE_PREF=v3 \
  "$PY" -u src/43_datos_mixtos.py > logs/43_mezcla_v3.log 2>&1 \
  || { di "FALLO la mezcla"; tail -20 logs/43_mezcla_v3.log | tee -a "$LOG"; exit 1; }
grep -E "backlogs del profesor|casi duplicadas|sinteticos utiles|variante|ficha de rol|s2s_v3" \
  logs/43_mezcla_v3.log | tee -a "$LOG"

# ---------------------------------------------------------------- fase 2
di "fase 2a: mini-entrenamiento de control (abajo)"
DATOS=v3 "$PY" src/09_train_mt5.py --mini "$ABAJO_BASE" 8 192 5e-4 \
  > logs/44_mini_abajo_v3.log 2>&1
tail -3 logs/44_mini_abajo_v3.log | tee -a "$LOG"

di "fase 2b: entrenando abajo, historia -> tareas (~30 min)"
DATOS=v3 "$PY" -u src/09_train_mt5.py "$ABAJO_BASE" 16 192 5e-4 4 \
  > logs/44_train_abajo_v3.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/44_train_abajo_v3.log | tee -a "$LOG"
di "VRAM maxima abajo: $(grep -oE 'VRAM [0-9.]+ GB' logs/44_train_abajo_v3.log | sort -k2 -n | tail -1)"

di "fase 2c: entrenando arriba SIN ficha (~5 min)"
DATOS=v3a "$PY" -u src/09_train_mt5.py "$ARRIBA_BASE" 32 48 5e-4 4 \
  > logs/44_train_arriba_v3a.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/44_train_arriba_v3a.log | tee -a "$LOG"

di "fase 2d: entrenando arriba CON ficha de rol (~10 min)"
DATOS=v3r "$PY" -u src/09_train_mt5.py "$ARRIBA_BASE" 32 48 5e-4 4 \
  > logs/44_train_arriba_v3r.log 2>&1
grep -E "^  epoca|parada|TOTAL" logs/44_train_arriba_v3r.log | tee -a "$LOG"

for d in "$A_SIN" "$A_CON" "$ABAJO"; do
  [ -d "$d" ] || { di "FALLO: no existe $d"; ls -d models/*v3* 2>/dev/null | tee -a "$LOG"; exit 1; }
done
di "modelos listos. VRAM: $(vram)"

# ------------------------------------------------------- medidas, iguales para todos
export PBE_DEVICE=cuda PBE_CONJUNTO=no

medir() {          # etiqueta, modelo_arriba, modelo_abajo
  local etq="$1" arriba="$2" abajo="$3"
  di "--- midiendo $etq"
  export PBE_ARRIBA="$arriba" PBE_ABAJO="$abajo" PBE_SUFIJO="_$etq"
  "$PY" -u src/52_coherencia.py modelo > "logs/52_$etq.log" 2>&1 \
    && grep -E "ROLES|TAREAS|ATRIB|^ +(cuota|sin |roles dist|top|eco|ancladas|margen|acierto)" \
       "logs/52_$etq.log" | tee -a "$LOG" \
    || di "  fallo 52 en $etq"
  "$PY" -u src/57_f1_arista.py --modelo > "logs/57_$etq.log" 2>&1 \
    && tail -7 "logs/57_$etq.log" | tee -a "$LOG" \
    || di "  fallo 57 en $etq"
  unset PBE_SUFIJO
}

di "fase 3: linea base = modelo de produccion (30sec)"
medir "base" "$A_PROD" "$B_PROD"

di "fase 4: metodo A, sin ficha + validador de rol"
medir "v3a" "$A_SIN" "$ABAJO"
# 'ambos' compara contra los modelos de produccion en el mismo proceso y da el
# intervalo pareado sobre las 328 historias HUMANAS: el unico test que no viene
# del profesor
PBE_ARRIBA="$A_SIN" PBE_ABAJO="$ABAJO" PBE_SUFIJO="_v3a" \
  "$PY" -u src/49_eval_real.py ambos > logs/49_v3a.log 2>&1
grep -E "^##### |nuevo - viejo|gana el nuevo" logs/49_v3a.log | tee -a "$LOG"
PBE_ARRIBA="$A_SIN" PBE_ABAJO="$ABAJO" "$PY" -u src/45_leer_salidas.py nuevo \
  > logs/45_v3a.log 2>&1

di "fase 5: metodo B, con ficha + planificador de roles"
medir "v3r" "$A_CON" "$ABAJO"
PBE_ARRIBA="$A_CON" PBE_ABAJO="$ABAJO" PBE_SUFIJO="_v3r" \
  "$PY" -u src/49_eval_real.py ambos > logs/49_v3r.log 2>&1
grep -E "^##### |nuevo - viejo|gana el nuevo" logs/49_v3r.log | tee -a "$LOG"
PBE_ARRIBA="$A_CON" PBE_ABAJO="$ABAJO" \
  "$PY" -u src/45_leer_salidas.py nuevo > logs/45_v3r.log 2>&1

di "fase 6: ablacion de los filtros (metodo A) sobre descripciones de consumo"
PBE_ARRIBA="$A_SIN" PBE_ABAJO="$ABAJO" PBE_POB=consumo \
  "$PY" -u src/58_ablacion_filtros.py 12 > logs/58_v3.log 2>&1
tail -10 logs/58_v3.log | tee -a "$LOG"

di "fase 7: tabla de resultados"
"$PY" -u src/59_tabla_v3.py > logs/59_tabla.log 2>&1 \
  && cat logs/59_tabla.log | tee -a "$LOG" \
  || di "  fallo la tabla, mirar logs/59_tabla.log"

di "TERMINADO. Comparar logs/45_v3a.log y logs/45_v3r.log; tabla en entrega/ROLES_Y_COHERENCIA.md"
