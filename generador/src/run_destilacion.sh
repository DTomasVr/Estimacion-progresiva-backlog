#!/usr/bin/env bash
# Distillation pilot, phases 3 to 5. Run AFTER 42_generar_sintetico.py finishes.
#
#   fase 3  mezclar real + sintetico (train/val/test disjuntos por descripcion)
#   fase 4  continuar el entrenamiento de las DOS mitades desde los modelos de
#           produccion, no desde cero: es una adaptacion de dominio, no un
#           reentrenamiento
#   fase 5  no-regresion en los conjuntos reales + acuerdo en consumo + lectura
#
# Una mitad a la vez: dos entrenamientos en el mismo proceso desbordan los 8 GB
# de VRAM y Windows lo resuelve en silencio con RAM compartida, 10x mas lento.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/destilacion.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"

# descargar el LLM profesor de la VRAM: 5 GB ocupados harian desbordar el
# entrenamiento a RAM compartida sin dar error, 10x mas lento
for M in qwen2.5:7b mistral:7b gemma2:9b llama3.1:8b; do
  curl -s -m 30 http://127.0.0.1:11434/api/generate     -d "{\"model\": \"$M\", \"keep_alive\": 0}" > /dev/null 2>&1 || true
done
sleep 5
echo "[$(date '+%F %T')] VRAM tras descargar Ollama: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "$LOG"

echo "[$(date '+%F %T')] fase 3: mezclando" >> "$LOG"
"$PY" src/43_datos_mixtos.py > logs/43_mezcla.log 2>&1 || { echo "fallo la mezcla" >> "$LOG"; exit 1; }
tail -8 logs/43_mezcla.log >> "$LOG"

# --- mitad de abajo: historia -> tareas
echo "[$(date '+%F %T')] fase 4a: mini-entrenamiento de control (abajo)" >> "$LOG"
DATOS=mix "$PY" src/09_train_mt5.py --mini models/mt5-small-pruned_hist_pbe_v1 8 192 5e-4 \
  > logs/44_mini_abajo.log 2>&1
grep -E "val loss|epoca 30" logs/44_mini_abajo.log | tail -2 >> "$LOG"

echo "[$(date '+%F %T')] fase 4b: entrenando abajo (mix)" >> "$LOG"
DATOS=mix "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_hist_pbe_v1 16 192 5e-4 4 \
  > logs/44_train_abajo.log 2>&1
grep -E "^  epoca|parada" logs/44_train_abajo.log >> "$LOG"

# --- mitad de arriba: descripcion -> historia
echo "[$(date '+%F %T')] fase 4c: entrenando arriba (mixa)" >> "$LOG"
DATOS=mixa "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1 32 48 5e-4 4 \
  > logs/44_train_arriba.log 2>&1
grep -E "^  epoca|parada" logs/44_train_arriba.log >> "$LOG"

# --- fase 5
echo "[$(date '+%F %T')] fase 5: evaluando" >> "$LOG"
"$PY" src/44_eval_mixto.py abajo  > logs/44_eval_abajo.log 2>&1
"$PY" src/44_eval_mixto.py arriba > logs/44_eval_arriba.log 2>&1
grep -E "real:|consumo:" logs/44_eval_abajo.log logs/44_eval_arriba.log >> "$LOG"

"$PY" src/45_leer_salidas.py nuevo > logs/45_nuevo.log 2>&1
echo "[$(date '+%F %T')] TERMINADO. Lee logs/45_nuevo.log contra logs/45_viejo.log" >> "$LOG"
