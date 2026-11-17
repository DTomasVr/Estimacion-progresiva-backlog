#!/usr/bin/env bash
# Noche del 25/09: destilacion v2 con la taxonomia ancha (211 sectores).
#
# La lectura del piloto v1 mostro que los sectores no entrenados caian al vecino
# mas cercano, asi que la apuesta es AMPLITUD: 211 sectores en vez de 30, con
# menos descripciones por sector.
#
# Se entrena desde los checkpoints de PRODUCCION (no desde los de v1), para que
# la comparacion v1 vs v2 sea limpia: misma base, distinto conjunto.
#
# Al final espera a que el scraper acabe y deja la senal para dormir la PC.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/noche_v2.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

di "esperando a que termine la generacion ancha"
while ! grep -qE "^listo:|Traceback" logs/42_sintetico_v2.log; do sleep 60; done
N=$(wc -l < data/synth/backlogs_qwen2.5_7b.jsonl)
di "generacion terminada: $N backlogs en total"
if [ "$N" -lt 900 ]; then di "ABORTO: solo $N backlogs"; touch logs/LISTO.flag; exit 1; fi

for M in qwen2.5:7b mistral:7b gemma2:9b llama3.1:8b; do
  curl -s -m 30 http://127.0.0.1:11434/api/generate -d "{\"model\": \"$M\", \"keep_alive\": 0}" > /dev/null 2>&1 || true
done
sleep 5
di "VRAM tras descargar Ollama: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

di "fase 3: mezclando"
"$PY" src/43_datos_mixtos.py > logs/43_mezcla_v2.log 2>&1 || { di "fallo la mezcla"; touch logs/LISTO.flag; exit 1; }
grep -E "duplicadas|utiles|s2s_mix" logs/43_mezcla_v2.log >> "$LOG"

di "fase 4a: entrenando abajo (mix v2)"
DATOS=mix "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_hist_pbe_v1 16 192 5e-4 4 \
  > logs/44_train_abajo_v2.log 2>&1
grep -E "^  epoca|parada|val loss inicial" logs/44_train_abajo_v2.log >> "$LOG"

di "fase 4b: entrenando arriba (mixa v2)"
DATOS=mixa "$PY" -u src/09_train_mt5.py models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1 32 48 5e-4 4 \
  > logs/44_train_arriba_v2.log 2>&1
grep -E "^  epoca|parada|val loss inicial" logs/44_train_arriba_v2.log >> "$LOG"

di "fase 5a: no-regresion y acuerdo con el profesor"
"$PY" src/44_eval_mixto.py abajo  > logs/44_eval_abajo_v2.log 2>&1
"$PY" src/44_eval_mixto.py arriba > logs/44_eval_arriba_v2.log 2>&1
grep -hE "real:|consumo:" logs/44_eval_abajo_v2.log logs/44_eval_arriba_v2.log >> "$LOG"

di "fase 5b: LA PRUEBA QUE IMPORTA - historias humanas reales (v2)"
PBE_DEVICE=cpu "$PY" src/49_eval_real.py ambos > logs/49_real_v2.log 2>&1
grep -E "^##### (viejo|nuevo):|nuevo - viejo|gana el nuevo" logs/49_real_v2.log >> "$LOG"

di "fase 5c: la misma prueba con la v1 de 30 sectores, para aislar el efecto de la amplitud"
PBE_DEVICE=cpu PBE_SUFIJO=_30sec \
  PBE_ARRIBA="$BASE/models/mt5-small-pruned_o2ofmt_30sec" \
  PBE_ABAJO="$BASE/models/mt5-small-pruned_hist_30sec" \
  "$PY" src/49_eval_real.py nuevo > logs/49_real_30sec.log 2>&1
grep -E "^##### nuevo:" logs/49_real_30sec.log >> "$LOG"

di "fase 5d: material de lectura"
"$PY" src/45_leer_salidas.py nuevo > logs/45_nuevo_v2.log 2>&1

di "esperando a que el scraper termine (tope 3 h)"
# pgrep no existe en este Git Bash: se cuenta el proceso via Windows
cuenta_scraper() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -like '*50_scraper_github*' | Measure-Object).Count" 2>/dev/null | tr -d ' '
}
ESPERA=0
while [ "$(cuenta_scraper)" != "0" ] && [ "$ESPERA" -lt 180 ]; do
  sleep 60; ESPERA=$((ESPERA+1))
done
di "espera al scraper terminada tras $ESPERA min"
di "scraper: $(wc -l < data/reales/github_issues.jsonl 2>/dev/null || echo 0) incidencias con descomposicion"
tail -4 logs/50_scraper.log >> "$LOG"

di "TERMINADO"
{ echo "destilacion v2 terminada $(date '+%F %T')"; tail -25 "$LOG"; } > logs/LISTO.flag
