#!/usr/bin/env bash
# Night chain: wait for the mT5-small run -> mini-train gate on mT5-base-pruned
# -> full mT5-base-pruned run only if the gate passes.
# Gate (fixed up front): final mini val loss finite and < 50 % of the initial one.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/PBE-Dataset/.venv/Scripts/python.exe"
MODEL="F:/Software Task generation Model/models/mt5-base-pruned"
LOG="$BASE/logs/noche.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
cd "$BASE"

echo "[$(date '+%F %T')] esperando a que termine mt5-small-pruned" >> "$LOG"
until grep -q "^TOTAL" logs/09_train_small_pruned.log; do sleep 30; done
echo "[$(date '+%F %T')] small terminado; mini-entrenamiento de base" >> "$LOG"

"$PY" src/09_train_mt5.py --mini "$MODEL" 4 192 7e-4 2>&1 | grep -v -i "warn\|Loading weights" > logs/09_mini_base_pruned.log

"$PY" - <<'EOF' >> "$LOG"
import json, math, sys
r = json.load(open(r"F:\Software Task generation Model\results\09_train_mt5-base-pruned_mini.json"))
fin, v0 = r["hist"][-1]["val_loss"], r["v0"]
ok = math.isfinite(fin) and fin < 0.5 * v0
print(f"  gate base: v0 {v0:.3f} -> final {fin:.3f}  {'PASA' if ok else 'NO PASA'}")
open(r"F:\Software Task generation Model\logs\gate_base.ok", "w").write("1" if ok else "0")
EOF

if [ "$(cat logs/gate_base.ok)" = "1" ]; then
  echo "[$(date '+%F %T')] lanzando mt5-base-pruned completo (bs4, lr 7e-4, <=12 epocas)" >> "$LOG"
  "$PY" src/09_train_mt5.py "$MODEL" 4 192 7e-4 12 2>&1 | grep --line-buffered -v -i "warn\|Loading weights" > logs/09_train_base_pruned.log
  echo "[$(date '+%F %T')] base terminado" >> "$LOG"
else
  echo "[$(date '+%F %T')] gate NO pasa: no se lanza base" >> "$LOG"
fi
