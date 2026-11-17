#!/usr/bin/env bash
# Comparacion CORRECTA de la reparacion del rol.
#
# El intento anterior salio +0,0000 con intervalo de ancho cero, y eso no era un
# resultado sino un fallo de diseno mio: AGIX_REPARAR_ROL es una variable del
# PROCESO, asi que en el modo "ambos" de 49_eval_real.py las dos ramas la
# recibian y generaban lo mismo.
#
# Aqui se corre el MISMO brazo (los modelos de produccion) en dos procesos
# distintos, con la misma semilla, cambiando solo la reparacion. La generacion es
# determinista con AGIX_SEED (comprobado: dos pasadas dan salida identica), asi
# que la comparacion sigue siendo pareada aunque no la calcule 49 por dentro.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 AGIX_DEVICE=cuda AGIX_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

di "esperando a que termine run_v3_reparar.sh"
while ! grep -qE "REPARAR TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== reparacion del rol, comparacion correcta (dos procesos) ==="

for s in 20260926 777; do
  AGIX_SEED="$s" AGIX_SUFIJO="_sinrep_s$s" "$PY" -u src/49_eval_real.py viejo \
    > "logs/49_sinrep_s$s.log" 2>&1
  AGIX_SEED="$s" AGIX_REPARAR_ROL=1 AGIX_SUFIJO="_conrep_s$s" "$PY" -u src/49_eval_real.py viejo \
    > "logs/49_conrep_s$s.log" 2>&1
  a=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_sinrep_s$s.log" | tail -1)
  b=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_conrep_s$s.log" | tail -1)
  di "semilla $s | SIN reparar $a | CON reparar $b"
done

"$PY" - <<'PYEOF' | tee -a "$LOG"
import glob, os, re
import pandas as pd
BASE = r"F:/Software Task generation Model"
filas = []
for s in ("20260926", "777"):
    pa, pb = f"{BASE}/results/49_real_sinrep_s{s}.csv", f"{BASE}/results/49_real_conrep_s{s}.csv"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    a = pd.read_csv(pa).set_index("proyecto").sF1
    b = pd.read_csv(pb).set_index("proyecto").sF1
    for p in a.index:
        filas.append(dict(semilla=s, proyecto=p, sin=a[p], con=b.get(p), dif=b.get(p) - a[p]))
if filas:
    D = pd.DataFrame(filas)
    print(D.round(4).to_string(index=False))
    print(f"\ndiferencia media de la reparacion: {D.dif.mean():+.4f}  "
          f"(cambia en {int((D.dif.abs() > 1e-6).sum())} de {len(D)} casos)")
    D.to_csv(f"{BASE}/results/49_reparacion.csv", index=False)
else:
    print("sin resultados")
PYEOF

di "REPARACION CORRECTA TERMINADA"
