#!/usr/bin/env bash
# El test humano REPETIDO con semilla fija, porque el ruido de generacion resulto
# ser del tamano del efecto.
#
# Medido: el mismo modelo de produccion puntuo 0,0122 menos al repetir la corrida
# (desvio medio por proyecto 0,0143, maximo 0,0260). Los efectos que estamos
# midiendo son -0,0276 y -0,0155, asi que sin fijar la semilla no se puede
# distinguir un metodo del otro.
#
# Con AGIX_SEED la generacion es determinista, y las dos ramas de cada corrida
# (produccion y candidato) comparten semilla, asi que la comparacion es pareada y
# reproducible. Se repite con tres semillas para no depender de un solo sorteo.
#
# Espera a que run_v3b.sh termine. NO EDITAR MIENTRAS CORRE.
set -u
BASE="/f/Software Task generation Model"
PY="/c/Users/User/Documents/Programacao/Agix-Dataset/.venv/Scripts/python.exe"
LOG="$BASE/logs/v3.log"
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 AGIX_DEVICE=cuda AGIX_CONJUNTO=no
cd "$BASE"
di() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

A_CON="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3r_agix_v1"
A_B="$BASE/models/mt5-small-pruned_o2o_agix_v1_o2ofmt_agix_v1_v3b_agix_v1"
ABAJO="$BASE/models/mt5-small-pruned_hist_agix_v1_v3_agix_v1"

di "esperando a que termine run_v3b.sh"
while ! grep -qE "SEGUIMIENTO TERMINADO|FALLO" "$LOG"; do sleep 45; done
sleep 10
di "=== test humano con semilla fija, 3 sorteos por candidato ==="

for s in 20260926 777 31337; do
  for par in "v3r|$A_CON" "v3b|$A_B"; do
    etq="${par%%|*}"; arriba="${par#*|}"
    [ -d "$arriba" ] || { di "  no existe $arriba, salto $etq"; continue; }
    AGIX_SEED="$s" AGIX_ARRIBA="$arriba" AGIX_ABAJO="$ABAJO" AGIX_SUFIJO="_${etq}_s$s" \
      "$PY" -u src/49_eval_real.py ambos > "logs/49_${etq}_s$s.log" 2>&1
    linea=$(grep -E "nuevo - viejo" "logs/49_${etq}_s$s.log" | head -1)
    viejo=$(grep -oE "viejo: sF1 conjunto [0-9.]+" "logs/49_${etq}_s$s.log" | tail -1)
    nuevo=$(grep -oE "nuevo: sF1 conjunto [0-9.]+" "logs/49_${etq}_s$s.log" | tail -1)
    di "semilla $s $etq | $viejo | $nuevo | $linea"
  done
done

di "resumen de las tres semillas"
"$PY" - <<'PYEOF' | tee -a "$LOG"
import glob, os, re
import pandas as pd
BASE = r"F:/Software Task generation Model"
filas = []
for f in sorted(glob.glob(f"{BASE}/results/49_real_v3*_s*.csv")):
    n = os.path.basename(f)[10:-4]          # v3r_s777
    etq, s = n.rsplit("_s", 1)
    d = pd.read_csv(f)
    t = d.pivot_table(index="proyecto", columns="variante", values="sF1")
    if not {"viejo", "nuevo"} <= set(t.columns):
        continue
    lim = t[~t.index.isin({"school", "supermarket"})]
    filas.append(dict(variante=etq, semilla=s,
                      viejo=t["viejo"].mean(), nuevo=t["nuevo"].mean(),
                      dif=(t["nuevo"] - t["viejo"]).mean(),
                      gana=f'{int((t["nuevo"] > t["viejo"]).sum())}/{len(t)}',
                      dif_limpios=(lim["nuevo"] - lim["viejo"]).mean(),
                      gana_limpios=f'{int((lim["nuevo"] > lim["viejo"]).sum())}/{len(lim)}'))
if filas:
    D = pd.DataFrame(filas)
    print(D.round(4).to_string(index=False))
    print("\nmedia de las tres semillas por variante:")
    print(D.groupby("variante")[["viejo", "nuevo", "dif", "dif_limpios"]].mean().round(4).to_string())
    D.to_csv(f"{BASE}/results/49_semillas.csv", index=False)
    print("\n-> results/49_semillas.csv")
else:
    print("sin resultados")
PYEOF

di "SEMILLAS TERMINADO"
