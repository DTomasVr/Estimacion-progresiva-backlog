"""Anade al artefacto v6 la calibracion de la banda POR TRAMO de tamano.

Son datos, no modelo: los residuos del total ya estan calculados en
results/curva_final/totales.csv. No hace falta reentrenar nada, asi que esto
evita una reconstruccion completa de 171_estimador_v6.py.

Deja 'residuos_proyecto' como estaba, por si algo lo lee, y anade
'residuos_por_tramo' y 'cortes_tramo', que son los que usa estimar_proyecto.
Al final comprueba que la banda sale y que es mas estrecha para los proyectos
grandes que para los pequenos, que es justo lo que no hacia antes.
"""
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
V6 = Path("models/estimador_pbe_v6.joblib")
CORTES_TRAMO = [20, 60, 200]
ETIQ = ["8-20", "21-60", "61-200", "mas de 200"]


def main():
    t0 = time.time()
    resp = V6.with_suffix(".joblib.antes_de_la_banda")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    p = dict(joblib.load(V6))
    tot = pd.read_csv("results/curva_final/totales.csv")
    res = np.log(tot.real.to_numpy() / tot.frio.to_numpy())
    gr = np.digitize(tot.n.to_numpy(), CORTES_TRAMO)
    p["residuos_proyecto"] = res
    p["residuos_por_tramo"] = {int(t): res[gr == t] for t in np.unique(gr)}
    p["cortes_tramo"] = CORTES_TRAMO
    for t, v in sorted(p["residuos_por_tramo"].items()):
        print(f"  tramo {ETIQ[t]:12s} n={len(v):3d}  "
              f"sd del residuo {v.std():.3f}")
    joblib.dump(p, V6, compress=3)
    print(f"escrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
